"""
MailMind FastAPI Application

Main application entry point for the MailMind RAG system.
Provides REST API endpoints for Gmail thread indexing and retrieval.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import os
import json
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from urllib.parse import urlparse, parse_qs

# Load environment variables from .env file (override any stale shell exports)
load_dotenv(override=True)

# Silence noisy deprecation warnings emitted via loguru when some dependencies
# import `fastembed` (e.g. qdrant-client).
try:  # pragma: no cover
    from loguru import logger as _loguru_logger

    _loguru_logger.disable("fastembed")
except Exception:  # pragma: no cover
    pass

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, status, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import uvicorn

# Import core components
from models import Base, SyncState, Thread, Message, Attachment
from service.gmail_client import GmailClient
from core.attachment_processor import AttachmentProcessor, AttachmentProcessorWorker
from core.sliding_context import SlidingContextProcessor
from core.action_extractor import ActionItemExtractor
from core.relationship_mapper import RelationshipMapper
from core.embedding_pipeline import EmbeddingPipeline, ExtractedTask, ExtractedEntity
from core.vector_store import VectorStore
from core.draft_reply_agent import DraftReplyAgent, ThreadContext, UserWritingStyle, ReplyType
from core.groq_client import get_async_groq_client

# Database and vector store imports (to be implemented)
# from database import get_db, engine
# from vector_store import QdrantManager

logger = logging.getLogger(__name__)

# Security
security = HTTPBearer()


# Pydantic models for API
class SyncRequest(BaseModel):
    user_id: str = Field(..., description="User ID for Gmail account")
    max_threads: int = Field(default=50, description="Maximum threads to sync")


class SyncResponse(BaseModel):
    sync_id: str
    status: str
    threads_processed: int
    messages_processed: int
    attachments_processed: int
    errors: List[str]


class SearchRequest(BaseModel):
    user_id: str
    query: str
    limit: int = Field(default=10, description="Maximum results to return")
    filters: Optional[Dict[str, Any]] = Field(default=None)


class SearchResponse(BaseModel):
    results: List[Dict[str, Any]]
    total_found: int
    search_time_ms: int


class ThreadResponse(BaseModel):
    thread_id: str
    subject: str
    message_count: int
    participant_emails: List[str]
    first_message_date: datetime
    last_message_date: datetime
    aggregated_content: str
    detected_tasks: List[str]
    related_threads: List[str]

class ThreadMessageResponse(BaseModel):
    message_id: str
    from_email: str
    to_emails: List[str]
    cc_emails: List[str] = Field(default_factory=list)
    body_text: str
    gmail_date: datetime


class ThreadDetailResponse(ThreadResponse):
    messages: List[ThreadMessageResponse] = Field(default_factory=list)
    action_items: List[Dict[str, Any]] = Field(default_factory=list)
    referenced_projects: List[str] = Field(default_factory=list)
    referenced_urls: List[str] = Field(default_factory=list)
    referenced_invoices: List[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    version: str
    components: Dict[str, str]


class AssistantChatRequest(BaseModel):
    user_id: str
    thread_id: str
    message: str
    instructions: Optional[str] = Field(default=None, description="Extra assistant instructions/prompt.")
    conversation: List[Dict[str, str]] = Field(default_factory=list, description="Prior conversation messages (role/content).")


class AssistantChatResponse(BaseModel):
    answer: str
    suggested_actions: List[Dict[str, Any]] = Field(default_factory=list)


# Global application state
class MailMindState:
    def __init__(self):
        # Gmail state is per-user (use the Gmail address as user_id)
        self.gmail_clients: Dict[str, GmailClient] = {}
        self.gmail_oauth_states: Dict[str, str] = {}  # oauth_state -> user_id
        self.gmail_credentials_path: Optional[str] = None
        self.gmail_token_dir: Optional[str] = None
        self.attachment_processor: Optional[AttachmentProcessor] = None
        self.sliding_context: Optional[SlidingContextProcessor] = None
        self.action_extractor: Optional[ActionItemExtractor] = None
        self.relationship_mapper: Optional[RelationshipMapper] = None
        self.attachment_worker: Optional[AttachmentProcessorWorker] = None


state = MailMindState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting MailMind application...")
    
    # Initialize components
    await initialize_components()
    
    yield
    
    # Shutdown
    logger.info("Shutting down MailMind application...")


async def initialize_components():
    """Initialize all core components."""
    try:
        # Get configuration from environment variables
        groq_api_key = os.getenv("GROQ_API_KEY")
        openai_api_key = os.getenv("OPENAI_API_KEY")
        gmail_credentials_path = os.getenv("GMAIL_CREDENTIALS_PATH", "credentials.json")
        gmail_token_dir = os.getenv("GMAIL_TOKEN_DIR", "tokens")
        
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable is required")

        state.gmail_credentials_path = gmail_credentials_path
        state.gmail_token_dir = gmail_token_dir
        
        # Initialize attachment processor
        state.attachment_processor = AttachmentProcessor()
        state.attachment_worker = AttachmentProcessorWorker(state.attachment_processor)
        
        # Initialize AI components
        state.sliding_context = SlidingContextProcessor()
        state.action_extractor = ActionItemExtractor()
        state.relationship_mapper = RelationshipMapper()
        
        # Initialize intelligence pipeline
        state.embedding_pipeline = EmbeddingPipeline(openai_api_key)
        
        # Initialize draft reply agent (uses Groq)
        state.draft_reply_agent = DraftReplyAgent()
        
        # Initialize vector store (will be initialized on demand)
        state.vector_store = None
        
        # Store API keys for later use
        state.groq_api_key = groq_api_key
        state.openai_api_key = openai_api_key
        
        logger.info("All components initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize components: {str(e)}")
        raise


async def _ensure_vector_store() -> VectorStore:
    """Initialize vector store once and return it."""
    if hasattr(state, "vector_store") and state.vector_store:
        return state.vector_store

    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    state.vector_store = VectorStore(qdrant_url, qdrant_api_key or "", state.openai_api_key)
    ok = await state.vector_store.initialize_collection()
    if not ok:
        raise HTTPException(
            status_code=503,
            detail="Vector store unavailable. Ensure Qdrant is running and QDRANT_URL is correct (e.g. http://localhost:6333).",
        )
    return state.vector_store


def _get_gmail_client_for_user(user_id: str) -> GmailClient:
    if not state.gmail_credentials_path or not state.gmail_token_dir:
        raise RuntimeError("Gmail configuration not initialized")

    safe_user_id = user_id.replace("/", "_")
    token_path = os.path.join(state.gmail_token_dir, f"{safe_user_id}.json")

    if user_id not in state.gmail_clients:
        state.gmail_clients[user_id] = GmailClient(state.gmail_credentials_path, token_path)
    return state.gmail_clients[user_id]


# Create FastAPI app
app = FastAPI(
    title="MailMind API",
    description="RAG system for Gmail thread indexing and semantic retrieval",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Dependency for user authentication
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Extract user ID from JWT token (simplified for demo)."""
    # In production, implement proper JWT validation
    token = credentials.credentials
    # For demo, assume token is the user_id
    return token


# Health check endpoint
@app.get("/")
async def root():
    return {
        "name": "MailMind API",
        "status": "ok",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check application health and component status."""
    components = {}
    
    # Check Gmail client
    try:
        components["gmail"] = "ready" if state.gmail_credentials_path else "not_initialized"
    except:
        components["gmail"] = "error"
    
    # Check other components
    components["attachment_processor"] = "initialized" if state.attachment_processor else "not_initialized"
    components["sliding_context"] = "initialized" if state.sliding_context else "not_initialized"
    components["action_extractor"] = "initialized" if state.action_extractor else "not_initialized"
    components["relationship_mapper"] = "initialized" if state.relationship_mapper else "not_initialized"
    
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        components=components
    )


# Authentication endpoint
@app.post("/auth/gmail")
async def authenticate_gmail(request: dict):
    """
    Start Gmail OAuth for a specific user_id (use your email address as user_id).
    Returns an authorization URL to open in the browser.
    """
    try:
        user_id = request.get("user_id")
        if not user_id:
            raise HTTPException(status_code=400, detail="user_id is required")

        gmail_client = _get_gmail_client_for_user(user_id)

        # Already authenticated? Return profile.
        if await gmail_client.authenticate():
            profile = await gmail_client.get_user_profile()
            return {
                "status": "authenticated",
                "user_id": user_id,
                "user_email": profile["email_address"],
                "messages_total": profile["messages_total"],
                "threads_total": profile["threads_total"],
                "history_id": profile.get("history_id"),
            }

        redirect_uri = os.getenv("GMAIL_OAUTH_REDIRECT_URI", "http://localhost:8000/auth/gmail/callback")
        auth_url = await gmail_client.initiate_oauth_flow(redirect_uri=redirect_uri)

        # Keep a server-side mapping so the callback can resolve user_id.
        oauth_state = parse_qs(urlparse(auth_url).query).get("state", [None])[0]
        if oauth_state:
            state.gmail_oauth_states[oauth_state] = user_id

        return {
            "status": "authorization_required",
            "user_id": user_id,
            "auth_url": auth_url,
            "redirect_uri": redirect_uri,
        }
            
    except Exception as e:
        logger.error(f"Gmail authentication error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/auth/gmail/callback")
async def gmail_oauth_callback(code: str, oauth_state: str = Query(alias="state")):
    """OAuth2 callback endpoint for Gmail authentication."""
    try:
        user_id = state.gmail_oauth_states.get(oauth_state)
        if not user_id:
            raise HTTPException(status_code=400, detail="Unknown or expired OAuth state. Restart /auth/gmail.")

        gmail_client = _get_gmail_client_for_user(user_id)
        ok = await gmail_client.complete_oauth_flow(auth_code=code, state=oauth_state)
        if not ok:
            raise HTTPException(status_code=401, detail="Gmail OAuth completion failed")

        profile = await gmail_client.get_user_profile()
        return {
            "status": "authenticated",
            "user_id": user_id,
            "user_email": profile["email_address"],
            "messages_total": profile["messages_total"],
            "threads_total": profile["threads_total"],
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Gmail OAuth callback error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/auth/gmail/status")
async def gmail_auth_status(user_id: str):
    """Check whether the given user_id has a valid Gmail token configured."""
    try:
        if not user_id:
            raise HTTPException(status_code=400, detail="user_id is required")

        gmail_client = _get_gmail_client_for_user(user_id)
        authed = await gmail_client.authenticate()
        if not authed:
            return {"status": "not_authenticated", "user_id": user_id}

        profile = await gmail_client.get_user_profile()
        return {"status": "authenticated", "user_id": user_id, "user_email": profile["email_address"]}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Gmail auth status error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Sync endpoints
@app.post("/sync/gmail", response_model=SyncResponse)
async def sync_gmail(
    request: dict,
    background_tasks: BackgroundTasks
):
    """Start Gmail synchronization process."""
    try:
        user_id = request.get("user_id")
        max_threads = request.get("max_threads", 50)
        
        if not user_id:
            raise HTTPException(status_code=400, detail="user_id is required")

        gmail_client = _get_gmail_client_for_user(user_id)
        if not await gmail_client.authenticate():
            raise HTTPException(status_code=401, detail="Gmail not authenticated. Call POST /auth/gmail first.")

        sync_id = f"sync_{user_id}_{datetime.utcnow().timestamp()}"
        
        # Add background task for sync
        background_tasks.add_task(
            perform_gmail_sync,
            sync_id=sync_id,
            user_id=user_id,
            max_threads=max_threads
        )
        
        return SyncResponse(
            sync_id=sync_id,
            status="started",
            threads_processed=0,
            messages_processed=0,
            attachments_processed=0,
            errors=[]
        )
        
    except Exception as e:
        logger.error(f"Sync start error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


async def perform_gmail_sync(sync_id: str, user_id: str, max_threads: int):
    """Background task for Gmail synchronization."""
    try:
        logger.info(f"Starting Gmail sync {sync_id} for user {user_id}")

        gmail_client = _get_gmail_client_for_user(user_id)
        if not await gmail_client.authenticate():
            logger.error(f"Gmail sync {sync_id} aborted: user {user_id} not authenticated")
            return

        # Initialize vector store once for this sync (if embeddings are configured).
        if (not hasattr(state, "vector_store") or not state.vector_store) and state.openai_api_key:
            qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
            qdrant_api_key = os.getenv("QDRANT_API_KEY")
            state.vector_store = VectorStore(qdrant_url, qdrant_api_key or "", state.openai_api_key)
            await state.vector_store.initialize_collection()
        
        # Get last sync state
        # last_sync = await get_last_sync_state(user_id)
        last_sync = None  # Simplified for demo
        
        # Fetch threads since last sync
        since_date = last_sync.last_synced_at if last_sync else datetime.utcnow() - timedelta(days=30)
        
        threads_processed = 0
        messages_processed = 0
        attachments_processed = 0
        errors = []
        
        async for gmail_thread in gmail_client.get_threads_since(since_date, max_threads):
            try:
                # Process thread
                thread_result = await process_thread(gmail_thread, user_id, gmail_client)
                
                threads_processed += 1
                messages_processed += thread_result["messages_processed"]
                attachments_processed += thread_result["attachments_processed"]
                
            except Exception as e:
                error_msg = f"Error processing thread {gmail_thread.thread_id}: {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)
        
        # Update sync state
        # await update_sync_state(user_id, datetime.utcnow())
        
        logger.info(f"Completed Gmail sync {sync_id}: {threads_processed} threads, {messages_processed} messages, {attachments_processed} attachments")
        
    except Exception as e:
        logger.error(f"Gmail sync {sync_id} failed: {str(e)}")


async def process_thread(gmail_thread, user_id: str, gmail_client: GmailClient) -> Dict[str, int]:
    """Process a single Gmail thread."""
    messages_processed = len(gmail_thread.messages)
    attachments_processed = 0
    
    # Convert messages to dict format
    messages = []
    for msg in gmail_thread.messages:
        message_data = {
            "message_id": msg.message_id,
            "thread_id": msg.thread_id,
            "subject": msg.subject,
            "from_email": msg.from_email,
            "to_emails": msg.to_emails,
            "cc_emails": msg.cc_emails,
            "bcc_emails": msg.bcc_emails,
            "body_text": msg.body_text,
            "body_html": msg.body_html,
            "gmail_date": msg.gmail_date,
            "internal_date": msg.internal_date,
            "attachments": msg.attachments
        }
        messages.append(message_data)
        
        # Process attachments
        for attachment in msg.attachments:
            try:
                # Download attachment content
                content = await gmail_client.download_attachment(
                    msg.message_id, 
                    attachment["attachment_id"]
                )
                
                # Extract text from attachment
                result = await state.attachment_processor.extract_text(
                    content,
                    attachment["filename"],
                    attachment["mime_type"]
                )
                
                if result["status"] == "completed":
                    attachments_processed += 1
                    
            except Exception as e:
                logger.warning(f"Failed to process attachment {attachment['filename']}: {str(e)}")
    
    # Apply sliding context
    context_result = await state.sliding_context.process_thread_content(messages)
    
    # Extract action items
    action_items = await state.action_extractor.extract_action_items(
        context_result["processed_content"],
        messages,
        gmail_thread.participant_emails
    )
    
    # Extract references for relationship mapping
    references = await state.relationship_mapper.extract_thread_references(
        context_result["processed_content"],
        messages
    )

    # Store in vector store (enables /search).
    try:
        vector_store = await _ensure_vector_store()
        thread_metadata = {
            "subject": gmail_thread.subject,
            "participant_emails": gmail_thread.participant_emails,
            "message_count": len(messages),
            "first_message_date": messages[0]["gmail_date"].isoformat() if messages else None,
            "last_message_date": messages[-1]["gmail_date"].isoformat() if messages else None,
            "has_attachments": attachments_processed > 0,
            "action_items": action_items,
            "referenced_projects": references.get("projects", []),
            "referenced_urls": references.get("urls", []),
            "referenced_invoices": references.get("invoices", []),
            "intelligence_metadata": {
                "has_tasks": bool(action_items),
                "has_entities": bool(references.get("projects") or references.get("invoices") or references.get("urls")),
            },
        }
        await vector_store.store_thread(
            thread_id=gmail_thread.thread_id,
            user_id=user_id,
            thread_content=context_result["processed_content"],
            metadata=thread_metadata,
        )
    except Exception as e:
        logger.warning(f"Failed to store thread {gmail_thread.thread_id} in vector store: {str(e)}")
    
    # Store in database (simplified for demo)
    # await store_thread_in_db(thread_data, context_result, action_items, references)
    
    return {
        "messages_processed": messages_processed,
        "attachments_processed": attachments_processed
    }


# Search endpoints
@app.post("/search", response_model=SearchResponse)
async def search_threads(request: SearchRequest):
    """Search threads using intelligent semantic similarity with self-correction."""
    try:
        user_id = request.user_id
        query = request.query
        limit = request.limit
        filters = request.filters
        
        if not user_id or not query:
            raise HTTPException(status_code=400, detail="user_id and query are required")
        start_time = datetime.utcnow()
        
        await _ensure_vector_store()
        
        # Detect query intent for self-correction
        query_intent = await _detect_query_intent(query)
        logger.info(f"Detected query intent: {query_intent}")
        
        # Perform intelligent search based on intent
        if query_intent == "tasks":
            results = await _search_tasks(request, user_id, query_intent)
        elif query_intent == "entities":
            results = await _search_entities(request, user_id, query_intent)
        else:
            results = await _search_semantic(request, user_id)
        
        search_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
        
        return SearchResponse(
            results=results,
            total_found=len(results),
            search_time_ms=search_time
        )
        
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


async def _detect_query_intent(query: str) -> str:
    """Detect the intent of the user's query for self-correction."""
    query_lower = query.lower()
    
    # Task-related queries
    task_keywords = [
        "task", "tasks", "action", "actions", "item", "items",
        "todo", "to-do", "deadline", "due", "assign", "assigned",
        "complete", "finish", "pending", "overdue", "reminder"
    ]
    
    # Entity-related queries
    entity_keywords = [
        "project", "projects", "invoice", "invoices", "jira", "ticket",
        "tickets", "meeting", "meetings", "document", "documents"
    ]
    
    # Check for task intent
    if any(keyword in query_lower for keyword in task_keywords):
        return "tasks"
    
    # Check for entity intent
    if any(keyword in query_lower for keyword in entity_keywords):
        return "entities"
    
    # Default to semantic search
    return "semantic"


async def _search_tasks(request: SearchRequest, user_id: str, intent: str) -> List[Dict[str, Any]]:
    """Search for threads with action items, prioritizing task metadata."""
    try:
        # Qdrant `search()` requires a query vector; for metadata-only queries use `scroll()`.
        points, _next_offset = state.vector_store.client.scroll(
            collection_name="mailmind_threads",
            scroll_filter={
                "must": [
                    {"key": "user_id", "match": {"value": user_id}},
                    {"key": "intelligence_metadata.has_tasks", "match": {"value": True}},
                ]
            },
            limit=request.limit,
            with_payload=True,
        )
        
        # Convert to response format
        formatted_results = []
        for point in points:
            payload = point.payload or {}
            action_items = payload.get("action_items", [])
            
            # Filter action items based on query
            relevant_tasks = []
            for task in action_items:
                task_text = task.get("task_text", "").lower()
                if any(word in task_text for word in request.query.lower().split()):
                    relevant_tasks.append(task)
            
            if relevant_tasks or not any(word in request.query.lower().split() for word in ["what", "show", "list"]):
                formatted_results.append({
                    "thread_id": payload["thread_id"],
                    "subject": payload["subject"],
                    "score": None,
                    "snippet": _extract_task_snippet(relevant_tasks or action_items),
                    "participants": payload["participant_emails"],
                    "date": payload["last_message_date"],
                    "action_items": relevant_tasks or action_items[:3],  # Show relevant or first 3 tasks
                    "total_tasks": len(action_items),
                    "search_type": "task_search"
                })
        
        return formatted_results
        
    except Exception as e:
        logger.error(f"Task search failed: {str(e)}")
        return []


async def _search_entities(request: SearchRequest, user_id: str, intent: str) -> List[Dict[str, Any]]:
    """Search for threads with specific entities."""
    try:
        # Extract potential entity values from query
        entity_values = _extract_entity_values(request.query)
        
        must_conditions = [{"key": "user_id", "match": {"value": user_id}}]
        if entity_values:
            must_conditions.append({"key": "entities.entity_value", "match": {"any": entity_values}})
        else:
            must_conditions.append({"key": "intelligence_metadata.has_entities", "match": {"value": True}})

        points, _next_offset = state.vector_store.client.scroll(
            collection_name="mailmind_threads",
            scroll_filter={"must": must_conditions},
            limit=request.limit,
            with_payload=True,
        )
        
        # Convert to response format
        formatted_results = []
        for point in points:
            payload = point.payload or {}
            entities = payload.get("entities", [])
            
            # Filter entities based on query
            relevant_entities = []
            for entity in entities:
                entity_value = entity.get("entity_value", "").lower()
                entity_type = entity.get("entity_type", "").lower()
                
                if (entity_value in request.query.lower() or 
                    entity_type in request.query.lower()):
                    relevant_entities.append(entity)
            
            formatted_results.append({
                "thread_id": payload["thread_id"],
                "subject": payload["subject"],
                "score": None,
                "snippet": _extract_entity_snippet(relevant_entities or entities),
                "participants": payload["participant_emails"],
                "date": payload["last_message_date"],
                "entities": relevant_entities or entities[:5],  # Show relevant or first 5 entities
                "total_entities": len(entities),
                "search_type": "entity_search"
            })
        
        return formatted_results
        
    except Exception as e:
        logger.error(f"Entity search failed: {str(e)}")
        return []


async def _search_semantic(request: SearchRequest, user_id: str) -> List[Dict[str, Any]]:
    """Perform standard semantic search."""
    try:
        # Generate query embedding
        query_embedding = await state.vector_store.embed_thread(request.query)
        
        # Search in Qdrant
        search_results = await state.vector_store.search_similar_threads(
            query=request.query,
            user_id=user_id,
            limit=request.limit,
            filters=request.filters,
            score_threshold=0.7
        )
        
        # Convert to response format
        formatted_results = []
        for result in search_results:
            formatted_results.append({
                "thread_id": result["thread_id"],
                "subject": result["subject"],
                "score": result["score"],
                "snippet": result.get("snippet", ""),
                "participants": result["participant_emails"],
                "date": result["last_message_date"],
                "action_items": result.get("detected_tasks", []),
                "entities": result.get("referenced_projects", []),
                "search_type": "semantic_search"
            })
        
        return formatted_results
        
    except Exception as e:
        logger.error(f"Semantic search failed: {str(e)}")
        return []


def _extract_task_snippet(tasks: List[Dict[str, Any]]) -> str:
    """Extract a snippet from task list."""
    if not tasks:
        return "No tasks found"
    
    task_texts = [task.get("task_text", "") for task in tasks[:3]]
    return f"Tasks: {'; '.join(task_texts)}"


def _extract_entity_snippet(entities: List[Dict[str, Any]]) -> str:
    """Extract a snippet from entity list."""
    if not entities:
        return "No entities found"
    
    entity_texts = []
    for entity in entities[:5]:
        entity_type = entity.get("entity_type", "")
        entity_value = entity.get("entity_value", "")
        entity_texts.append(f"{entity_type}: {entity_value}")
    
    return f"Entities: {'; '.join(entity_texts)}"


def _extract_entity_values(query: str) -> List[str]:
    """Extract potential entity values from query."""
    # Simple extraction for common patterns
    import re
    
    # Project codes (PROJ-123)
    project_pattern = r'\b[A-Z]{2,}-\d{3,}\b'
    projects = re.findall(project_pattern, query.upper())
    
    # Invoice numbers
    invoice_pattern = r'\b(invoice|inv|receipt|po)\s*#?\s*\d{4,}\b'
    invoices = re.findall(invoice_pattern, query, re.IGNORECASE)
    
    # JIRA tickets
    jira_pattern = r'\b[A-Z]+-\d{3,}\b'
    jira_tickets = re.findall(jira_pattern, query.upper())
    
    return projects + invoices + jira_tickets


@app.post("/draft-reply")
async def generate_draft_reply(request: dict):
    """Generate a professional email reply draft."""
    try:
        thread_id = request.get("thread_id")
        user_id = request.get("user_id")
        reply_type = request.get("reply_type")
        custom_instructions = request.get("custom_instructions")
        
        if not thread_id or not user_id:
            raise HTTPException(status_code=400, detail="thread_id and user_id are required")
        # Initialize draft reply agent if not available
        if not hasattr(state, 'draft_reply_agent') or not state.draft_reply_agent:
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                raise HTTPException(status_code=500, detail="OpenAI API key not configured")
            
            state.draft_reply_agent = DraftReplyAgent(AsyncOpenAI(api_key=openai_api_key))
        
        # Get thread data (mock for now, would come from database)
        thread_data = await _get_thread_data_for_reply(thread_id, user_id)
        
        if not thread_data:
            raise HTTPException(status_code=404, detail="Thread not found")
        
        # Create thread context
        context = ThreadContext(
            thread_id=thread_id,
            subject=thread_data["subject"],
            participants=thread_data["participant_emails"],
            messages=thread_data["messages"],
            action_items=thread_data.get("action_items", []),
            entities=thread_data.get("entities", []),
            last_message_sender=thread_data["last_message_sender"],
            last_message_content=thread_data["last_message_content"],
            user_writing_style=thread_data.get("user_writing_style")
        )
        
        # Convert reply type string to enum
        reply_type_enum = None
        if reply_type:
            try:
                reply_type_enum = ReplyType(reply_type)
            except ValueError:
                logger.warning(f"Invalid reply type: {reply_type}")
        
        # Generate the reply
        logger.info(f"Generating draft reply for thread {thread_id}")
        generated_reply = await state.draft_reply_agent.generate_reply(
            context=context,
            reply_type=reply_type_enum,
            custom_instructions=custom_instructions
        )
        
        return {
            "thread_id": thread_id,
            "reply": {
                "subject": generated_reply.subject,
                "greeting": generated_reply.greeting,
                "body": generated_reply.body,
                "closing": generated_reply.closing,
                "signature": generated_reply.signature,
                "full_email": f"{generated_reply.greeting}\n\n{generated_reply.body}\n\n{generated_reply.closing}\n{generated_reply.signature or ''}",
                "action_items_addressed": generated_reply.action_items_addressed,
                "entities_referenced": generated_reply.entities_referenced,
                "confidence_score": generated_reply.confidence_score,
                "tone": generated_reply.tone.value,
                "estimated_reading_time": generated_reply.estimated_reading_time,
                "word_count": generated_reply.word_count
            },
            "metadata": {
                "thread_subject": thread_data["subject"],
                "last_message_sender": thread_data["last_message_sender"],
                "thread_participants": thread_data["participant_emails"],
                "generated_at": datetime.utcnow().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Draft reply generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/threads/{thread_id}", response_model=ThreadDetailResponse)
async def get_thread(thread_id: str, user_id: str):
    """Get detailed thread information."""
    try:
        if not user_id:
            raise HTTPException(status_code=400, detail="user_id is required")
        gmail_client = _get_gmail_client_for_user(user_id)
        if not await gmail_client.authenticate():
            raise HTTPException(status_code=401, detail="Gmail not authenticated. Call POST /auth/gmail first.")

        gmail_thread = await gmail_client.get_thread(thread_id)
        messages_sorted = sorted(gmail_thread.messages, key=lambda m: m.gmail_date)

        first_date = messages_sorted[0].gmail_date if messages_sorted else datetime.utcnow()
        last_date = messages_sorted[-1].gmail_date if messages_sorted else first_date

        aggregated_content = "\n\n".join([m.body_text for m in messages_sorted if m.body_text])

        # Best-effort enrichment from vector store (if indexed)
        action_items: List[Dict[str, Any]] = []
        referenced_projects: List[str] = []
        referenced_urls: List[str] = []
        referenced_invoices: List[str] = []

        if hasattr(state, "vector_store") and state.vector_store:
            try:
                points, _ = state.vector_store.client.scroll(
                    collection_name=VectorStore.COLLECTION_NAME,
                    scroll_filter={
                        "must": [
                            {"key": "user_id", "match": {"value": user_id}},
                            {"key": "thread_id", "match": {"value": thread_id}},
                        ]
                    },
                    limit=1,
                    with_payload=True,
                )
                if points:
                    payload = points[0].payload or {}
                    action_items = payload.get("action_items", []) or []
                    referenced_projects = payload.get("referenced_projects", []) or []
                    referenced_urls = payload.get("referenced_urls", []) or []
                    referenced_invoices = payload.get("referenced_invoices", []) or []
            except Exception as e:
                logger.debug(f"Vector enrichment failed for thread {thread_id}: {str(e)}")

        thread_data = {
            "thread_id": thread_id,
            "subject": gmail_thread.subject or "No Subject",
            "message_count": len(messages_sorted),
            "participant_emails": gmail_thread.participant_emails,
            "first_message_date": first_date,
            "last_message_date": last_date,
            "aggregated_content": aggregated_content,
            "detected_tasks": [t.get("task_text") for t in action_items if isinstance(t, dict) and t.get("task_text")],
            "related_threads": [],
            "messages": [
                {
                    "message_id": m.message_id,
                    "from_email": m.from_email,
                    "to_emails": m.to_emails,
                    "cc_emails": m.cc_emails,
                    "body_text": m.body_text,
                    "gmail_date": m.gmail_date,
                }
                for m in messages_sorted
            ],
            "action_items": action_items,
            "referenced_projects": referenced_projects,
            "referenced_urls": referenced_urls,
            "referenced_invoices": referenced_invoices,
        }

        return ThreadDetailResponse(**thread_data)
        
    except Exception as e:
        logger.error(f"Get thread error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/assistant/chat", response_model=AssistantChatResponse)
async def assistant_chat(payload: AssistantChatRequest):
    """Chat with an AI assistant grounded in a specific Gmail thread."""
    try:
        if not payload.user_id:
            raise HTTPException(status_code=400, detail="user_id is required")
        if not payload.thread_id:
            raise HTTPException(status_code=400, detail="thread_id is required")
        if not payload.message:
            raise HTTPException(status_code=400, detail="message is required")

        gmail_client = _get_gmail_client_for_user(payload.user_id)
        if not await gmail_client.authenticate():
            raise HTTPException(status_code=401, detail="Gmail not authenticated. Call POST /auth/gmail first.")

        gmail_thread = await gmail_client.get_thread(payload.thread_id)
        messages_sorted = sorted(gmail_thread.messages, key=lambda m: m.gmail_date)

        # Build a compact context block (avoid huge payloads)
        context_messages = []
        for m in messages_sorted[-10:]:
            body = (m.body_text or "").strip()
            if len(body) > 4000:
                body = body[:4000] + "…"
            context_messages.append(
                f"From: {m.from_email}\nDate: {m.gmail_date}\nBody:\n{body}"
            )

        enrichment = {"action_items": [], "referenced_projects": [], "referenced_urls": [], "referenced_invoices": []}
        if hasattr(state, "vector_store") and state.vector_store:
            try:
                points, _ = state.vector_store.client.scroll(
                    collection_name=VectorStore.COLLECTION_NAME,
                    scroll_filter={
                        "must": [
                            {"key": "user_id", "match": {"value": payload.user_id}},
                            {"key": "thread_id", "match": {"value": payload.thread_id}},
                        ]
                    },
                    limit=1,
                    with_payload=True,
                )
                if points:
                    p = points[0].payload or {}
                    enrichment["action_items"] = p.get("action_items", []) or []
                    enrichment["referenced_projects"] = p.get("referenced_projects", []) or []
                    enrichment["referenced_urls"] = p.get("referenced_urls", []) or []
                    enrichment["referenced_invoices"] = p.get("referenced_invoices", []) or []
            except Exception as e:
                logger.debug(f"Assistant enrichment failed for thread {payload.thread_id}: {str(e)}")

        context_block = (
            f"THREAD SUBJECT: {gmail_thread.subject or 'No Subject'}\n"
            f"PARTICIPANTS: {', '.join(gmail_thread.participant_emails)}\n\n"
            f"KNOWN ACTION ITEMS (may be empty): {json.dumps(enrichment['action_items'])}\n"
            f"REFERENCED PROJECTS: {json.dumps(enrichment['referenced_projects'])}\n"
            f"REFERENCED URLS: {json.dumps(enrichment['referenced_urls'])}\n"
            f"REFERENCED INVOICES: {json.dumps(enrichment['referenced_invoices'])}\n\n"
            f"LATEST MESSAGES:\n\n" + "\n\n---\n\n".join(context_messages)
        )

        extra_instructions = (payload.instructions or "").strip()
        system_prompt = (
            "You are MailMind, an assistant for understanding and operating on a Gmail thread.\n"
            "Rules:\n"
            "- Ground answers in the provided thread context.\n"
            "- If the user asks to do an operation (reply/follow-up/summarize/extract tasks), explain what to do and provide the content.\n"
            "- You cannot actually send email or modify Gmail state unless an explicit API exists; be honest.\n"
            "- Keep responses concise and actionable.\n"
        )
        if extra_instructions:
            system_prompt = system_prompt + "\nExtra instructions from the user:\n" + extra_instructions + "\n"

        # Normalize conversation (role/content only)
        convo = []
        for m in payload.conversation[-20:]:
            role = (m.get("role") or "").strip()
            content = (m.get("content") or "").strip()
            if role in {"user", "assistant"} and content:
                convo.append({"role": role, "content": content})

        # Avoid duplicating the last user message if the client already included it
        if not (convo and convo[-1]["role"] == "user" and convo[-1]["content"] == payload.message.strip()):
            convo.append({"role": "user", "content": payload.message.strip()})

        llm = await get_async_groq_client()
        llm_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"THREAD CONTEXT (read-only):\n\n{context_block}"},
            *convo,
        ]

        response = await llm.chat_completion(llm_messages, temperature=0.2, max_tokens=800)
        answer = ""
        try:
            answer = (response.choices[0].message.content or "").strip()
        except Exception:
            answer = ""

        # Heuristic suggestions for UI buttons
        msg_lower = payload.message.lower()
        suggested_actions: List[Dict[str, Any]] = []
        if any(k in msg_lower for k in ["draft", "reply", "respond"]):
            suggested_actions.append({"action": "draft_reply", "label": "Generate draft reply"})
        if any(k in msg_lower for k in ["action item", "todo", "task"]):
            suggested_actions.append({"action": "extract_tasks", "label": "Extract action items"})
        if any(k in msg_lower for k in ["summarize", "summary"]):
            suggested_actions.append({"action": "summarize", "label": "Summarize thread"})

        return AssistantChatResponse(answer=answer, suggested_actions=suggested_actions)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Assistant chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


async def _get_thread_data_for_reply(thread_id: str, user_id: str) -> Optional[Dict[str, Any]]:
    """Get thread data formatted for reply generation."""
    try:
        # Mock thread data - in production, this would come from database
        thread_data = {
            "thread_id": thread_id,
            "subject": "Project Update - Q1 Planning",
            "participant_emails": ["alice@example.com", "bob@example.com", user_id],
            "messages": [
                {
                    "message_id": "msg_1",
                    "from_email": "alice@example.com",
                    "to_emails": ["bob@example.com", user_id],
                    "body_text": "Hi team, I wanted to follow up on our Q1 planning. Can we schedule a meeting to discuss resource allocation?",
                    "gmail_date": "2024-01-15T10:30:00Z"
                },
                {
                    "message_id": "msg_2",
                    "from_email": "bob@example.com",
                    "to_emails": ["alice@example.com", user_id],
                    "body_text": "I'm available Tuesday afternoon. Also, we need to complete the quarterly report by end of month.",
                    "gmail_date": "2024-01-15T11:15:00Z"
                }
            ],
            "action_items": [
                {
                    "task_text": "Schedule Q1 planning meeting",
                    "priority": "high",
                    "assignee": user_id,
                    "due_date": "2024-01-20",
                    "status": "pending"
                },
                {
                    "task_text": "Complete quarterly report",
                    "priority": "high",
                    "assignee": "bob@example.com",
                    "due_date": "2024-01-31",
                    "status": "pending"
                }
            ],
            "entities": [
                {
                    "entity_type": "project",
                    "entity_value": "Q1-Planning",
                    "confidence_score": 0.9
                },
                {
                    "entity_type": "meeting",
                    "entity_value": "Q1 Planning Meeting",
                    "confidence_score": 0.8
                }
            ],
            "last_message_sender": "bob@example.com",
            "last_message_content": "I'm available Tuesday afternoon. Also, we need to complete the quarterly report by end of month.",
            "user_writing_style": UserWritingStyle(
                tone=ReplyTone.PROFESSIONAL,
                formality_level=0.7,
                average_sentence_length=15.0,
                greeting_style="Hi",
                closing_style="Best regards",
                signature_included=True,
                use_emojis=False,
                use_bullets=True,
                response_length_preference="medium"
            )
        }
        
        return thread_data
        
    except Exception as e:
        logger.error(f"Failed to get thread data for reply: {str(e)}")
        return None


@app.get("/tasks")
async def get_top_tasks(user_id: str, limit: int = 5, priority_filter: Optional[str] = None):
    """Get top action items for a user."""
    try:
        if not user_id:
            raise HTTPException(status_code=400, detail="user_id is required")
        
        # Mock tasks data for demo
        mock_tasks = [
            {
                "task_id": "task_1",
                "task_text": "Complete quarterly report by end of month",
                "priority": "high",
                "assignee": user_id,
                "due_date": "2024-01-31",
                "thread_id": "thread_123",
                "subject": "Project Update - Q1 Planning"
            },
            {
                "task_id": "task_2", 
                "task_text": "Schedule team meeting for Q1 planning",
                "priority": "urgent",
                "assignee": user_id,
                "due_date": "2024-01-20",
                "thread_id": "thread_456",
                "subject": "Meeting Request - Q1 Planning"
            },
            {
                "task_id": "task_3",
                "task_text": "Review project proposal",
                "priority": "medium",
                "assignee": "team@example.com",
                "due_date": "2024-01-25",
                "thread_id": "thread_789",
                "subject": "Project Proposal Review"
            },
            {
                "task_id": "task_4",
                "task_text": "Update documentation",
                "priority": "low",
                "assignee": user_id,
                "due_date": "2024-02-01",
                "thread_id": "thread_101",
                "subject": "Documentation Update"
            },
            {
                "task_id": "task_5",
                "task_text": "Client call preparation",
                "priority": "high",
                "assignee": user_id,
                "due_date": "2024-01-18",
                "thread_id": "thread_202",
                "subject": "Client Meeting"
            }
        ]
        
        # Apply priority filter if specified
        if priority_filter and priority_filter != "all":
            mock_tasks = [t for t in mock_tasks if t.get("priority") == priority_filter]
        
        # Sort by priority
        priority_order = {"urgent": 0, "high": 1, "medium": 2, "low": 3}
        mock_tasks.sort(key=lambda x: priority_order.get(x.get("priority", "medium"), 3))
        
        # Return limited tasks
        top_tasks = mock_tasks[:limit]
        
        return {
            "tasks": top_tasks,
            "total_found": len(mock_tasks),
            "limit": limit
        }
        
    except Exception as e:
        logger.error(f"Get tasks error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sync/status/{sync_id}")
async def get_sync_status(sync_id: str, user_id: str):
    """Get sync status."""
    try:
        if not user_id:
            raise HTTPException(status_code=400, detail="user_id is required")
        
        # Mock sync status - in production, this would come from database
        return {
            "sync_id": sync_id,
            "status": "completed",
            "progress": 100,
            "threads_processed": 10,
            "messages_processed": 25,
            "attachments_processed": 3,
            "errors": []
        }
        
    except Exception as e:
        logger.error(f"Get sync status error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/threads")
async def list_threads(user_id: str, limit: int = 20, offset: int = 0, sort_by: str = "last_message_date"):
    """List threads for a user with pagination and sorting."""
    try:
        if not user_id:
            raise HTTPException(status_code=400, detail="user_id is required")

        gmail_client = _get_gmail_client_for_user(user_id)
        if not await gmail_client.authenticate():
            raise HTTPException(status_code=401, detail="Gmail not authenticated. Call POST /auth/gmail first.")

        thread_ids, _next_page = await gmail_client.get_threads(max_results=limit)
        threads = []
        for thread_id in thread_ids:
            try:
                th = await gmail_client.get_thread(thread_id)
                messages_sorted = sorted(th.messages, key=lambda m: m.gmail_date)
                last_date = messages_sorted[-1].gmail_date if messages_sorted else datetime.utcnow()
                threads.append(
                    {
                        "thread_id": th.thread_id,
                        "subject": th.subject or "No Subject",
                        "message_count": len(messages_sorted),
                        "participant_emails": th.participant_emails,
                        "last_message_date": last_date,
                    }
                )
            except Exception as e:
                logger.debug(f"Failed to fetch thread {thread_id}: {str(e)}")
                continue
        
        return {
            "threads": threads,
            "total": len(threads),
            "limit": limit,
            "offset": offset
        }
        
    except Exception as e:
        logger.error(f"List threads error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Development server
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )
