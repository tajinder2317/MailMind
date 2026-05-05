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
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, status
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


class HealthResponse(BaseModel):
    status: str
    version: str
    components: Dict[str, str]


# Global application state
class MailMindState:
    def __init__(self):
        self.gmail_client: Optional[GmailClient] = None
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
        gmail_token_path = os.getenv("GMAIL_TOKEN_PATH", "token.json")
        
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable is required")
        
        # Initialize Gmail client
        state.gmail_client = GmailClient(gmail_credentials_path, gmail_token_path)
        
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
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check application health and component status."""
    components = {}
    
    # Check Gmail client
    try:
        if state.gmail_client:
            components["gmail"] = "connected" if await state.gmail_client.test_connection() else "disconnected"
        else:
            components["gmail"] = "not_initialized"
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
    """Authenticate with Gmail API."""
    try:
        user_id = request.get("user_id")
        if not user_id:
            raise HTTPException(status_code=400, detail="user_id is required")
            
        if not state.gmail_client:
            raise HTTPException(status_code=500, detail="Gmail client not initialized")
        
        # For demo purposes, return mock authentication
        # In production, this would handle OAuth2 flow properly
        logger.info(f"Mock authentication for user: {user_id}")
        
        # Mock successful authentication
        return {
            "status": "authenticated",
            "user_email": f"{user_id}@example.com",
            "messages_total": 1250,
            "threads_total": 342,
            "note": "This is mock authentication. In production, implement OAuth2 flow."
        }
        
        # Original OAuth2 code (commented out for demo)
        # success = await state.gmail_client.authenticate()
        # 
        # if success:
        #     profile = await state.gmail_client.get_user_profile()
        #     return {
        #         "status": "authenticated",
        #         "user_email": profile["email_address"],
        #         "messages_total": profile["messagesTotal"],
        #         "threads_total": profile["threadsTotal"]
        #     }
        # else:
        #     raise HTTPException(status_code=401, detail="Gmail authentication failed")
            
    except Exception as e:
        logger.error(f"Gmail authentication error: {str(e)}")
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
        
        # Get last sync state
        # last_sync = await get_last_sync_state(user_id)
        last_sync = None  # Simplified for demo
        
        # Fetch threads since last sync
        since_date = last_sync.last_synced_at if last_sync else datetime.utcnow() - timedelta(days=30)
        
        threads_processed = 0
        messages_processed = 0
        attachments_processed = 0
        errors = []
        
        async for gmail_thread in state.gmail_client.get_threads_since(since_date, max_threads):
            try:
                # Process thread
                thread_result = await process_thread(gmail_thread, user_id)
                
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


async def process_thread(gmail_thread, user_id: str) -> Dict[str, int]:
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
                content = await state.gmail_client.download_attachment(
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
    
    # Store in database (simplified for demo)
    # await store_thread_in_db(thread_data, context_result, action_items, references)
    
    return {
        "messages_processed": messages_processed,
        "attachments_processed": attachments_processed
    }


# Search endpoints
@app.post("/search", response_model=SearchResponse)
async def search_threads(request: dict):
    """Search threads using intelligent semantic similarity with self-correction."""
    try:
        user_id = request.get("user_id")
        query = request.get("query")
        limit = request.get("limit", 10)
        filters = request.get("filters")
        
        if not user_id or not query:
            raise HTTPException(status_code=400, detail="user_id and query are required")
        start_time = datetime.utcnow()
        
        # Initialize vector store if not available
        if not hasattr(state, 'vector_store') or not state.vector_store:
            qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
            qdrant_api_key = os.getenv("QDRANT_API_KEY")
            
            # Use OpenAI API key for embeddings (Groq doesn't provide embeddings)
            if not state.openai_api_key:
                raise HTTPException(status_code=500, detail="OpenAI API key not configured for embeddings")
            
            state.vector_store = VectorStore(qdrant_url, qdrant_api_key or "", state.openai_api_key)
            await state.vector_store.initialize_collection()
        
        # Detect query intent for self-correction
        query_intent = await _detect_query_intent(request.query)
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
        # First, search for threads with action items using metadata filtering
        results = await state.vector_store.client.search(
            collection_name="mailmind_threads",
            query_filter={
                "must": [
                    {"key": "user_id", "match": {"value": user_id}},
                    {"key": "intelligence_metadata.has_tasks", "match": {"value": True}}
                ]
            },
            limit=request.limit,
            with_payload=True
        )
        
        # Convert to response format
        formatted_results = []
        for hit in results:
            payload = hit.payload
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
                    "score": hit.score,
                    "snippet": _extract_task_snippet(relevant_tasks or action_items),
                    "participants": payload["participant_emails"],
                    "date": payload["last_message_date"],
                    "action_items": relevant_tasks or action_items[:3],  # Show relevant or first 3 tasks
                    "total_tasks": len(action_items),
                    "search_type": "task_search"
                })
        
        # If no results from task search, fallback to semantic search
        if not formatted_results:
            return await _search_semantic(request, user_id)
        
        return formatted_results
        
    except Exception as e:
        logger.error(f"Task search failed: {str(e)}")
        return await _search_semantic(request, user_id)


async def _search_entities(request: SearchRequest, user_id: str, intent: str) -> List[Dict[str, Any]]:
    """Search for threads with specific entities."""
    try:
        # Extract potential entity values from query
        entity_values = _extract_entity_values(request.query)
        
        if entity_values:
            # Search for threads containing these entities
            results = await state.vector_store.client.search(
                collection_name="mailmind_threads",
                query_filter={
                    "must": [
                        {"key": "user_id", "match": {"value": user_id}},
                        {"key": "entities.entity_value", "match": {"any": entity_values}}
                    ]
                },
                limit=request.limit,
                with_payload=True
            )
        else:
            # Search for threads with any entities
            results = await state.vector_store.client.search(
                collection_name="mailmind_threads",
                query_filter={
                    "must": [
                        {"key": "user_id", "match": {"value": user_id}},
                        {"key": "intelligence_metadata.has_entities", "match": {"value": True}}
                    ]
                },
                limit=request.limit,
                with_payload=True
            )
        
        # Convert to response format
        formatted_results = []
        for hit in results:
            payload = hit.payload
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
                "score": hit.score,
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
        return await _search_semantic(request, user_id)


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


@app.get("/threads/{thread_id}", response_model=ThreadResponse)
async def get_thread(thread_id: str, user_id: str):
    """Get detailed thread information."""
    try:
        if not user_id:
            raise HTTPException(status_code=400, detail="user_id is required")
        # Retrieve thread from database (to be implemented)
        # thread_data = await get_thread_from_db(thread_id, user_id)
        
        # Mock thread data for demo
        thread_data = {
            "thread_id": thread_id,
            "subject": "Project Update - Q1 Planning",
            "message_count": 5,
            "participant_emails": ["alice@example.com", "bob@example.com"],
            "first_message_date": "2024-01-15T10:30:00Z",
            "last_message_date": "2024-01-15T14:30:00Z",
            "has_attachments": True,
            "detected_tasks": ["Complete quarterly report", "Schedule team meeting"],
            "aggregated_content": "Discussion about Q1 planning and resource allocation...",
            "messages": [
                {
                    "message_id": "msg_1",
                    "from_email": "alice@example.com",
                    "to_emails": ["bob@example.com"],
                    "subject": "Project Update - Q1 Planning",
                    "body_text": "Let's discuss our Q1 planning...",
                    "gmail_date": "2024-01-15T10:30:00Z"
                }
            ],
            "action_items": [
                {
                    "task_text": "Complete quarterly report",
                    "priority": "high",
                    "assignee": "bob@example.com",
                    "due_date": "2024-01-20",
                    "status": "pending"
                }
            ],
            "entities": [
                {
                    "entity_type": "project",
                    "entity_value": "Q1-Planning",
                    "confidence_score": 0.9
                }
            ],
            "related_threads": ["thread_456", "thread_789"]
        }
        
        return ThreadResponse(**thread_data)
        
    except Exception as e:
        logger.error(f"Get thread error: {str(e)}")
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
        
        # Get threads from database (to be implemented)
        # threads = await list_user_threads(user_id, limit, offset, sort_by)
        
        # Mock threads for demo
        threads = [
            {
                "thread_id": "thread_123",
                "subject": "Project Update - Q1 Planning",
                "message_count": 5,
                "participant_emails": ["alice@example.com", "bob@example.com"],
                "last_message_date": datetime.utcnow() - timedelta(days=1),
                "has_attachments": True,
                "detected_tasks": 2
            }
        ]
        
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
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
