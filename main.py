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
        openai_api_key = os.getenv("OPENAI_API_KEY")
        gmail_credentials_path = os.getenv("GMAIL_CREDENTIALS_PATH", "credentials.json")
        gmail_token_path = os.getenv("GMAIL_TOKEN_PATH", "token.json")
        
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        # Initialize Gmail client
        state.gmail_client = GmailClient(gmail_credentials_path, gmail_token_path)
        
        # Initialize attachment processor
        state.attachment_processor = AttachmentProcessor()
        state.attachment_worker = AttachmentProcessorWorker(state.attachment_processor)
        
        # Initialize AI components
        state.sliding_context = SlidingContextProcessor(openai_api_key)
        state.action_extractor = ActionItemExtractor(openai_api_key)
        state.relationship_mapper = RelationshipMapper(openai_api_key)
        
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
async def authenticate_gmail(user_id: str = Depends(get_current_user)):
    """Authenticate with Gmail API."""
    try:
        if not state.gmail_client:
            raise HTTPException(status_code=500, detail="Gmail client not initialized")
        
        success = await state.gmail_client.authenticate()
        
        if success:
            profile = await state.gmail_client.get_user_profile()
            return {
                "status": "authenticated",
                "user_email": profile["email_address"],
                "messages_total": profile["messagesTotal"],
                "threads_total": profile["threadsTotal"]
            }
        else:
            raise HTTPException(status_code=401, detail="Gmail authentication failed")
            
    except Exception as e:
        logger.error(f"Gmail authentication error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Sync endpoints
@app.post("/sync/gmail", response_model=SyncResponse)
async def sync_gmail(
    request: SyncRequest,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(get_current_user)
):
    """Start Gmail synchronization process."""
    try:
        sync_id = f"sync_{user_id}_{datetime.utcnow().timestamp()}"
        
        # Add background task for sync
        background_tasks.add_task(
            perform_gmail_sync,
            sync_id=sync_id,
            user_id=request.user_id,
            max_threads=request.max_threads
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
async def search_threads(request: SearchRequest, user_id: str = Depends(get_current_user)):
    """Search threads using semantic similarity."""
    try:
        start_time = datetime.utcnow()
        
        # Perform semantic search (to be implemented with Qdrant)
        # results = await vector_store.search(request.query, user_id, request.limit, request.filters)
        
        # Mock results for demo
        results = [
            {
                "thread_id": "thread_123",
                "subject": "Project Update - Q1 Planning",
                "score": 0.95,
                "snippet": "Discussion about Q1 planning and resource allocation...",
                "participants": ["alice@example.com", "bob@example.com"],
                "date": "2024-01-15T10:30:00Z"
            }
        ]
        
        search_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
        
        return SearchResponse(
            results=results,
            total_found=len(results),
            search_time_ms=search_time
        )
        
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/threads/{thread_id}", response_model=ThreadResponse)
async def get_thread(thread_id: str, user_id: str = Depends(get_current_user)):
    """Get detailed thread information."""
    try:
        # Retrieve thread from database (to be implemented)
        # thread_data = await get_thread_from_db(thread_id, user_id)
        
        # Mock thread data for demo
        thread_data = {
            "thread_id": thread_id,
            "subject": "Project Update - Q1 Planning",
            "message_count": 5,
            "participant_emails": ["alice@example.com", "bob@example.com"],
            "first_message_date": datetime.utcnow() - timedelta(days=5),
            "last_message_date": datetime.utcnow() - timedelta(days=1),
            "aggregated_content": "Full thread content would be here...",
            "detected_tasks": ["Complete Q1 budget proposal", "Schedule team meeting"],
            "related_threads": ["thread_456", "thread_789"]
        }
        
        return ThreadResponse(**thread_data)
        
    except Exception as e:
        logger.error(f"Get thread error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Management endpoints
@app.get("/sync/status/{sync_id}")
async def get_sync_status(sync_id: str, user_id: str = Depends(get_current_user)):
    """Get status of a sync operation."""
    try:
        # Get sync status from database (to be implemented)
        # sync_status = await get_sync_status_from_db(sync_id, user_id)
        
        # Mock status for demo
        return {
            "sync_id": sync_id,
            "status": "completed",
            "progress": 100,
            "threads_processed": 25,
            "messages_processed": 87,
            "attachments_processed": 12,
            "started_at": datetime.utcnow() - timedelta(minutes=10),
            "completed_at": datetime.utcnow() - timedelta(minutes=2),
            "errors": []
        }
        
    except Exception as e:
        logger.error(f"Get sync status error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/threads")
async def list_threads(
    user_id: str = Depends(get_current_user),
    limit: int = 20,
    offset: int = 0,
    sort_by: str = "last_message_date"
):
    """List user's threads with pagination."""
    try:
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
