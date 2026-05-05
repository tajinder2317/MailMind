"""
MailMind Database Models

This module defines the SQLAlchemy models for the MailMind RAG system.
It includes models for threads, messages, attachments, sync tracking,
and cross-thread relationships.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from sqlalchemy import (
    Column, String, DateTime, Text, Integer, Boolean, 
    ForeignKey, JSON, Index, UniqueConstraint
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid

Base = declarative_base()


class SyncState(Base):
    """Tracks the last sync state for preventing duplicate processing."""
    __tablename__ = "sync_states"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String(255), nullable=False, index=True)
    last_synced_at = Column(DateTime(timezone=True), nullable=False)
    last_message_id = Column(String(255), nullable=False)
    last_history_id = Column(String(255), nullable=True, index=True)  # Gmail history ID for incremental sync
    sync_status = Column(String(50), default="completed")  # in_progress, completed, failed
    sync_type = Column(String(50), default="incremental")  # incremental, full, backfill
    threads_processed = Column(Integer, default=0)
    messages_processed = Column(Integer, default=0)
    errors = Column(JSON, default=list)  # List of error messages
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (
        Index("idx_sync_user_updated", "user_id", "updated_at"),
        Index("idx_sync_history_id", "last_history_id"),
    )


class Thread(Base):
    """Represents a Gmail thread with aggregated content and metadata."""
    __tablename__ = "threads"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    gmail_thread_id = Column(String(255), nullable=False, unique=True, index=True)
    user_id = Column(String(255), nullable=False, index=True)
    gmail_history_id = Column(String(255), nullable=True, index=True)  # Gmail history ID for thread
    subject = Column(Text, nullable=False)
    participant_emails = Column(JSON, nullable=False)  # List of email addresses
    message_count = Column(Integer, default=0)
    first_message_date = Column(DateTime(timezone=True), nullable=False)
    last_message_date = Column(DateTime(timezone=True), nullable=False)
    sync_priority = Column(String(20), default="normal")  # high, normal, low
    
    # Aggregated content for embedding
    aggregated_content = Column(Text, nullable=False)
    content_hash = Column(String(64), nullable=False, unique=True, index=True)
    token_count = Column(Integer, default=0)
    
    # Sliding context
    has_summary = Column(Boolean, default=False)
    running_summary = Column(Text)
    
    # Action items extracted from thread (enhanced structure)
    detected_tasks = Column(JSON, default=list)  # Legacy field for compatibility
    action_items = Column(JSON, default=list)  # Structured action items with full metadata
    
    # Cross-thread relationships and entities
    referenced_projects = Column(JSON, default=list)  # List of project codes
    referenced_urls = Column(JSON, default=list)  # List of URLs
    referenced_invoices = Column(JSON, default=list)  # List of invoice numbers
    extracted_entities = Column(JSON, default=list)  # Structured entities with metadata
    
    # Vector metadata
    vector_id = Column(String(255), nullable=True, unique=True)  # Qdrant vector ID
    embedded_at = Column(DateTime(timezone=True), nullable=True)
    
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    messages = relationship("Message", back_populates="thread", cascade="all, delete-orphan")
    attachments = relationship("Attachment", back_populates="thread", cascade="all, delete-orphan")
    thread_relationships = relationship("ThreadRelationship", foreign_keys="ThreadRelationship.source_thread_id", back_populates="source_thread")
    
    __table_args__ = (
        Index("idx_thread_user_date", "user_id", "last_message_date"),
        Index("idx_thread_content_hash", "content_hash"),
        Index("idx_thread_priority", "sync_priority"),
        Index("idx_thread_history_id", "gmail_history_id"),
    )


class Message(Base):
    """Represents an individual Gmail message within a thread."""
    __tablename__ = "messages"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    gmail_message_id = Column(String(255), nullable=False, unique=True, index=True)
    thread_id = Column(UUID(as_uuid=True), ForeignKey("threads.id"), nullable=False)
    user_id = Column(String(255), nullable=False, index=True)
    
    # Message metadata
    from_email = Column(String(255), nullable=False)
    to_emails = Column(JSON, nullable=False)  # List of recipient emails
    cc_emails = Column(JSON, default=list)  # List of CC emails
    bcc_emails = Column(JSON, default=list)  # List of BCC emails
    subject = Column(Text, nullable=False)
    body_text = Column(Text, nullable=False)
    body_html = Column(Text, nullable=True)
    
    # Timestamps
    gmail_date = Column(DateTime(timezone=True), nullable=False)
    internal_date = Column(Integer, nullable=False)  # Gmail internal timestamp
    
    # Processing flags
    processed = Column(Boolean, default=False)
    embedded = Column(Boolean, default=False)
    
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    thread = relationship("Thread", back_populates="messages")
    attachments = relationship("Attachment", back_populates="message", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index("idx_message_thread_date", "thread_id", "gmail_date"),
        Index("idx_message_user_processed", "user_id", "processed"),
    )


class Attachment(Base):
    """Represents file attachments with extracted text content."""
    __tablename__ = "attachments"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    gmail_attachment_id = Column(String(255), nullable=False, index=True)
    message_id = Column(UUID(as_uuid=True), ForeignKey("messages.id"), nullable=False)
    thread_id = Column(UUID(as_uuid=True), ForeignKey("threads.id"), nullable=False)
    user_id = Column(String(255), nullable=False, index=True)
    
    # File metadata
    filename = Column(String(512), nullable=False)
    mime_type = Column(String(255), nullable=False)
    size_bytes = Column(Integer, nullable=False)
    
    # Extracted content
    extracted_text = Column(Text, nullable=True)
    text_extraction_status = Column(String(50), default="pending")  # pending, processing, completed, failed
    extraction_error = Column(Text, nullable=True)
    
    # Processing flags
    processed = Column(Boolean, default=False)
    
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    message = relationship("Message", back_populates="attachments")
    thread = relationship("Thread", back_populates="attachments")
    
    __table_args__ = (
        Index("idx_attachment_message", "message_id"),
        Index("idx_attachment_thread", "thread_id"),
        Index("idx_attachment_user_status", "user_id", "text_extraction_status"),
    )


class ThreadRelationship(Base):
    """Tracks relationships between threads based on shared references."""
    __tablename__ = "thread_relationships"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_thread_id = Column(UUID(as_uuid=True), ForeignKey("threads.id"), nullable=False)
    target_thread_id = Column(UUID(as_uuid=True), ForeignKey("threads.id"), nullable=False)
    user_id = Column(String(255), nullable=False, index=True)
    
    # Relationship metadata
    relationship_type = Column(String(50), nullable=False)  # project, url, invoice, email_reference
    shared_reference = Column(String(512), nullable=False)  # The actual shared value
    confidence_score = Column(Integer, default=1)  # 1-10 confidence in relationship
    
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    
    # Relationships
    source_thread = relationship("Thread", foreign_keys=[source_thread_id], back_populates="thread_relationships")
    target_thread = relationship("Thread", foreign_keys=[target_thread_id])
    
    __table_args__ = (
        UniqueConstraint("source_thread_id", "target_thread_id", "shared_reference", name="unique_thread_relationship"),
        Index("idx_relationship_source", "source_thread_id"),
        Index("idx_relationship_target", "target_thread_id"),
        Index("idx_relationship_user_type", "user_id", "relationship_type"),
    )


class ProcessingLog(Base):
    """Logs processing events for debugging and monitoring."""
    __tablename__ = "processing_logs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String(255), nullable=False, index=True)
    
    # Event details
    event_type = Column(String(100), nullable=False)  # sync, embed, extract_text, detect_relationships
    entity_type = Column(String(50), nullable=False)  # thread, message, attachment
    entity_id = Column(String(255), nullable=False)
    
    # Status and details
    status = Column(String(50), nullable=False)  # started, completed, failed
    message = Column(Text, nullable=True)
    event_metadata = Column(JSON, nullable=True)  # Additional event-specific data
    
    # Timestamps
    duration_ms = Column(Integer, nullable=True)  # Processing duration in milliseconds
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    
    __table_args__ = (
        Index("idx_log_user_event", "user_id", "event_type", "created_at"),
        Index("idx_log_entity", "entity_type", "entity_id"),
    )


class UserSettings(Base):
    """Stores user-specific configuration and preferences."""
    __tablename__ = "user_settings"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String(255), nullable=False, unique=True, index=True)
    
    # Gmail API settings
    gmail_access_token = Column(Text, nullable=True)
    gmail_refresh_token = Column(Text, nullable=True)
    gmail_token_expires_at = Column(DateTime(timezone=True), nullable=True)
    
    # Processing preferences
    auto_sync_enabled = Column(Boolean, default=True)
    sync_interval_minutes = Column(Integer, default=60)
    include_attachments = Column(Boolean, default=True)
    
    # AI model settings
    embedding_model = Column(String(100), default="text-embedding-3-small")
    summary_model = Column(String(100), default="gpt-4o-mini")
    
    # Feature flags
    enable_action_extraction = Column(Boolean, default=True)
    enable_thread_relationships = Column(Boolean, default=True)
    enable_sliding_context = Column(Boolean, default=True)
    
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (
        Index("idx_settings_user", "user_id"),
    )
