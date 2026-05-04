"""
Intelligence Pipeline for MailMind

This module provides the core intelligence processing pipeline including:
- Action-Item Extraction using GPT-4o-mini
- Entity-Based Mapping for relationships
- Thread processing and embedding
- Metadata enrichment for search
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import json
import re
from dataclasses import dataclass, asdict
from enum import Enum

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    """Priority levels for extracted tasks."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class TaskStatus(Enum):
    """Status of extracted tasks."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    OVERDUE = "overdue"


class EntityType(Enum):
    """Types of entities that can be extracted."""
    PROJECT = "project"
    INVOICE = "invoice"
    JIRA = "jira"
    MEETING = "meeting"
    DOCUMENT = "document"
    PERSON = "person"
    DATE = "date"
    URL = "url"


@dataclass
class ExtractedTask:
    """Represents an extracted action item."""
    task_text: str
    priority: TaskPriority
    assignee: Optional[str] = None
    due_date: Optional[datetime] = None
    status: TaskStatus = TaskStatus.PENDING
    confidence_score: float = 0.8
    source_message_id: Optional[str] = None
    keywords: List[str] = None
    context: str = ""


@dataclass
class ExtractedEntity:
    """Represents an extracted entity."""
    entity_type: EntityType
    entity_value: str
    context: str
    confidence_score: float = 0.8
    source_message_id: Optional[str] = None
    metadata: Dict[str, Any] = None


@dataclass
class ProcessedThread:
    """Represents a fully processed thread with intelligence."""
    thread_id: str
    user_id: str
    content: str
    embedding: List[float]
    action_items: List[ExtractedTask]
    entities: List[ExtractedEntity]
    metadata: Dict[str, Any]
    processed_at: datetime


class ActionItemExtractor:
    """
    Extracts action items from thread content using GPT-4o-mini.
    """
    
    def __init__(self, openai_client: AsyncOpenAI):
        self.client = openai_client
        self.model = "gpt-4o-mini"
    
    async def extract_action_items(
        self,
        thread_content: str,
        messages: List[Dict[str, Any]],
        participants: List[str]
    ) -> List[ExtractedTask]:
        """
        Extract action items from thread content using LLM.
        
        Args:
            thread_content: Full thread content
            messages: List of message dictionaries
            participants: List of participant emails
            
        Returns:
            List of extracted action items
        """
        try:
            # Prepare the prompt for action item extraction
            participants_str = ", ".join(participants)
            
            prompt = f"""
Analyze the following email thread and extract all action items, tasks, and commitments.

Participants: {participants_str}

For each action item, provide:
1. The specific task or commitment
2. Priority level (low/medium/high/urgent)
3. Who is assigned (email address if mentioned)
4. Due date (if mentioned)
5. Current status (pending/in_progress/completed)
6. Confidence score (0.0-1.0)
7. Relevant keywords
8. Context from the email

Email Thread:
{thread_content[:8000]}

Respond in JSON format:
{{
    "action_items": [
        {{
            "task_text": "Complete the quarterly report",
            "priority": "high",
            "assignee": "john@example.com",
            "due_date": "2024-01-15",
            "status": "pending",
            "confidence_score": 0.9,
            "keywords": ["report", "quarterly", "complete"],
            "context": "John needs to complete the quarterly report by Friday",
            "source_message_id": "msg_123"
        }}
    ]
}}"""
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at identifying action items and tasks in email conversations. Always respond in valid JSON format."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
                temperature=0.1
            )
            
            content = response.choices[0].message.content.strip()
            
            # Parse JSON response
            try:
                data = json.loads(content)
                action_items = []
                
                for item_data in data.get('action_items', []):
                    # Parse due date
                    due_date = None
                    if item_data.get('due_date'):
                        try:
                            due_date = datetime.fromisoformat(item_data['due_date'].replace('Z', '+00:00'))
                        except:
                            due_date = self._parse_relative_date(item_data['due_date'])
                    
                    # Create task object
                    task = ExtractedTask(
                        task_text=item_data['task_text'],
                        priority=TaskPriority(item_data.get('priority', 'medium')),
                        assignee=item_data.get('assignee'),
                        due_date=due_date,
                        status=TaskStatus(item_data.get('status', 'pending')),
                        confidence_score=float(item_data.get('confidence_score', 0.8)),
                        source_message_id=item_data.get('source_message_id'),
                        keywords=item_data.get('keywords', []),
                        context=item_data.get('context', '')
                    )
                    
                    action_items.append(task)
                
                logger.info(f"Extracted {len(action_items)} action items")
                return action_items
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse action items JSON: {str(e)}")
                return []
                
        except Exception as e:
            logger.error(f"Action item extraction failed: {str(e)}")
            return []
    
    def _parse_relative_date(self, date_str: str) -> Optional[datetime]:
        """Parse relative date expressions."""
        date_str = date_str.lower()
        
        # Simple relative date parsing
        if 'today' in date_str:
            return datetime.utcnow()
        elif 'tomorrow' in date_str:
            return datetime.utcnow() + timedelta(days=1)
        elif 'next week' in date_str:
            return datetime.utcnow() + timedelta(days=7)
        elif 'this week' in date_str:
            return datetime.utcnow() + timedelta(days=3)
        
        return None


class EntityExtractor:
    """
    Extracts entities from thread content for relationship mapping.
    """
    
    def __init__(self):
        # Entity patterns
        self.patterns = {
            EntityType.PROJECT: [
                r'\b([A-Z]{2,}-\d{3,})\b',  # PROJ-123, ABC-456
                r'\b([A-Z]{3,}\d{3,})\b',   # ABC123, XYZ789
                r'\b(project\s+\w+\s*#\s*\d+)\b',
                r'\b(task\s+\w+\s*#\s*\d+)\b',
                r'\b(epic\s+\w+\s*#\s*\d+)\b',
            ],
            EntityType.INVOICE: [
                r'\b(invoice\s+#?\s*\d{4,})\b',
                r'\b(inv\s+#?\s*\d{4,})\b',
                r'\b(receipt\s+#?\s*\d{4,})\b',
                r'\b(order\s+#?\s*\d{4,})\b',
                r'\b(po\s+#?\s*\d{4,})\b',
            ],
            EntityType.JIRA: [
                r'\b([A-Z]+-\d{3,})\b',  # JIRA-123, PROJ-456
                r'\b(jira\s+#?\s*\d+)\b',
                r'\b(ticket\s+#?\s*\d+)\b',
            ],
            EntityType.MEETING: [
                r'\b(meeting\s+on\s+\d{1,2}/\d{1,2}/\d{4})\b',
                r'\b(call\s+on\s+\d{1,2}/\d{1,2}/\d{4})\b',
                r'\b(discussion\s+on\s+\d{1,2}/\d{1,2}/\d{4})\b',
            ],
            EntityType.DOCUMENT: [
                r'\b(document\s+#?\s*\w+)\b',
                r'\b(doc\s+#?\s*\w+)\b',
                r'\b(file\s+#?\s*\w+)\b',
                r'\b(attachment\s+#?\s*\w+)\b',
            ],
            EntityType.URL: [
                r'https?://[^\s<>"{}|\\^`[\]]+',
                r'www\.[^\s<>"{}|\\^`[\]]+\.[a-zA-Z]{2,}',
            ],
        }
        
        # Compile patterns
        self.compiled_patterns = {}
        for entity_type, patterns in self.patterns.items():
            self.compiled_patterns[entity_type] = [
                re.compile(pattern, re.IGNORECASE) for pattern in patterns
            ]
    
    async def extract_entities(
        self,
        thread_content: str,
        messages: List[Dict[str, Any]]
    ) -> List[ExtractedEntity]:
        """
        Extract entities from thread content.
        
        Args:
            thread_content: Full thread content
            messages: List of message dictionaries
            
        Returns:
            List of extracted entities
        """
        entities = []
        
        for entity_type, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                matches = pattern.finditer(thread_content)
                
                for match in matches:
                    try:
                        entity_value = match.group(1) if match.groups() else match.group(0)
                        
                        # Normalize entity value
                        normalized_value = self._normalize_entity(entity_value, entity_type)
                        
                        if not normalized_value:
                            continue
                        
                        # Find source message
                        source_message_id = self._find_source_message(match.start(), thread_content, messages)
                        
                        # Extract context
                        context = self._extract_context(match.start(), thread_content)
                        
                        entity = ExtractedEntity(
                            entity_type=entity_type,
                            entity_value=normalized_value,
                            context=context,
                            confidence_score=0.8,
                            source_message_id=source_message_id,
                            metadata={"match_pattern": pattern.pattern}
                        )
                        
                        entities.append(entity)
                        
                    except Exception as e:
                        logger.warning(f"Error processing entity match: {str(e)}")
                        continue
        
        # Remove duplicates
        unique_entities = self._deduplicate_entities(entities)
        
        logger.info(f"Extracted {len(unique_entities)} entities")
        return unique_entities
    
    def _normalize_entity(self, value: str, entity_type: EntityType) -> str:
        """Normalize entity value based on type."""
        value = value.strip().upper()
        
        if entity_type == EntityType.PROJECT:
            # Normalize project codes
            if not re.match(r'^[A-Z]+-\d+$', value):
                match = re.match(r'^([A-Z]+)(\d+)$', value)
                if match:
                    value = f"{match.group(1)}-{match.group(2)}"
        
        elif entity_type == EntityType.INVOICE:
            # Normalize invoice numbers
            if not value.startswith(('INVOICE-', 'INV-', 'PO-')):
                value = f"INVOICE-{value}"
        
        elif entity_type == EntityType.URL:
            # Normalize URLs
            if not value.startswith(('http://', 'https://')):
                value = f"https://{value}"
        
        return value
    
    def _find_source_message(self, position: int, thread_content: str, messages: List[Dict[str, Any]]) -> Optional[str]:
        """Find the message that contains the given position."""
        # Simplified implementation - in production, track message positions
        if messages:
            return messages[-1].get('message_id')
        return None
    
    def _extract_context(self, position: int, thread_content: str) -> str:
        """Extract context around a position."""
        start = max(0, position - 100)
        end = min(len(thread_content), position + 100)
        return thread_content[start:end].strip()
    
    def _deduplicate_entities(self, entities: List[ExtractedEntity]) -> List[ExtractedEntity]:
        """Remove duplicate entities."""
        seen = set()
        unique_entities = []
        
        for entity in entities:
            key = (entity.entity_type, entity.entity_value)
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)
        
        return unique_entities


class EmbeddingPipeline:
    """
    Main intelligence pipeline for processing threads.
    """
    
    def __init__(self, openai_api_key: str):
        self.openai_client = AsyncOpenAI(api_key=openai_api_key)
        self.action_extractor = ActionItemExtractor(self.openai_client)
        self.entity_extractor = EntityExtractor()
        self.embedding_model = "text-embedding-3-small"
    
    async def process_thread(
        self,
        thread_id: str,
        user_id: str,
        thread_content: str,
        messages: List[Dict[str, Any]],
        participants: List[str]
    ) -> ProcessedThread:
        """
        Process a thread through the intelligence pipeline.
        
        Args:
            thread_id: Gmail thread ID
            user_id: User ID
            thread_content: Processed thread content
            messages: List of message dictionaries
            participants: List of participant emails
            
        Returns:
            Processed thread with intelligence
        """
        try:
            logger.info(f"Processing thread {thread_id} through intelligence pipeline")
            
            # Step 1: Extract action items
            action_items = await self.action_extractor.extract_action_items(
                thread_content, messages, participants
            )
            
            # Step 2: Extract entities
            entities = await self.entity_extractor.extract_entities(
                thread_content, messages
            )
            
            # Step 3: Generate embedding
            embedding = await self._generate_embedding(thread_content)
            
            # Step 4: Create metadata
            metadata = self._create_metadata(action_items, entities, participants)
            
            # Step 5: Create processed thread
            processed_thread = ProcessedThread(
                thread_id=thread_id,
                user_id=user_id,
                content=thread_content,
                embedding=embedding,
                action_items=action_items,
                entities=entities,
                metadata=metadata,
                processed_at=datetime.utcnow()
            )
            
            logger.info(f"Successfully processed thread {thread_id}")
            return processed_thread
            
        except Exception as e:
            logger.error(f"Failed to process thread {thread_id}: {str(e)}")
            raise
    
    async def _generate_embedding(self, content: str) -> List[float]:
        """Generate embedding for thread content."""
        try:
            # Truncate content if too long
            max_chars = 8000
            if len(content) > max_chars:
                content = content[:max_chars] + "..."
            
            response = await self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=content
            )
            
            return response.data[0].embedding
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {str(e)}")
            raise
    
    def _create_metadata(
        self,
        action_items: List[ExtractedTask],
        entities: List[ExtractedEntity],
        participants: List[str]
    ) -> Dict[str, Any]:
        """Create metadata for the processed thread."""
        # Action items metadata
        action_metadata = {
            "total_tasks": len(action_items),
            "pending_tasks": len([t for t in action_items if t.status == TaskStatus.PENDING]),
            "high_priority_tasks": len([t for t in action_items if t.priority == TaskPriority.HIGH]),
            "urgent_tasks": len([t for t in action_items if t.priority == TaskPriority.URGENT]),
            "overdue_tasks": len([t for t in action_items if t.due_date and t.due_date < datetime.utcnow()]),
        }
        
        # Entities metadata
        entity_counts = {}
        for entity in entities:
            entity_counts[entity.entity_type.value] = entity_counts.get(entity.entity_type.value, 0) + 1
        
        # Participant metadata
        participant_metadata = {
            "total_participants": len(participants),
            "participant_emails": participants,
        }
        
        return {
            "action_items": action_metadata,
            "entities": entity_counts,
            "participants": participant_metadata,
            "has_tasks": len(action_items) > 0,
            "has_entities": len(entities) > 0,
            "intelligence_version": "1.0.0"
        }
    
    def prepare_qdrant_payload(
        self,
        processed_thread: ProcessedThread,
        thread_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Prepare payload for Qdrant vector storage.
        
        Args:
            processed_thread: Processed thread object
            thread_metadata: Additional thread metadata
            
        Returns:
            Qdrant payload dictionary
        """
        # Convert action items to serializable format
        action_items_serializable = []
        for task in processed_thread.action_items:
            task_dict = asdict(task)
            if task_dict.get('due_date'):
                task_dict['due_date'] = task_dict['due_date'].isoformat()
            action_items_serializable.append(task_dict)
        
        # Convert entities to serializable format
        entities_serializable = []
        for entity in processed_thread.entities:
            entity_dict = asdict(entity)
            entity_dict['entity_type'] = entity.entity_type.value
            entities_serializable.append(entity_dict)
        
        # Build payload
        payload = {
            "thread_id": processed_thread.thread_id,
            "user_id": processed_thread.user_id,
            "subject": thread_metadata.get("subject", ""),
            "participant_emails": thread_metadata.get("participant_emails", []),
            "message_count": thread_metadata.get("message_count", 0),
            "first_message_date": thread_metadata.get("first_message_date"),
            "last_message_date": thread_metadata.get("last_message_date"),
            "has_attachments": thread_metadata.get("has_attachments", False),
            "sync_priority": thread_metadata.get("sync_priority", "normal"),
            
            # Intelligence data
            "action_items": action_items_serializable,
            "entities": entities_serializable,
            "intelligence_metadata": processed_thread.metadata,
            
            # Timestamps
            "processed_at": processed_thread.processed_at.isoformat(),
            "embedded_at": datetime.utcnow().isoformat(),
        }
        
        return payload
