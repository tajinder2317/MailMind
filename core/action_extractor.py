"""
Action Item Extractor for MailMind

This module extracts action items, tasks, and commitments from email threads
using natural language processing and LLM analysis.
"""

import asyncio
import logging
import os
from typing import List, Dict, Any, Optional, Set
from datetime import datetime, timedelta
import re
import json

from core.groq_client import get_async_groq_client
from dataclasses import dataclass
from enum import Enum

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


@dataclass
class ActionItem:
    """Represents an extracted action item."""
    task_text: str
    priority: TaskPriority
    assignee: Optional[str]  # Email address of person assigned
    due_date: Optional[datetime]
    context: str  # Surrounding context for the task
    confidence_score: float  # 0.0 to 1.0
    source_message_id: str
    keywords: List[str]


class ActionItemExtractor:
    """
    Extracts action items from email thread content using NLP and LLM analysis.
    
    Features:
    - Pattern-based task detection
    - LLM-powered semantic analysis
    - Priority and deadline extraction
    - Assignee identification
    - Confidence scoring
    """
    
    # Task detection patterns
    TASK_PATTERNS = [
        r'\b(please|can you|could you|will you|would you)\s+(.+?)\s+(by|before|on)\s+([A-Za-z]+\s+\d{1,2}|\d{1,2}/\d{1,2}|\d{1,2}-\d{1,2})',
        r'\b(I will|I\'ll|we will|we\'ll)\s+(.+?)(?:\s+by\s+([A-Za-z]+\s+\d{1,2}|\d{1,2}/\d{1,2}|\d{1,2}-\d{1,2}))?',
        r'\b(action item|todo|to-do|task|follow up|follow-up)\s*[:\-]?\s*(.+)',
        r'\b(need to|must|should|have to)\s+(.+?)(?:\s+by\s+([A-Za-z]+\s+\d{1,2}|\d{1,2}/\d{1,2}|\d{1,2}-\d{1,2}))?',
        r'\b(schedule|arrange|organize|set up)\s+(.+?)(?:\s+(for|on|by)\s+([A-Za-z]+\s+\d{1,2}|\d{1,2}/\d{1,2}|\d{1,2}-\d{1,2}))?',
        r'\b(send|email|call|contact|reach out to)\s+(.+?)(?:\s+by\s+([A-Za-z]+\s+\d{1,2}|\d{1,2}/\d{1,2}|\d{1,2}-\d{1,2}))?',
    ]
    
    # Priority indicators
    HIGH_PRIORITY_KEYWORDS = [
        'urgent', 'asap', 'immediately', 'critical', 'important', 'priority',
        'as soon as possible', 'right away', 'emergency'
    ]
    
    MEDIUM_PRIORITY_KEYWORDS = [
        'this week', 'soon', 'promptly', 'quickly', 'next week'
    ]
    
    # Time expressions for due dates
    TIME_EXPRESSIONS = {
        'today': 0,
        'tomorrow': 1,
        'next week': 7,
        'this week': 3,
        'end of week': 5,
        'next month': 30,
        'end of month': 25,
        'asap': 0,
        'urgent': 0,
        'immediately': 0
    }
    
    # OpenAI settings

    def __init__(self):
        self.groq_client = None
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.TASK_PATTERNS]

    def _build_message_context(self, messages: List[Dict[str, Any]], max_messages: int = 10) -> str:
        """Build a compact message context block for LLM prompts."""
        if not messages:
            return ""

        # Keep only the most recent messages for brevity.
        msgs = messages[-max_messages:]
        parts: List[str] = []
        for msg in msgs:
            from_email = msg.get("from_email", "")
            date = msg.get("gmail_date")
            date_str = date.isoformat() if hasattr(date, "isoformat") else str(date or "")
            body = (msg.get("body_text") or "").strip()
            body = body[:1000] + ("..." if len(body) > 1000 else "")
            parts.append(f"- {date_str} {from_email}: {body}")

        return "\n".join(parts)

    def _validate_action_item(self, item: Dict[str, Any], participants: List[str]) -> Optional[ActionItem]:
        """Validate and normalize an LLM-extracted action item."""
        try:
            task_text = (item.get("task_text") or "").strip()
            if not task_text or len(task_text) < 5:
                return None

            priority_raw = (item.get("priority") or "medium").strip().lower()
            if priority_raw not in {"low", "medium", "high", "urgent"}:
                priority_raw = "medium"
            priority = TaskPriority(priority_raw)

            assignee = item.get("assignee")
            if isinstance(assignee, str):
                assignee = assignee.strip()
                if assignee.lower() in {"unassigned", "none", ""}:
                    assignee = None
            else:
                assignee = None

            due_date = None
            due_raw = item.get("due_date")
            if isinstance(due_raw, str) and due_raw.strip():
                due_date = self._extract_due_date(due_raw.strip()) or self._parse_date_string(due_raw.strip())

            context = (item.get("context") or "").strip()
            confidence = float(item.get("confidence_score", 0.7) or 0.7)
            confidence = max(0.1, min(1.0, confidence))

            source_message_id = str(item.get("source_message_id") or "")
            keywords = item.get("keywords") or self._extract_keywords(task_text)
            if not isinstance(keywords, list):
                keywords = self._extract_keywords(task_text)

            return ActionItem(
                task_text=task_text,
                priority=priority,
                assignee=assignee,
                due_date=due_date,
                context=context,
                confidence_score=confidence,
                source_message_id=source_message_id,
                keywords=keywords,
            )
        except Exception:
            return None

    def _parse_date_string(self, date_str: str) -> Optional[datetime]:
        """Best-effort parse of a date string into a datetime."""
        try:
            # ISO or ISO-like
            return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        except Exception:
            return None
    
    async def _get_client(self):
        """Get Groq client instance."""
        if self.groq_client is None:
            self.groq_client = await get_async_groq_client()
        return self.groq_client
    
    async def _compile_patterns(self):
        """Backward-compatible no-op (patterns compiled in __init__)."""
        if not hasattr(self, "compiled_patterns") or not self.compiled_patterns:
            self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.TASK_PATTERNS]
    
    async def extract_action_items(
        self,
        thread_content: str,
        messages: List[Dict[str, Any]],
        participants: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Extract action items from thread content.
        
        Args:
            thread_content: Full thread content (may include summary)
            messages: List of message dictionaries with metadata
            participants: List of participant email addresses
            
        Returns:
            List of action item dictionaries
        """
        try:
            logger.info(f"Extracting action items from thread with {len(messages)} messages")
            
            # Step 1: Pattern-based extraction
            pattern_items = await self._extract_with_patterns(thread_content, messages)
            
            # Step 2: LLM-based semantic extraction (optional; can be disabled to avoid Groq rate limits)
            enable_llm = os.getenv("ENABLE_LLM_ACTION_EXTRACTION", "true").strip().lower() in {"1", "true", "yes", "y"}
            llm_items = []
            if enable_llm:
                llm_items = await self._extract_with_llm(thread_content, messages, participants)
            
            # Step 3: Merge and deduplicate results
            merged_items = await self._merge_action_items(pattern_items, llm_items)
            
            # Step 4: Score and rank items
            scored_items = await self._score_action_items(merged_items, messages)
            
            # Convert to serializable format
            result_items = []
            for item in scored_items:
                result_items.append({
                    'task_text': item.task_text,
                    'priority': item.priority.value,
                    'assignee': item.assignee,
                    'due_date': item.due_date.isoformat() if item.due_date else None,
                    'context': item.context,
                    'confidence_score': item.confidence_score,
                    'source_message_id': item.source_message_id,
                    'keywords': item.keywords
                })
            
            logger.info(f"Extracted {len(result_items)} action items")
            return result_items
            
        except Exception as e:
            logger.error(f"Error extracting action items: {str(e)}")
            return []
    
    async def _extract_with_patterns(
        self,
        thread_content: str,
        messages: List[Dict[str, Any]]
    ) -> List[ActionItem]:
        """
        Extract action items using regex patterns.
        
        Args:
            thread_content: Full thread content
            messages: Message list for context
            
        Returns:
            List of ActionItem objects
        """
        items = []
        
        for pattern in self.compiled_patterns:
            matches = pattern.finditer(thread_content)
            
            for match in matches:
                try:
                    # Extract task text and metadata
                    groups = match.groups()
                    task_text = groups[0] if groups else match.group(0)
                    
                    # Clean up task text
                    task_text = self._clean_task_text(task_text)
                    
                    if not task_text or len(task_text) < 5:
                        continue
                    
                    # Extract priority
                    priority = self._extract_priority(task_text)
                    
                    # Extract due date
                    due_date = self._extract_due_date(task_text)
                    
                    # Find source message
                    source_message_id = self._find_source_message(match.start(), thread_content, messages)
                    
                    # Extract context
                    context = self._extract_context(match.start(), thread_content)
                    
                    # Calculate confidence
                    confidence = self._calculate_pattern_confidence(task_text, match)
                    
                    # Extract keywords
                    keywords = self._extract_keywords(task_text)
                    
                    item = ActionItem(
                        task_text=task_text,
                        priority=priority,
                        assignee=None,  # Will be determined later
                        due_date=due_date,
                        context=context,
                        confidence_score=confidence,
                        source_message_id=source_message_id,
                        keywords=keywords
                    )
                    
                    items.append(item)
                    
                except Exception as e:
                    logger.warning(f"Error processing pattern match: {str(e)}")
                    continue
        
        return items
    
    async def _extract_with_llm(
        self,
        thread_content: str,
        messages: List[Dict[str, Any]],
        participants: List[str]
    ) -> List[ActionItem]:
        """
        Extract action items using LLM semantic analysis.
        
        Args:
            thread_content: Full thread content
            messages: Message list for context
            participants: Participant email addresses
            
        Returns:
            List of ActionItem objects
        """
        try:
            # Get Groq client
            client = await self._get_client()
            
            # Build context from messages
            message_context = self._build_message_context(messages, max_messages=5)
            safe_thread_content = (thread_content or "").strip()
            if len(safe_thread_content) > 3500:
                safe_thread_content = safe_thread_content[:3500] + "..."
            
            prompt = f"""
You are an expert action item extractor. Analyze the following email thread content and extract all action items, tasks, commitments, and deadlines.

Thread Participants: {', '.join(participants)}

Message Context:
{message_context}

Content to Analyze:
{safe_thread_content}

Extract ALL action items and return them as a JSON array with the following structure:
[
    {{
        "task_text": "Complete the quarterly report",
        "priority": "high",
        "assignee": "john@example.com",
        "due_date": "2024-01-31",
        "status": "pending",
        "confidence_score": 0.9,
        "source_message_id": "msg_123",
        "context": "Mentioned in the meeting notes"
    }}
]

Requirements:
1. Extract ALL action items, tasks, and commitments
2. Include both explicit and implied tasks
3. Identify assignees from participants or use "Unassigned"
4. Extract due dates or infer from context
5. Assign priority based on urgency indicators
6. Include confidence scores (0.1-1.0)
7. Provide brief context for each task
8. Return valid JSON array only

Action Items:
"""
            
            response = await client.chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert action item extractor. Extract all tasks, commitments, and action items from email content. Return valid JSON only."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model=os.getenv("ACTION_MODEL", os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")),
                max_tokens=1500,
                temperature=0.1
            )
            
            content = response.choices[0].message.content.strip()
            
            # Extract the first JSON array/object from the response (models often add prose).
            content = self._extract_first_json_block(content)
            
            # Parse JSON
            try:
                action_items = json.loads(content)
                if not isinstance(action_items, list):
                    action_items = []
                
                # Validate and enhance action items
                validated_items = []
                for item in action_items:
                    validated_item = self._validate_action_item(item, participants)
                    if validated_item:
                        validated_items.append(validated_item)
                
                logger.info(f"Extracted {len(validated_items)} action items via LLM")
                return validated_items
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM response as JSON: {str(e)}")
                logger.error(f"Raw content: {content}")
                return []
            
        except Exception as e:
            logger.error(f"LLM extraction failed: {str(e)}")
            return []

    def _extract_first_json_block(self, text: str) -> str:
        """Extract the first top-level JSON array/object from text."""
        if not text:
            return ""

        stripped = text.strip()

        # Fast path: code-fenced JSON
        if "```" in stripped:
            parts = stripped.split("```")
            for part in parts:
                p = part.strip()
                if p.startswith("json"):
                    p = p[4:].strip()
                if p.startswith("[") or p.startswith("{"):
                    stripped = p
                    break

        # Find first '{' or '['
        start_candidates = [i for i in [stripped.find("["), stripped.find("{")] if i != -1]
        if not start_candidates:
            return stripped
        start = min(start_candidates)
        s = stripped[start:]

        # Scan to matching closing bracket/brace
        opening = s[0]
        closing = "]" if opening == "[" else "}"
        depth = 0
        in_string = False
        escape = False
        for idx, ch in enumerate(s):
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == "\"":
                    in_string = False
                continue

            if ch == "\"":
                in_string = True
                continue

            if ch == opening:
                depth += 1
            elif ch == closing:
                depth -= 1
                if depth == 0:
                    return s[: idx + 1].strip()

        return s.strip()
    
    async def _merge_action_items(
        self,
        pattern_items: List[ActionItem],
        llm_items: List[ActionItem]
    ) -> List[ActionItem]:
        """
        Merge and deduplicate action items from different extraction methods.
        
        Args:
            pattern_items: Items from pattern extraction
            llm_items: Items from LLM extraction
            
        Returns:
            Merged list of unique action items
        """
        all_items = pattern_items + llm_items
        merged_items = []
        
        for item in all_items:
            # Check for duplicates
            is_duplicate = False
            for existing in merged_items:
                similarity = self._calculate_similarity(item.task_text, existing.task_text)
                if similarity > 0.8:  # 80% similarity threshold
                    # Keep the one with higher confidence
                    if item.confidence_score > existing.confidence_score:
                        merged_items.remove(existing)
                        merged_items.append(item)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                merged_items.append(item)
        
        return merged_items
    
    async def _score_action_items(
        self,
        items: List[ActionItem],
        messages: List[Dict[str, Any]]
    ) -> List[ActionItem]:
        """
        Score and rank action items based on various factors.
        
        Args:
            items: List of action items
            messages: Message list for context
            
        Returns:
            Scored and sorted list of action items
        """
        for item in items:
            # Boost confidence based on multiple factors
            confidence_boost = 0.0
            
            # Priority boost
            if item.priority == TaskPriority.URGENT:
                confidence_boost += 0.2
            elif item.priority == TaskPriority.HIGH:
                confidence_boost += 0.1
            
            # Due date boost
            if item.due_date and item.due_date <= datetime.utcnow() + timedelta(days=7):
                confidence_boost += 0.1
            
            # Assignee specificity boost
            if item.assignee:
                confidence_boost += 0.1
            
            # Message recency boost
            if item.source_message_id:
                source_msg = next((m for m in messages if m.get('message_id') == item.source_message_id), None)
                if source_msg and source_msg.get('gmail_date'):
                    days_old = (datetime.utcnow() - source_msg['gmail_date']).days
                    if days_old <= 7:
                        confidence_boost += 0.1
            
            # Apply boost (cap at 1.0)
            item.confidence_score = min(1.0, item.confidence_score + confidence_boost)
        
        # Sort by confidence and priority
        priority_order = {TaskPriority.URGENT: 4, TaskPriority.HIGH: 3, TaskPriority.MEDIUM: 2, TaskPriority.LOW: 1}
        
        return sorted(
            items,
            key=lambda x: (x.confidence_score, priority_order[x.priority]),
            reverse=True
        )
    
    def _clean_task_text(self, text: str) -> str:
        """Clean and normalize task text."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove leading/trailing punctuation
        text = text.strip('.,;:!?')
        
        # Capitalize first letter
        if text:
            text = text[0].upper() + text[1:]
        
        return text
    
    def _extract_priority(self, text: str) -> TaskPriority:
        """Extract priority from task text."""
        text_lower = text.lower()
        
        for keyword in self.HIGH_PRIORITY_KEYWORDS:
            if keyword in text_lower:
                return TaskPriority.HIGH
        
        for keyword in self.MEDIUM_PRIORITY_KEYWORDS:
            if keyword in text_lower:
                return TaskPriority.MEDIUM
        
        return TaskPriority.LOW
    
    def _extract_due_date(self, text: str) -> Optional[datetime]:
        """Extract due date from task text."""
        text_lower = text.lower()
        
        # Check for relative time expressions
        for expression, days in self.TIME_EXPRESSIONS.items():
            if expression in text_lower:
                return datetime.utcnow() + timedelta(days=days)
        
        # Check for date patterns (simplified)
        date_patterns = [
            r'(\d{1,2})/(\d{1,2})/(\d{4})',
            r'(\d{1,2})-(\d{1,2})-(\d{4})',
            r'([A-Za-z]+)\s+(\d{1,2}),?\s+(\d{4})'
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    # This is simplified - in production, use a proper date parser
                    return datetime.utcnow() + timedelta(days=7)  # Placeholder
                except:
                    continue
        
        return None
    
    def _find_source_message(self, position: int, thread_content: str, messages: List[Dict[str, Any]]) -> str:
        """Find the message that contains the given position."""
        # This is simplified - in production, track message positions
        if messages:
            return messages[-1].get('message_id', '')
        return ''
    
    def _extract_context(self, position: int, thread_content: str) -> str:
        """Extract context around a position."""
        start = max(0, position - 200)
        end = min(len(thread_content), position + 200)
        return thread_content[start:end].strip()
    
    def _calculate_pattern_confidence(self, task_text: str, match) -> float:
        """Calculate confidence score for pattern-based extraction."""
        base_confidence = 0.6
        
        # Boost for longer tasks
        if len(task_text) > 20:
            base_confidence += 0.1
        
        # Boost for specific action verbs
        action_verbs = ['send', 'call', 'schedule', 'complete', 'review', 'approve', 'prepare']
        if any(verb in task_text.lower() for verb in action_verbs):
            base_confidence += 0.1
        
        # Boost for specific patterns
        if 'by' in task_text.lower() or 'before' in task_text.lower():
            base_confidence += 0.1
        
        return min(1.0, base_confidence)
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from task text."""
        # Simple keyword extraction
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Filter out common words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'will', 'please'}
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        return list(set(keywords))  # Remove duplicates
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings."""
        # Simple word overlap similarity
        words1 = set(self._extract_keywords(text1))
        words2 = set(self._extract_keywords(text2))
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _find_message_for_task(self, task_text: str, messages: List[Dict[str, Any]]) -> str:
        """Find the most likely message containing a task."""
        best_match = ''
        best_score = 0.0
        
        for message in messages:
            body = message.get('body_text', '')
            if task_text.lower() in body.lower():
                return message.get('message_id', '')
            
            # Check for partial matches
            similarity = self._calculate_similarity(task_text, body[:500])
            if similarity > best_score:
                best_score = similarity
                best_match = message.get('message_id', '')
        
        return best_match
    
    def _parse_relative_date(self, date_str: str) -> Optional[datetime]:
        """Parse relative date expressions."""
        date_str = date_str.lower()
        
        for expression, days in self.TIME_EXPRESSIONS.items():
            if expression in date_str:
                return datetime.utcnow() + timedelta(days=days)
        
        return None
