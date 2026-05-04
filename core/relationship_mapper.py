"""
Cross-Thread Relationship Mapper for MailMind

This module detects and maps relationships between email threads based on
shared references like project codes, invoice numbers, and URLs.
"""

import asyncio
import logging
from typing import List, Dict, Any, Set, Tuple, Optional
from datetime import datetime
import re
import json
from dataclasses import dataclass
from enum import Enum
from urllib.parse import urlparse

from core.groq_client import get_async_groq_client

logger = logging.getLogger(__name__)


class RelationshipType(Enum):
    """Types of relationships between threads."""
    PROJECT = "project"
    INVOICE = "invoice"
    URL = "url"
    EMAIL_REFERENCE = "email_reference"
    DOCUMENT = "document"
    MEETING = "meeting"


@dataclass
class ThreadReference:
    """Represents a reference found in a thread."""
    reference_type: RelationshipType
    reference_value: str
    context: str
    confidence_score: float
    source_message_id: str


@dataclass
class ThreadRelationship:
    """Represents a relationship between two threads."""
    source_thread_id: str
    target_thread_id: str
    relationship_type: RelationshipType
    shared_reference: str
    confidence_score: int  # 1-10
    created_at: datetime


class ReferenceDetector:
    """
    Detects various types of references in email threads.
    """
    
    # Project code patterns
    PROJECT_PATTERNS = [
        r'\b([A-Z]{2,}-\d{3,})\b',  # PROJ-123, ABC-456
        r'\b([A-Z]{3,}\d{3,})\b',   # ABC123, XYZ789
        r'\b(project\s+\w+\s*#\s*\d+)\b',  # project Alpha #123
        r'\b(task\s+\w+\s*#\s*\d+)\b',    # task Beta #456
        r'\b(epic\s+\w+\s*#\s*\d+)\b',    # epic Gamma #789
    ]
    
    # Invoice number patterns
    INVOICE_PATTERNS = [
        r'\b(invoice\s+#?\s*\d{4,})\b',
        r'\b(inv\s+#?\s*\d{4,})\b',
        r'\b(receipt\s+#?\s*\d{4,})\b',
        r'\b(order\s+#?\s*\d{4,})\b',
        r'\b(po\s+#?\s*\d{4,})\b',  # Purchase Order
    ]
    
    # URL patterns
    URL_PATTERNS = [
        r'https?://[^\s<>"{}|\\^`[\]]+',
        r'www\.[^\s<>"{}|\\^`[\]]+\.[a-zA-Z]{2,}',
    ]
    
    # Document reference patterns
    DOCUMENT_PATTERNS = [
        r'\b(document\s+#?\s*\w+)\b',
        r'\b(doc\s+#?\s*\w+)\b',
        r'\b(file\s+#?\s*\w+)\b',
        r'\b(attachment\s+#?\s*\w+)\b',
    ]
    
    # Meeting reference patterns
    MEETING_PATTERNS = [
        r'\b(meeting\s+on\s+\d{1,2}/\d{1,2}/\d{4})\b',
        r'\b(call\s+on\s+\d{1,2}/\d{1,2}/\d{4})\b',
        r'\b(discussion\s+on\s+\d{1,2}/\d{1,2}/\d{4})\b',
        r'\b(conference\s+on\s+\d{1,2}/\d{1,2}/\d{4})\b',
    ]
    
    def __init__(self):
        """Initialize reference detector with compiled patterns."""
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Pre-compile regex patterns for better performance."""
        self.project_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.PROJECT_PATTERNS]
        self.invoice_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.INVOICE_PATTERNS]
        self.url_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.URL_PATTERNS]
        self.document_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.DOCUMENT_PATTERNS]
        self.meeting_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.MEETING_PATTERNS]
    
    def detect_references(self, thread_content: str, messages: List[Dict[str, Any]]) -> List[ThreadReference]:
        """
        Detect all references in thread content.
        
        Args:
            thread_content: Full thread content
            messages: List of message dictionaries
            
        Returns:
            List of detected references
        """
        references = []
        
        # Detect project references
        references.extend(self._detect_pattern_references(
            thread_content, messages, self.project_patterns, RelationshipType.PROJECT
        ))
        
        # Detect invoice references
        references.extend(self._detect_pattern_references(
            thread_content, messages, self.invoice_patterns, RelationshipType.INVOICE
        ))
        
        # Detect URL references
        references.extend(self._detect_pattern_references(
            thread_content, messages, self.url_patterns, RelationshipType.URL
        ))
        
        # Detect document references
        references.extend(self._detect_pattern_references(
            thread_content, messages, self.document_patterns, RelationshipType.DOCUMENT
        ))
        
        # Detect meeting references
        references.extend(self._detect_pattern_references(
            thread_content, messages, self.meeting_patterns, RelationshipType.MEETING
        ))
        
        return references
    
    def _detect_pattern_references(
        self,
        thread_content: str,
        messages: List[Dict[str, Any]],
        patterns: List[re.Pattern],
        reference_type: RelationshipType
    ) -> List[ThreadReference]:
        """Detect references using regex patterns."""
        references = []
        
        for pattern in patterns:
            matches = pattern.finditer(thread_content)
            
            for match in matches:
                try:
                    reference_value = match.group(1) if match.groups() else match.group(0)
                    
                    # Clean and normalize the reference
                    reference_value = self._normalize_reference(reference_value, reference_type)
                    
                    if not reference_value:
                        continue
                    
                    # Find source message
                    source_message_id = self._find_source_message(match.start(), thread_content, messages)
                    
                    # Extract context
                    context = self._extract_context(match.start(), thread_content)
                    
                    # Calculate confidence
                    confidence = self._calculate_pattern_confidence(reference_value, reference_type, match)
                    
                    reference = ThreadReference(
                        reference_type=reference_type,
                        reference_value=reference_value,
                        context=context,
                        confidence_score=confidence,
                        source_message_id=source_message_id
                    )
                    
                    references.append(reference)
                    
                except Exception as e:
                    logger.warning(f"Error processing pattern match: {str(e)}")
                    continue
        
        return references
    
    def _normalize_reference(self, reference: str, reference_type: RelationshipType) -> str:
        """Normalize reference value based on type."""
        reference = reference.strip().upper()
        
        if reference_type == RelationshipType.PROJECT:
            # Normalize project codes
            reference = re.sub(r'\s+', '', reference)
            if not re.match(r'^[A-Z]+-\d+$', reference):
                # Try to format as PROJECT-123
                match = re.match(r'^([A-Z]+)(\d+)$', reference)
                if match:
                    reference = f"{match.group(1)}-{match.group(2)}"
        
        elif reference_type == RelationshipType.INVOICE:
            # Normalize invoice numbers
            reference = re.sub(r'\s+', '', reference)
            # Ensure INVOICE prefix
            if not reference.startswith(('INVOICE', 'INV', 'RECEIPT', 'ORDER', 'PO')):
                reference = f"INVOICE-{reference}"
        
        elif reference_type == RelationshipType.URL:
            # Normalize URLs
            if not reference.startswith(('http://', 'https://')):
                reference = f"https://{reference}"
            
            # Parse and normalize
            try:
                parsed = urlparse(reference)
                reference = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
            except:
                pass
        
        return reference
    
    def _find_source_message(self, position: int, thread_content: str, messages: List[Dict[str, Any]]) -> str:
        """Find the message that contains the given position."""
        # This is simplified - in production, track message positions
        if messages:
            return messages[-1].get('message_id', '')
        return ''
    
    def _extract_context(self, position: int, thread_content: str) -> str:
        """Extract context around a position."""
        start = max(0, position - 100)
        end = min(len(thread_content), position + 100)
        return thread_content[start:end].strip()
    
    def _calculate_pattern_confidence(self, reference: str, reference_type: RelationshipType, match) -> float:
        """Calculate confidence score for pattern-based detection."""
        base_confidence = 0.7
        
        # Boost for specific formats
        if reference_type == RelationshipType.PROJECT:
            if re.match(r'^[A-Z]+-\d+$', reference):
                base_confidence += 0.2
        elif reference_type == RelationshipType.INVOICE:
            if reference.startswith(('INVOICE-', 'INV-', 'PO-')):
                base_confidence += 0.2
        elif reference_type == RelationshipType.URL:
            if '.' in reference and len(reference.split('.')) >= 2:
                base_confidence += 0.2
        
        # Boost for longer references (less likely to be false positive)
        if len(reference) > 5:
            base_confidence += 0.1
        
        return min(1.0, base_confidence)


class RelationshipMapper:
    """Maps relationships between email threads using AI analysis."""
    
    def __init__(self, openai_api_key: str = None):
        """Initialize the relationship mapper."""
        self.groq_client = None  # Will be initialized when needed
        self.model = os.getenv("RELATIONSHIP_MODEL", "llama3-70b-8192")
    
    async def _get_client(self):
        """Get Groq client instance."""
        if self.groq_client is None:
            self.groq_client = await get_async_groq_client()
        return self.groq_client
    
    async def map_thread_relationships(
        self,
        thread_id: str,
        thread_content: str,
        messages: List[Dict[str, Any]],
        existing_threads: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Map relationships between the current thread and existing threads.
        
        Args:
            thread_id: Current thread ID
            thread_content: Current thread content
            messages: Messages in current thread
            existing_threads: List of existing thread data
            
        Returns:
            List of relationship dictionaries
        """
        try:
            logger.info(f"Mapping relationships for thread {thread_id} against {len(existing_threads)} existing threads")
            
            # Step 1: Detect references in current thread
            current_references = self.reference_detector.detect_references(thread_content, messages)
            
            if not current_references:
                logger.info("No references detected in current thread")
                return []
            
            # Step 2: Find relationships with existing threads
            relationships = []
            
            for existing_thread in existing_threads:
                if existing_thread['thread_id'] == thread_id:
                    continue  # Skip self
                
                # Check for shared references
                shared_relationships = await self._find_shared_references(
                    thread_id,
                    current_references,
                    existing_thread
                )
                
                relationships.extend(shared_relationships)
            
            # Step 3: Apply semantic analysis for additional relationships
            semantic_relationships = await self._find_semantic_relationships(
                thread_id,
                thread_content,
                existing_threads
            )
            
            relationships.extend(semantic_relationships)
            
            # Step 4: Score and rank relationships
            scored_relationships = await self._score_relationships(relationships)
            
            logger.info(f"Found {len(scored_relationships)} thread relationships")
            return scored_relationships
            
        except Exception as e:
            logger.error(f"Error mapping thread relationships: {str(e)}")
            return []
    
    async def _find_shared_references(
        self,
        thread_id: str,
        current_references: List[ThreadReference],
        existing_thread: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Find relationships based on shared references."""
        relationships = []
        
        # Get existing thread references
        existing_references = existing_thread.get('references', [])
        
        for current_ref in current_references:
            for existing_ref in existing_references:
                # Check if references match
                if self._references_match(current_ref, existing_ref):
                    relationship = {
                        'source_thread_id': thread_id,
                        'target_thread_id': existing_thread['thread_id'],
                        'relationship_type': current_ref.reference_type.value,
                        'shared_reference': current_ref.reference_value,
                        'confidence_score': int(min(10, (current_ref.confidence_score + existing_ref.get('confidence_score', 0)) * 5)),
                        'created_at': datetime.utcnow().isoformat()
                    }
                    relationships.append(relationship)
        
        return relationships
    
    async def _find_semantic_relationships(
        self,
        thread_id: str,
        thread_content: str,
        existing_threads: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Find relationships using semantic analysis."""
        relationships = []
        
        # Limit to recent threads to avoid too many API calls
        recent_threads = existing_threads[:10]  # Take 10 most recent
        
        for existing_thread in recent_threads:
            if existing_thread['thread_id'] == thread_id:
                continue
            
            # Use LLM to detect semantic relationships
            semantic_rel = await self._detect_semantic_relationship(
                thread_id,
                thread_content,
                existing_thread
            )
            
            if semantic_rel:
                relationships.append(semantic_rel)
        
        return relationships
    
    async def _detect_semantic_relationship(
        self,
        thread_id: str,
        thread_content: str,
        existing_thread: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Detect semantic relationship between two threads."""
        try:
            existing_content = existing_thread.get('aggregated_content', '')
            
            if not existing_content:
                return None
            
            # Prepare prompt for LLM
            prompt = f"""
Analyze these two email threads and determine if they are related.

Thread A (Current):
{thread_content[:2000]}

Thread B (Existing):
{existing_content[:2000]}

Determine if these threads are related and if so, how:
1. Same project or topic
2. Same people involved
3. Continuation of conversation
4. Reference to same documents/events

Respond in JSON format:
{{
    "is_related": true/false,
    "relationship_type": "project/topic/people/continuation/reference",
    "confidence": 0.0-1.0,
    "explanation": "Brief explanation of the relationship"
}}"""
            
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert at determining relationships between email conversations. Respond in valid JSON format."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.1
            )
            
            content = response.choices[0].message.content.strip()
            
            try:
                data = json.loads(content)
                
                if data.get('is_related', False):
                    confidence = data.get('confidence', 0.5)
                    if confidence > 0.6:  # Only include high-confidence relationships
                        return {
                            'source_thread_id': thread_id,
                            'target_thread_id': existing_thread['thread_id'],
                            'relationship_type': 'semantic',
                            'shared_reference': data.get('relationship_type', 'related'),
                            'confidence_score': int(confidence * 10),
                            'created_at': datetime.utcnow().isoformat(),
                            'explanation': data.get('explanation', '')
                        }
                
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse semantic relationship JSON: {str(e)}")
            
            return None
            
        except Exception as e:
            logger.warning(f"Semantic relationship detection failed: {str(e)}")
            return None
    
    def _references_match(self, ref1: ThreadReference, ref2: Dict[str, Any]) -> bool:
        """Check if two references match."""
        if ref1.reference_type.value != ref2.get('reference_type'):
            return False
        
        # Exact match
        if ref1.reference_value == ref2.get('reference_value'):
            return True
        
        # Fuzzy match for certain types
        if ref1.reference_type == RelationshipType.PROJECT:
            # Project codes should match exactly after normalization
            return ref1.reference_value == ref2.get('reference_value')
        
        elif ref1.reference_type == RelationshipType.URL:
            # URLs should match after normalization
            return ref1.reference_value == ref2.get('reference_value')
        
        return False
    
    async def _score_relationships(self, relationships: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Score and rank relationships."""
        # Sort by confidence score
        return sorted(relationships, key=lambda x: x['confidence_score'], reverse=True)
    
    async def extract_thread_references(
        self,
        thread_content: str,
        messages: List[Dict[str, Any]]
    ) -> Dict[str, List[str]]:
        """
        Extract and categorize all references from a thread.
        
        Args:
            thread_content: Thread content
            messages: Message list
            
        Returns:
            Dictionary of reference types to lists of references
        """
        references = self.reference_detector.detect_references(thread_content, messages)
        
        categorized = {
            'projects': [],
            'invoices': [],
            'urls': [],
            'documents': [],
            'meetings': []
        }
        
        for ref in references:
            if ref.confidence_score > 0.5:  # Only include confident references
                if ref.reference_type == RelationshipType.PROJECT:
                    categorized['projects'].append(ref.reference_value)
                elif ref.reference_type == RelationshipType.INVOICE:
                    categorized['invoices'].append(ref.reference_value)
                elif ref.reference_type == RelationshipType.URL:
                    categorized['urls'].append(ref.reference_value)
                elif ref.reference_type == RelationshipType.DOCUMENT:
                    categorized['documents'].append(ref.reference_value)
                elif ref.reference_type == RelationshipType.MEETING:
                    categorized['meetings'].append(ref.reference_value)
        
        # Remove duplicates
        for category in categorized:
            categorized[category] = list(set(categorized[category]))
        
        return categorized
