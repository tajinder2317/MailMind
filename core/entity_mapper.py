"""
Entity-Based Relationship Mapper for MailMind

This module creates relationships between threads based on shared entities
like project codes, invoice numbers, JIRA tickets, etc.
"""

import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum

from core.embedding_pipeline import ExtractedEntity, EntityType

logger = logging.getLogger(__name__)


@dataclass
class EntityRelationship:
    """Represents a relationship between threads based on shared entities."""
    source_thread_id: str
    target_thread_id: str
    entity_type: str
    entity_value: str
    confidence_score: int  # 1-10
    created_at: datetime


class EntityRelationshipMapper:
    """
    Maps relationships between threads based on shared entities.
    
    When two different thread_ids share the same entity (project code, invoice #, JIRA ID),
    this creates a record in the ThreadRelationship table.
    """
    
    def __init__(self):
        """Initialize the entity relationship mapper."""
        self.entity_weights = {
            EntityType.PROJECT: 9,      # High confidence for project codes
            EntityType.INVOICE: 8,      # High confidence for invoices
            EntityType.JIRA: 9,         # High confidence for JIRA tickets
            EntityType.MEETING: 6,       # Medium confidence for meetings
            EntityType.DOCUMENT: 5,      # Lower confidence for documents
            EntityType.URL: 4,          # Lower confidence for URLs
            EntityType.PERSON: 3,       # Low confidence for people
        }
    
    async def create_entity_relationships(
        self,
        thread_id: str,
        user_id: str,
        entities: List[ExtractedEntity],
        existing_threads: List[Dict[str, Any]]
    ) -> List[EntityRelationship]:
        """
        Create relationships between threads based on shared entities.
        
        Args:
            thread_id: Current thread ID
            user_id: User ID
            entities: Entities extracted from current thread
            existing_threads: List of existing thread data with their entities
            
        Returns:
            List of created entity relationships
        """
        try:
            logger.info(f"Creating entity relationships for thread {thread_id} with {len(entities)} entities")
            
            relationships = []
            
            # Group entities by type and value for efficient lookup
            entity_map = self._group_entities_by_type_value(entities)
            
            # Check each existing thread for shared entities
            for existing_thread in existing_threads:
                if existing_thread['thread_id'] == thread_id:
                    continue  # Skip self
                
                existing_entities = existing_thread.get('entities', [])
                existing_entity_map = self._group_entities_by_type_value(existing_entities)
                
                # Find shared entities
                shared_entities = self._find_shared_entities(entity_map, existing_entity_map)
                
                # Create relationships for shared entities
                for entity_type, entity_value in shared_entities:
                    confidence = self._calculate_relationship_confidence(
                        entity_type, entity_value, entities, existing_entities
                    )
                    
                    relationship = EntityRelationship(
                        source_thread_id=thread_id,
                        target_thread_id=existing_thread['thread_id'],
                        entity_type=entity_type.value,
                        entity_value=entity_value,
                        confidence_score=confidence,
                        created_at=datetime.utcnow()
                    )
                    
                    relationships.append(relationship)
            
            logger.info(f"Created {len(relationships)} entity relationships")
            return relationships
            
        except Exception as e:
            logger.error(f"Failed to create entity relationships: {str(e)}")
            return []
    
    def _group_entities_by_type_value(self, entities: List[Dict[str, Any]]) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
        """
        Group entities by type and value for efficient lookup.
        
        Args:
            entities: List of entity dictionaries
            
        Returns:
            Dictionary mapping (entity_type, entity_value) to list of entities
        """
        entity_map = {}
        
        for entity in entities:
            entity_type = entity.get('entity_type')
            entity_value = entity.get('entity_value')
            
            if entity_type and entity_value:
                key = (entity_type, entity_value)
                if key not in entity_map:
                    entity_map[key] = []
                entity_map[key].append(entity)
        
        return entity_map
    
    def _find_shared_entities(
        self,
        current_entity_map: Dict[Tuple[str, str], List[Dict[str, Any]]],
        existing_entity_map: Dict[Tuple[str, str], List[Dict[str, Any]]]
    ) -> Set[Tuple[EntityType, str]]:
        """
        Find shared entities between two entity maps.
        
        Args:
            current_entity_map: Current thread's entity map
            existing_entity_map: Existing thread's entity map
            
        Returns:
            Set of shared entity (type, value) tuples
        """
        shared_entities = set()
        
        for key in current_entity_map.keys():
            if key in existing_entity_map:
                entity_type_str, entity_value = key
                
                # Convert string back to enum
                try:
                    entity_type = EntityType(entity_type_str)
                    shared_entities.add((entity_type, entity_value))
                except ValueError:
                    # Skip unknown entity types
                    continue
        
        return shared_entities
    
    def _calculate_relationship_confidence(
        self,
        entity_type: EntityType,
        entity_value: str,
        current_entities: List[Dict[str, Any]],
        existing_entities: List[Dict[str, Any]]
    ) -> int:
        """
        Calculate confidence score for the relationship.
        
        Args:
            entity_type: Type of the shared entity
            entity_value: Value of the shared entity
            current_entities: All entities from current thread
            existing_entities: All entities from existing thread
            
        Returns:
            Confidence score (1-10)
        """
        # Base confidence from entity type
        base_confidence = self.entity_weights.get(entity_type, 5)
        
        # Boost confidence based on entity specificity
        specificity_boost = 0
        if entity_type == EntityType.PROJECT:
            # More specific project codes get higher confidence
            if len(entity_value) > 8:  # Long project codes are more specific
                specificity_boost = 1
        elif entity_type == EntityType.INVOICE:
            # Higher invoice numbers suggest more formal processes
            if entity_value.replace('-', '').replace('#', '').isdigit():
                if int(entity_value.replace('-', '').replace('#', '')) > 1000:
                    specificity_boost = 1
        elif entity_type == EntityType.JIRA:
            # JIRA tickets are usually reliable
            specificity_boost = 1
        
        # Boost confidence based on entity frequency (rare entities are more significant)
        frequency_boost = 0
        current_type_count = sum(1 for e in current_entities if e.get('entity_type') == entity_type.value)
        existing_type_count = sum(1 for e in existing_entities if e.get('entity_type') == entity_type.value)
        
        # If this is the only entity of this type in both threads, boost confidence
        if current_type_count == 1 and existing_type_count == 1:
            frequency_boost = 1
        
        # Calculate final confidence
        final_confidence = base_confidence + specificity_boost + frequency_boost
        return min(10, final_confidence)  # Cap at 10
    
    async def update_thread_relationships(
        self,
        thread_id: str,
        user_id: str,
        entities: List[ExtractedEntity]
    ) -> List[Dict[str, Any]]:
        """
        Update thread relationships in the database.
        
        Args:
            thread_id: Thread ID to update
            user_id: User ID
            entities: New entities for the thread
            
        Returns:
            List of created relationship records
        """
        try:
            # This would typically involve database operations
            # For now, return the relationship data structure
            
            relationships = []
            
            # Process each entity to find potential relationships
            for entity in entities:
                # In a real implementation, this would query the database
                # for other threads with the same entity
                relationship_data = {
                    "source_thread_id": thread_id,
                    "target_thread_id": f"related_to_{entity.entity_value}",
                    "relationship_type": entity.entity_type.value,
                    "shared_reference": entity.entity_value,
                    "confidence_score": self.entity_weights.get(entity.entity_type, 5),
                    "created_at": datetime.utcnow().isoformat()
                }
                relationships.append(relationship_data)
            
            return relationships
            
        except Exception as e:
            logger.error(f"Failed to update thread relationships: {str(e)}")
            return []
    
    async def find_related_threads(
        self,
        thread_id: str,
        user_id: str,
        entity_types: Optional[List[EntityType]] = None
    ) -> List[Dict[str, Any]]:
        """
        Find threads related to the given thread based on shared entities.
        
        Args:
            thread_id: Thread ID to find relationships for
            user_id: User ID
            entity_types: Optional filter for specific entity types
            
        Returns:
            List of related thread information
        """
        try:
            # This would typically query the ThreadRelationship table
            # For now, return a mock response
            
            related_threads = []
            
            # Mock related threads based on entity sharing
            mock_relationships = [
                {
                    "thread_id": f"related_to_{thread_id}_1",
                    "relationship_type": "project",
                    "shared_reference": "PROJ-123",
                    "confidence_score": 9,
                    "created_at": datetime.utcnow().isoformat()
                },
                {
                    "thread_id": f"related_to_{thread_id}_2", 
                    "relationship_type": "invoice",
                    "shared_reference": "INV-456",
                    "confidence_score": 8,
                    "created_at": datetime.utcnow().isoformat()
                }
            ]
            
            # Filter by entity types if specified
            if entity_types:
                entity_type_strings = [et.value for et in entity_types]
                mock_relationships = [
                    rel for rel in mock_relationships 
                    if rel["relationship_type"] in entity_type_strings
                ]
            
            return mock_relationships
            
        except Exception as e:
            logger.error(f"Failed to find related threads: {str(e)}")
            return []
    
    def get_entity_statistics(
        self,
        entities: List[ExtractedEntity]
    ) -> Dict[str, Any]:
        """
        Get statistics about entities in a thread.
        
        Args:
            entities: List of extracted entities
            
        Returns:
            Dictionary with entity statistics
        """
        stats = {
            "total_entities": len(entities),
            "entity_types": {},
            "high_confidence_entities": 0,
            "unique_values": set()
        }
        
        for entity in entities:
            entity_type = entity.entity_type.value
            entity_value = entity.entity_value
            
            # Count by type
            if entity_type not in stats["entity_types"]:
                stats["entity_types"][entity_type] = 0
            stats["entity_types"][entity_type] += 1
            
            # Count high confidence entities
            if entity.confidence_score >= 0.8:
                stats["high_confidence_entities"] += 1
            
            # Track unique values
            stats["unique_values"].add(entity_value)
        
        stats["unique_values"] = len(stats["unique_values"])
        
        return stats


class ThreadRelationshipManager:
    """
    Manages thread relationships in the database.
    """
    
    def __init__(self):
        """Initialize the relationship manager."""
        self.entity_mapper = EntityRelationshipMapper()
    
    async def process_thread_for_relationships(
        self,
        thread_id: str,
        user_id: str,
        entities: List[ExtractedEntity],
        existing_threads: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Process a thread for entity-based relationships.
        
        Args:
            thread_id: Thread ID
            user_id: User ID
            entities: Extracted entities from the thread
            existing_threads: Existing threads to compare against
            
        Returns:
            Processing results with relationship information
        """
        try:
            logger.info(f"Processing thread {thread_id} for entity relationships")
            
            # Create entity relationships
            relationships = await self.entity_mapper.create_entity_relationships(
                thread_id, user_id, entities, existing_threads
            )
            
            # Get entity statistics
            entity_stats = self.entity_mapper.get_entity_statistics(entities)
            
            # Prepare results
            results = {
                "thread_id": thread_id,
                "user_id": user_id,
                "entities_processed": len(entities),
                "relationships_created": len(relationships),
                "entity_statistics": entity_stats,
                "relationships": [
                    {
                        "target_thread_id": rel.target_thread_id,
                        "entity_type": rel.entity_type,
                        "entity_value": rel.entity_value,
                        "confidence_score": rel.confidence_score
                    }
                    for rel in relationships
                ],
                "processed_at": datetime.utcnow().isoformat()
            }
            
            logger.info(f"Processed thread {thread_id}: {len(relationships)} relationships created")
            return results
            
        except Exception as e:
            logger.error(f"Failed to process thread {thread_id} for relationships: {str(e)}")
            return {
                "thread_id": thread_id,
                "user_id": user_id,
                "error": str(e),
                "processed_at": datetime.utcnow().isoformat()
            }
    
    async def bulk_process_relationships(
        self,
        threads_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Process multiple threads for relationships in bulk.
        
        Args:
            threads_data: List of thread data with entities
            
        Returns:
            Bulk processing results
        """
        try:
            logger.info(f"Bulk processing {len(threads_data)} threads for relationships")
            
            total_relationships = 0
            processed_threads = 0
            errors = []
            
            for thread_data in threads_data:
                try:
                    thread_id = thread_data["thread_id"]
                    user_id = thread_data["user_id"]
                    entities = thread_data["entities"]
                    existing_threads = thread_data.get("existing_threads", [])
                    
                    # Process thread
                    results = await self.process_thread_for_relationships(
                        thread_id, user_id, entities, existing_threads
                    )
                    
                    if "error" not in results:
                        total_relationships += results["relationships_created"]
                        processed_threads += 1
                    else:
                        errors.append(f"Thread {thread_id}: {results['error']}")
                        
                except Exception as e:
                    errors.append(f"Thread {thread_data.get('thread_id', 'unknown')}: {str(e)}")
            
            return {
                "total_threads": len(threads_data),
                "processed_threads": processed_threads,
                "total_relationships": total_relationships,
                "errors": errors,
                "processed_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Bulk relationship processing failed: {str(e)}")
            return {
                "error": str(e),
                "processed_at": datetime.utcnow().isoformat()
            }
