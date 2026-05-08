"""
Qdrant Vector Store Integration for MailMind

This module provides vector storage and retrieval capabilities using Qdrant
for semantic search and similarity matching of email threads.
"""

import asyncio
import logging
import os
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import uuid
import hashlib
import json

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    SearchParams,
    HasIdCondition,
    CollectionInfo
)
from qdrant_client.http.models import CollectionStatus
import numpy as np

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Qdrant-based vector store for MailMind thread embeddings.
    
    Features:
    - Thread embedding storage with metadata
    - Semantic search with filtering
    - Collection management
    - Batch operations
    """
    
    # Collection and vector configuration
    COLLECTION_NAME = "mailmind_threads"
    DISTANCE_METRIC = Distance.COSINE
    
    MAX_BATCH_SIZE = 100
    
    def __init__(self, qdrant_url: str, qdrant_api_key: str, openai_api_key: Optional[str] = None):
        """
        Initialize vector store.
        
        Args:
            qdrant_url: Qdrant server URL
            qdrant_api_key: Qdrant API key
            openai_api_key: OpenAI API key for embeddings (only needed if EMBEDDINGS_PROVIDER=openai)
        """
        self.qdrant_url = (qdrant_url or "").strip()
        self.qdrant_api_key = qdrant_api_key or ""
        self.client = self._create_client()

        self.embeddings_provider = os.getenv("EMBEDDINGS_PROVIDER", "openai").strip().lower()
        self.embedding_model: str
        self.vector_size: int
        self.openai_client: Optional[AsyncOpenAI] = None
        self._local_embedder = None

        if self.embeddings_provider in {"local", "fastembed"}:
            try:
                from fastembed import TextEmbedding
            except Exception:  # pragma: no cover
                # Some fastembed versions locate TextEmbedding elsewhere.
                from fastembed.embedding import TextEmbedding

            model_name = os.getenv("LOCAL_EMBED_MODEL", "BAAI/bge-small-en-v1.5").strip()
            self._local_embedder = TextEmbedding(model_name=model_name)
            probe_vec = next(self._local_embedder.embed(["dimension probe"]))
            self.vector_size = len(probe_vec)
            self.embedding_model = model_name
            logger.info(f"Using local embeddings model={model_name} dim={self.vector_size}")
        else:
            if not openai_api_key:
                raise ValueError("OPENAI_API_KEY is required when EMBEDDINGS_PROVIDER=openai")
            self.embedding_model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small").strip()
            self.vector_size = int(os.getenv("OPENAI_EMBED_DIM", "1536"))
            self.openai_client = AsyncOpenAI(api_key=openai_api_key)
            logger.info(f"Using OpenAI embeddings model={self.embedding_model} dim={self.vector_size}")

    def _create_client(self) -> QdrantClient:
        """
        Create a Qdrant client.

        Supports:
        - Remote: QDRANT_URL like http://localhost:6333
        - Embedded/local: QDRANT_URL=local (stores data on disk, no server needed)
        """
        if not self.qdrant_url or self.qdrant_url.lower() in {"local", "embedded"}:
            return QdrantClient(path="qdrant_local")

        return QdrantClient(url=self.qdrant_url, api_key=self.qdrant_api_key)
        
    async def initialize_collection(self) -> bool:
        """
        Initialize Qdrant collection for thread storage.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_exists = any(col.name == self.COLLECTION_NAME for col in collections)
            
            if not collection_exists:
                logger.info(f"Creating collection {self.COLLECTION_NAME}")
                
                # Create collection with vector configuration
                self.client.create_collection(
                    collection_name=self.COLLECTION_NAME,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=self.DISTANCE_METRIC
                    )
                )
                
                logger.info(f"Collection {self.COLLECTION_NAME} created successfully")
            else:
                logger.info(f"Collection {self.COLLECTION_NAME} already exists")
                
                # Verify collection configuration
                collection_info = self.client.get_collection(self.COLLECTION_NAME)
                try:
                    vectors_cfg = collection_info.config.params.vectors
                    existing_size = getattr(vectors_cfg, "size", None)
                    if existing_size is None and isinstance(vectors_cfg, dict):
                        existing_size = vectors_cfg.get("size")
                except Exception:
                    existing_size = None

                if existing_size and int(existing_size) != int(self.vector_size):
                    logger.warning(
                        f"Collection {self.COLLECTION_NAME} has dim={existing_size} but embedder dim={self.vector_size}; recreating collection"
                    )
                    self.client.delete_collection(collection_name=self.COLLECTION_NAME)
                    self.client.create_collection(
                        collection_name=self.COLLECTION_NAME,
                        vectors_config=VectorParams(
                            size=self.vector_size,
                            distance=self.DISTANCE_METRIC,
                        ),
                    )
                else:
                    logger.info(f"Collection info: {collection_info.config.params}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize collection: {str(e)}")

            # Auto-fallback to embedded/local mode for developer convenience.
            # This lets the app run without Docker/Qdrant server.
            try:
                logger.warning("Falling back to embedded Qdrant (local storage) at qdrant_local/")
                self.qdrant_url = "local"
                self.client = self._create_client()

                collections = self.client.get_collections().collections
                collection_exists = any(col.name == self.COLLECTION_NAME for col in collections)
                if not collection_exists:
                    self.client.create_collection(
                        collection_name=self.COLLECTION_NAME,
                        vectors_config=VectorParams(
                            size=self.vector_size,
                            distance=self.DISTANCE_METRIC
                        )
                    )
                return True
            except Exception as fallback_error:
                logger.error(f"Embedded Qdrant fallback failed: {str(fallback_error)}")
                return False
    
    async def embed_thread(self, thread_content: str) -> List[float]:
        """
        Generate embedding for thread content.
        
        Args:
            thread_content: Thread text to embed
            
        Returns:
            Embedding vector as list of floats
        """
        try:
            # Truncate content if too long (keeps latency predictable)
            max_chars = 8000  # Conservative estimate
            if len(thread_content) > max_chars:
                thread_content = thread_content[:max_chars] + "..."

            if self.embeddings_provider in {"local", "fastembed"}:
                vec = next(self._local_embedder.embed([thread_content]))
                embedding = vec.tolist() if hasattr(vec, "tolist") else list(vec)
                return embedding

            if not self.openai_client:
                raise RuntimeError("OpenAI client not initialized for embeddings")

            response = await self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=thread_content,
            )
            return response.data[0].embedding
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {str(e)}")
            raise
    
    def _generate_content_hash(self, content: str) -> str:
        """
        Generate hash for content to detect duplicates.
        
        Args:
            content: Thread content
            
        Returns:
            SHA-256 hash of content
        """
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    async def store_thread(
        self,
        thread_id: str,
        user_id: str,
        thread_content: str,
        metadata: Dict[str, Any]
    ) -> str:
        """
        Store thread embedding and metadata in Qdrant.
        
        Args:
            thread_id: Gmail thread ID
            user_id: User ID
            thread_content: Thread text content
            metadata: Thread metadata (participants, dates, tasks, etc.)
            
        Returns:
            Vector point ID
        """
        try:
            # Generate embedding
            embedding = await self.embed_thread(thread_content)
            
            # Generate content hash for deduplication
            content_hash = self._generate_content_hash(thread_content)
            
            # Create point ID
            point_id = str(uuid.uuid4())
            
            # Prepare point payload
            payload = {
                "thread_id": thread_id,
                "user_id": user_id,
                "content_hash": content_hash,
                "subject": metadata.get("subject", ""),
                "participant_emails": metadata.get("participant_emails", []),
                "message_count": metadata.get("message_count", 0),
                "first_message_date": metadata.get("first_message_date"),
                "last_message_date": metadata.get("last_message_date"),
                "detected_tasks": metadata.get("detected_tasks", []),
                "referenced_projects": metadata.get("referenced_projects", []),
                "referenced_urls": metadata.get("referenced_urls", []),
                "referenced_invoices": metadata.get("referenced_invoices", []),
                "has_attachments": metadata.get("has_attachments", False),
                # Enhanced intelligence payload (optional)
                "action_items": metadata.get("action_items", []),
                "entities": metadata.get("entities", []),
                "intelligence_metadata": metadata.get("intelligence_metadata", {}),
                "created_at": datetime.utcnow().isoformat(),
                "embedded_at": datetime.utcnow().isoformat()
            }
            
            # Create point
            point = PointStruct(
                id=point_id,
                vector=embedding,
                payload=payload
            )
            
            # Store in Qdrant
            self.client.upsert(
                collection_name=self.COLLECTION_NAME,
                points=[point]
            )
            
            logger.info(f"Stored thread {thread_id} with point ID {point_id}")
            return point_id
            
        except Exception as e:
            logger.error(f"Failed to store thread {thread_id}: {str(e)}")
            raise
    
    async def store_batch_threads(
        self,
        threads_data: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Store multiple threads in batch for better performance.
        
        Args:
            threads_data: List of thread dictionaries with keys:
                - thread_id: str
                - user_id: str
                - thread_content: str
                - metadata: Dict[str, Any]
                
        Returns:
            List of point IDs
        """
        try:
            point_ids = []
            points = []
            
            # Process in batches
            for i in range(0, len(threads_data), self.MAX_BATCH_SIZE):
                batch = threads_data[i:i + self.MAX_BATCH_SIZE]
                
                # Generate embeddings for batch
                embedding_tasks = []
                for thread_data in batch:
                    embedding_tasks.append(self.embed_thread(thread_data["thread_content"]))
                
                embeddings = await asyncio.gather(*embedding_tasks)
                
                # Create points
                for j, thread_data in enumerate(batch):
                    embedding = embeddings[j]
                    
                    # Generate content hash
                    content_hash = self._generate_content_hash(thread_data["thread_content"])
                    
                    # Create point ID
                    point_id = str(uuid.uuid4())
                    point_ids.append(point_id)
                    
                    # Prepare payload
                    metadata = thread_data["metadata"]
                    payload = {
                        "thread_id": thread_data["thread_id"],
                        "user_id": thread_data["user_id"],
                        "content_hash": content_hash,
                        "subject": metadata.get("subject", ""),
                        "participant_emails": metadata.get("participant_emails", []),
                        "message_count": metadata.get("message_count", 0),
                        "first_message_date": metadata.get("first_message_date"),
                        "last_message_date": metadata.get("last_message_date"),
                        "detected_tasks": metadata.get("detected_tasks", []),
                        "referenced_projects": metadata.get("referenced_projects", []),
                        "referenced_urls": metadata.get("referenced_urls", []),
                        "referenced_invoices": metadata.get("referenced_invoices", []),
                        "has_attachments": metadata.get("has_attachments", False),
                        "created_at": datetime.utcnow().isoformat(),
                        "embedded_at": datetime.utcnow().isoformat()
                    }
                    
                    point = PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload=payload
                    )
                    points.append(point)
                
                # Store batch in Qdrant
                self.client.upsert(
                    collection_name=self.COLLECTION_NAME,
                    points=points
                )
                
                logger.info(f"Stored batch of {len(points)} threads")
                points = []  # Reset for next batch
            
            return point_ids
            
        except Exception as e:
            logger.error(f"Failed to store batch threads: {str(e)}")
            raise
    
    async def search_similar_threads(
        self,
        query: str,
        user_id: str,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        score_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Search for similar threads using semantic similarity.
        
        Args:
            query: Search query text
            user_id: User ID to filter by
            limit: Maximum number of results
            filters: Additional filters (participants, date range, etc.)
            score_threshold: Minimum similarity score
            
        Returns:
            List of similar thread results
        """
        try:
            # Generate query embedding
            query_embedding = await self.embed_thread(query)
            
            # Build search filter
            search_filter = self._build_search_filter(user_id, filters)
            
            # Search parameters
            search_params = SearchParams(
                exact=True,  # Use exact search
                hnsw_ef=128   # HNSW search parameter
            )
            
            # Perform search
            search_result = self.client.search(
                collection_name=self.COLLECTION_NAME,
                query_vector=query_embedding,
                query_filter=search_filter,
                search_params=search_params,
                limit=limit,
                with_payload=True,
                with_vectors=False
            )
            
            # Process results
            results = []
            for hit in search_result:
                if hit.score >= score_threshold:
                    result = {
                        "thread_id": hit.payload["thread_id"],
                        "subject": hit.payload["subject"],
                        "score": hit.score,
                        "participant_emails": hit.payload["participant_emails"],
                        "message_count": hit.payload["message_count"],
                        "last_message_date": hit.payload["last_message_date"],
                        "detected_tasks": hit.payload["detected_tasks"],
                        "referenced_projects": hit.payload["referenced_projects"],
                        "has_attachments": hit.payload["has_attachments"],
                        "created_at": hit.payload["created_at"]
                    }
                    results.append(result)
            
            logger.info(f"Found {len(results)} similar threads for query: {query[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"Failed to search similar threads: {str(e)}")
            raise
    
    def _build_search_filter(self, user_id: str, filters: Optional[Dict[str, Any]]) -> Optional[Filter]:
        """
        Build Qdrant filter from search parameters.
        
        Args:
            user_id: User ID to filter by
            filters: Additional filter criteria
            
        Returns:
            Qdrant Filter object or None
        """
        conditions = []
        
        # Always filter by user_id
        conditions.append(
            FieldCondition(
                key="user_id",
                match=MatchValue(value=user_id)
            )
        )
        
        # Add additional filters
        if filters:
            # Filter by participants
            if "participants" in filters:
                for participant in filters["participants"]:
                    conditions.append(
                        FieldCondition(
                            key="participant_emails",
                            match=MatchValue(value=participant)
                        )
                    )
            
            # Filter by date range
            if "date_from" in filters:
                conditions.append(
                    FieldCondition(
                        key="last_message_date",
                        match=MatchValue(value={"gte": filters["date_from"]})
                    )
                )
            
            if "date_to" in filters:
                conditions.append(
                    FieldCondition(
                        key="last_message_date",
                        match=MatchValue(value={"lte": filters["date_to"]})
                    )
                )
            
            # Filter by projects
            if "projects" in filters:
                for project in filters["projects"]:
                    conditions.append(
                        FieldCondition(
                            key="referenced_projects",
                            match=MatchValue(value=project)
                        )
                    )
            
            # Filter by attachments
            if "has_attachments" in filters:
                conditions.append(
                    FieldCondition(
                        key="has_attachments",
                        match=MatchValue(value=filters["has_attachments"])
                    )
                )
        
        return Filter(must=conditions) if conditions else None
    
    async def get_thread_by_id(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve thread by thread ID.
        
        Args:
            thread_id: Gmail thread ID
            
        Returns:
            Thread data or None if not found
        """
        try:
            # Search by thread_id filter
            search_filter = Filter(
                must=[
                    FieldCondition(
                        key="thread_id",
                        match=MatchValue(value=thread_id)
                    )
                ]
            )
            
            search_result = self.client.scroll(
                collection_name=self.COLLECTION_NAME,
                scroll_filter=search_filter,
                limit=1,
                with_payload=True
            )
            
            points, _ = search_result
            
            if points:
                point = points[0]
                return {
                    "thread_id": point.payload["thread_id"],
                    "user_id": point.payload["user_id"],
                    "subject": point.payload["subject"],
                    "participant_emails": point.payload["participant_emails"],
                    "message_count": point.payload["message_count"],
                    "first_message_date": point.payload["first_message_date"],
                    "last_message_date": point.payload["last_message_date"],
                    "detected_tasks": point.payload["detected_tasks"],
                    "referenced_projects": point.payload["referenced_projects"],
                    "referenced_urls": point.payload["referenced_urls"],
                    "referenced_invoices": point.payload["referenced_invoices"],
                    "has_attachments": point.payload["has_attachments"],
                    "created_at": point.payload["created_at"],
                    "embedded_at": point.payload["embedded_at"]
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get thread {thread_id}: {str(e)}")
            raise
    
    async def delete_thread(self, thread_id: str) -> bool:
        """
        Delete thread from vector store.
        
        Args:
            thread_id: Gmail thread ID
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Find point by thread_id
            search_filter = Filter(
                must=[
                    FieldCondition(
                        key="thread_id",
                        match=MatchValue(value=thread_id)
                    )
                ]
            )
            
            search_result = self.client.scroll(
                collection_name=self.COLLECTION_NAME,
                scroll_filter=search_filter,
                limit=1,
                with_payload=False
            )
            
            points, _ = search_result
            
            if points:
                point_id = points[0].id
                self.client.delete(
                    collection_name=self.COLLECTION_NAME,
                    points_selector=point_id
                )
                logger.info(f"Deleted thread {thread_id} with point ID {point_id}")
                return True
            else:
                logger.warning(f"Thread {thread_id} not found for deletion")
                return False
                
        except Exception as e:
            logger.error(f"Failed to delete thread {thread_id}: {str(e)}")
            return False
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get collection statistics.
        
        Returns:
            Collection statistics
        """
        try:
            collection_info = self.client.get_collection(self.COLLECTION_NAME)
            
            return {
                "collection_name": self.COLLECTION_NAME,
                "vectors_count": collection_info.vectors_count,
                "indexed_vectors_count": collection_info.indexed_vectors_count,
                "points_count": collection_info.points_count,
                "status": collection_info.status,
                "optimizer_status": collection_info.optimizer_status,
                "vector_size": self.vector_size,
                "distance_metric": self.DISTANCE_METRIC.value
            }
            
        except Exception as e:
            logger.error(f"Failed to get collection stats: {str(e)}")
            raise
    
    async def health_check(self) -> bool:
        """
        Check if vector store is healthy and accessible.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            # Try to get collection info
            self.client.get_collection(self.COLLECTION_NAME)
            return True
        except Exception as e:
            logger.error(f"Vector store health check failed: {str(e)}")
            return False
