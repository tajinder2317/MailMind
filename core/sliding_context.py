"""
Sliding Context Processor for MailMind

This module implements the sliding context feature that generates running summaries
for long threads using GPT-4o-mini to keep content within token limits.
"""

import asyncio
import logging
import os
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import tiktoken
import json

from core.groq_client import get_async_groq_client
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MessageContent:
    """Represents a message with its content and metadata."""
    message_id: str
    from_email: str
    gmail_date: datetime
    body_text: str
    token_count: int


@dataclass
class ThreadSummary:
    """Represents a thread summary with metadata."""
    summary_text: str
    covered_message_ids: List[str]
    summary_token_count: int
    created_at: datetime


class SlidingContextProcessor:
    """
    Processes long email threads using sliding window approach with LLM summarization.
    
    Features:
    - Token counting using tiktoken
    - GPT-4o-mini summarization
    - Running summary generation
    - Context window management
    """
    
    # Token limits (conservative estimates)
    MAX_CONTEXT_TOKENS = 4000
    SUMMARY_TARGET_TOKENS = 800
    MIN_MESSAGES_FOR_SUMMARY = 4  # Only summarize if we have at least this many messages

    # Retry settings for summary generation/extension
    MAX_RETRIES = 3
    RETRY_DELAY = 1  # seconds
    
    # OpenAI model settings

    def __init__(self, tokenizer_model: str = "gpt-4o-mini"):
        # Tokenizer used only for rough token counting.
        try:
            self.tokenizer = tiktoken.encoding_for_model(tokenizer_model)
        except Exception:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

        self.groq_client = None
    
    async def process_thread_content(
        self,
        messages: List[Dict[str, Any]],
        existing_summary: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process thread content using sliding context approach.
        
        Args:
            messages: List of message dictionaries with keys:
                - message_id: str
                - from_email: str
                - gmail_date: datetime
                - body_text: str
            existing_summary: Optional existing running summary
            
        Returns:
            Dictionary containing:
                - processed_content: str (final content for embedding)
                - has_summary: bool
                - running_summary: Optional[str]
                - token_count: int
                - covered_messages: List[str]
        """
        try:
            enable_summary = os.getenv("ENABLE_SLIDING_CONTEXT_SUMMARY", "true").strip().lower() in {"1", "true", "yes", "y"}
            # Convert messages to MessageContent objects
            message_contents = []
            for msg in messages:
                token_count = self._count_tokens(msg['body_text'])
                message_contents.append(MessageContent(
                    message_id=msg['message_id'],
                    from_email=msg['from_email'],
                    gmail_date=msg['gmail_date'],
                    body_text=msg['body_text'],
                    token_count=token_count
                ))
            
            # Sort messages chronologically
            message_contents.sort(key=lambda x: x.gmail_date)
            
            # Calculate total tokens
            total_tokens = sum(msg.token_count for msg in message_contents)
            
            logger.info(f"Processing thread with {len(messages)} messages, {total_tokens} tokens")
            
            if not enable_summary:
                content = self._concatenate_messages(message_contents)
                return {
                    'processed_content': content,
                    'has_summary': False,
                    'running_summary': None,
                    'token_count': total_tokens,
                    'covered_messages': [msg.message_id for msg in message_contents],
                }

            # If within token limit, return as-is
            if total_tokens <= self.MAX_CONTEXT_TOKENS:
                content = self._concatenate_messages(message_contents)
                return {
                    'processed_content': content,
                    'has_summary': False,
                    'running_summary': None,
                    'token_count': total_tokens,
                    'covered_messages': [msg.message_id for msg in message_contents]
                }
            
            # Apply sliding context processing
            result = await self._apply_sliding_context(message_contents, existing_summary)
            
            logger.info(f"Sliding context applied: summary={result['has_summary']}, final_tokens={result['token_count']}")
            return result
            
        except Exception as e:
            logger.error(f"Error in sliding context processing: {str(e)}")
            # Fallback to simple concatenation
            content = self._concatenate_messages([
                MessageContent(
                    message_id=msg['message_id'],
                    from_email=msg['from_email'],
                    gmail_date=msg['gmail_date'],
                    body_text=msg['body_text'],
                    token_count=self._count_tokens(msg['body_text'])
                )
                for msg in messages
            ])
            return {
                'processed_content': content,
                'has_summary': False,
                'running_summary': None,
                'token_count': self._count_tokens(content),
                'covered_messages': [msg['message_id'] for msg in messages],
                'error': str(e)
            }
    
    async def _apply_sliding_context(
        self,
        messages: List[MessageContent],
        existing_summary: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Apply sliding context algorithm to generate summary and final content.
        
        Args:
            messages: Chronologically sorted messages
            existing_summary: Optional existing summary to extend
            
        Returns:
            Processing result dictionary
        """
        if len(messages) < self.MIN_MESSAGES_FOR_SUMMARY:
            # Not enough messages for meaningful summary
            content = self._concatenate_messages(messages)
            return {
                'processed_content': content,
                'has_summary': False,
                'running_summary': None,
                'token_count': sum(msg.token_count for msg in messages),
                'covered_messages': [msg.message_id for msg in messages]
            }
        
        # Calculate how many messages to keep in full (latest 3)
        latest_messages = messages[-3:] if len(messages) >= 3 else messages[-len(messages):]
        earlier_messages = messages[:-3] if len(messages) >= 3 else []
        
        # Generate or update summary for earlier messages
        if earlier_messages:
            if existing_summary:
                # Extend existing summary
                summary = await self._extend_summary(existing_summary, earlier_messages)
                covered_message_ids = self._extract_covered_messages_from_summary(existing_summary) + \
                                   [msg.message_id for msg in earlier_messages]
            else:
                # Generate new summary
                summary = await self._generate_summary(earlier_messages)
                covered_message_ids = [msg.message_id for msg in earlier_messages]
        else:
            summary = existing_summary
            covered_message_ids = self._extract_covered_messages_from_summary(existing_summary) if existing_summary else []
        
        # Combine summary with latest messages
        final_content_parts = []
        
        if summary:
            final_content_parts.append(f"[THREAD SUMMARY]\n{summary}\n")
        
        final_content_parts.append("[LATEST MESSAGES]")
        final_content_parts.extend(self._format_message(msg) for msg in latest_messages)
        
        final_content = "\n\n".join(final_content_parts)
        final_token_count = self._count_tokens(final_content)
        
        return {
            'processed_content': final_content,
            'has_summary': True,
            'running_summary': summary,
            'token_count': final_token_count,
            'covered_messages': covered_message_ids + [msg.message_id for msg in latest_messages]
        }

    async def _generate_summary(
        self,
        messages: List[MessageContent],
        context: str = ""
    ) -> str:
        """Generate a summary of the given messages using Groq."""
        try:
            # Get Groq client
            client = await self._get_client()
            
            # Build the prompt
            messages_text = "\n\n".join([
                f"From: {msg.from_email}\nDate: {msg.gmail_date}\n{msg.body_text}"
                for msg in messages
            ])
            
            prompt = f"""
You are an expert email thread summarizer. Create a concise summary of the following email messages.

Context: {context}

Messages to summarize:
{messages_text}

Requirements:
1. Create a clear, concise summary (max {self.SUMMARY_TARGET_TOKENS} tokens)
2. Focus on key decisions, action items, and important information
3. Maintain chronological flow
4. Include participant names and their roles
5. Highlight any deadlines or commitments
6. Preserve critical context for future messages

Summary:
"""
            
            response = await client.chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert email thread summarizer. Create clear, concise summaries that capture all essential information."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model=os.getenv("SUMMARY_MODEL", os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")),
                max_tokens=self.SUMMARY_TARGET_TOKENS,
                temperature=0.3
            )
            
            summary = response.choices[0].message.content.strip()
            logger.info(f"Generated summary: {len(summary)} characters")
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate summary: {str(e)}")
            return f"Summary generation failed: {str(e)}"
    
    async def _extend_summary(self, existing_summary: str, new_messages: List[MessageContent]) -> str:
        """
        Extend an existing summary with new messages.
        
        Args:
            existing_summary: Current running summary
            new_messages: New messages to incorporate
            
        Returns:
            Updated summary text
        """
        new_messages_text = "\n\n".join(self._format_message(msg) for msg in new_messages)
        
        prompt = f"""
Please update the existing email thread summary to incorporate these new messages.

Existing Summary:
{existing_summary}

New Messages to Add:
{new_messages_text}

Instructions:
1. Integrate the new information seamlessly into the existing summary
2. Update any action items, decisions, or deadlines
3. Maintain the same concise style (around {self.SUMMARY_TARGET_TOKENS} tokens)
4. Preserve the metadata format at the end

Updated Summary:"""
        
        client = await self._get_client()

        for attempt in range(self.MAX_RETRIES):
            try:
                response = await client.chat_completion(
                    messages=[
                        {
                            "role": "system",
                            "content": "You update email thread summaries while maintaining accuracy and conciseness."
                        },
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=1200,
                    temperature=0.3,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                logger.warning(f"Summary extension attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.MAX_RETRIES - 1:
                    await asyncio.sleep(self.RETRY_DELAY * (2 ** attempt))
                # Fallback: append new messages to existing summary
                return f"{existing_summary}\n\n[NEW_MESSAGES]\n{new_messages_text}"
        
        raise Exception("Failed to extend summary after all retries")
    
    def _count_tokens(self, text: str) -> int:
        """
        Count tokens in text using tiktoken.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of tokens
        """
        try:
            return len(self.tokenizer.encode(text))
        except Exception as e:
            logger.warning(f"Token counting failed: {str(e)}")
            # Fallback: rough estimate (1 token ≈ 4 characters)
            return len(text) // 4
    
    def _concatenate_messages(self, messages: List[MessageContent]) -> str:
        """
        Concatenate messages into a single string.
        
        Args:
            messages: Messages to concatenate
            
        Returns:
            Concatenated content
        """
        formatted_messages = [self._format_message(msg) for msg in messages]
        return "\n\n".join(formatted_messages)
    
    def _format_message(self, message: MessageContent) -> str:
        """
        Format a message for processing.
        
        Args:
            message: Message to format
            
        Returns:
            Formatted message string
        """
        date_str = message.gmail_date.strftime("%Y-%m-%d %H:%M")
        return f"[{date_str}] {message.from_email}:\n{message.body_text}"
    
    def _extract_covered_messages_from_summary(self, summary: str) -> List[str]:
        """
        Extract message IDs covered by a summary from metadata.
        
        Args:
            summary: Summary text with metadata
            
        Returns:
            List of covered message IDs (empty if not found)
        """
        try:
            # Look for metadata in summary
            if "[SUMMARY_METADATA:" in summary:
                metadata_start = summary.find("[SUMMARY_METADATA:")
                metadata_end = summary.find("]", metadata_start)
                if metadata_end != -1:
                    metadata_str = summary[metadata_start:metadata_end + 1]
                    # For now, return empty list as we don't store individual message IDs
                    # In a future enhancement, we could store this information differently
                    return []
            return []
        except Exception:
            return []
    
    async def test_summary_generation(self, test_messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Test summary generation with sample messages.
        
        Args:
            test_messages: Sample messages for testing
            
        Returns:
            Test results
        """
        try:
            start_time = datetime.utcnow()
            result = await self.process_thread_content(test_messages)
            end_time = datetime.utcnow()
            
            return {
                'success': True,
                'processing_time_ms': int((end_time - start_time).total_seconds() * 1000),
                'input_messages': len(test_messages),
                'output_tokens': result['token_count'],
                'has_summary': result['has_summary'],
                'summary_length': len(result['running_summary']) if result['running_summary'] else 0
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'processing_time_ms': 0
            }
