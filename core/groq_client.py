"""
Groq Client for MailMind

High-speed LLM client using Groq's API with rate limiting and retry logic.
"""

import os
import time
import asyncio
import logging
from typing import List, Dict, Any, Optional
from openai import OpenAI, AsyncOpenAI

logger = logging.getLogger(__name__)


class GroqClient:
    """
    Groq client with rate limiting and retry logic for MailMind.
    
    Uses Groq's high-speed LLM models with automatic rate limiting
    to stay within free tier limits.
    """
    
    def __init__(self):
        """Initialize Groq client with configuration."""
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY environment variable is required")
        
        self.base_url = "https://groq.com"
        self.model = os.getenv("LLM_MODEL", "llama3-70b-8192")
        
        # Rate limiting configuration
        self.call_delay = float(os.getenv("LLM_CALL_DELAY", "2.0"))
        self.max_concurrent = int(os.getenv("MAX_CONCURRENT_LLM_REQUESTS", "5"))
        self.max_retries = int(os.getenv("LLM_MAX_RETRIES", "3"))
        self.retry_delay = float(os.getenv("LLM_RETRY_DELAY", "5.0"))
        
        # Semaphore for concurrent request limiting
        self.semaphore = asyncio.Semaphore(self.max_concurrent)
        
        # Initialize clients
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        self.async_client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        # Track last call time for rate limiting
        self.last_call_time = 0.0
        
        logger.info(f"Groq client initialized with model: {self.model}")
        logger.info(f"Rate limiting: {self.call_delay}s delay, max {self.max_concurrent} concurrent")
    
    async def _rate_limit(self):
        """Apply rate limiting between calls."""
        current_time = time.time()
        time_since_last = current_time - self.last_call_time
        
        if time_since_last < self.call_delay:
            sleep_time = self.call_delay - time_since_last
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f}s")
            await asyncio.sleep(sleep_time)
        
        self.last_call_time = time.time()
    
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: float = 0.3,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create chat completion with rate limiting and retry logic.
        
        Args:
            messages: List of message dictionaries
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            **kwargs: Additional OpenAI parameters
            
        Returns:
            Chat completion response
        """
        async with self.semaphore:
            await self._rate_limit()
            
            for attempt in range(self.max_retries + 1):
                try:
                    logger.debug(f"Groq API call attempt {attempt + 1}/{self.max_retries + 1}")
                    
                    response = await self.async_client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        **kwargs
                    )
                    
                    logger.debug(f"Groq API call successful")
                    return response
                    
                except Exception as e:
                    logger.warning(f"Groq API call attempt {attempt + 1} failed: {str(e)}")
                    
                    if attempt < self.max_retries:
                        # Exponential backoff for retries
                        retry_delay = self.retry_delay * (2 ** attempt)
                        logger.info(f"Retrying in {retry_delay}s...")
                        await asyncio.sleep(retry_delay)
                    else:
                        logger.error(f"All Groq API retry attempts failed: {str(e)}")
                        raise
    
    async def extract_json_response(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: float = 0.3
    ) -> Dict[str, Any]:
        """
        Create chat completion and extract JSON response.
        
        Args:
            messages: List of message dictionaries
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            
        Returns:
            Parsed JSON response
        """
        response = await self.chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        content = response.choices[0].message.content
        
        # Clean up the response
        content = content.strip()
        
        # Extract JSON if it's wrapped in code blocks
        if content.startswith("```json"):
            content = content[7:-3].strip()
        elif content.startswith("```"):
            content = content[3:-3].strip()
        
        # Parse JSON
        import json
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {str(e)}")
            logger.error(f"Raw content: {content}")
            raise ValueError(f"Invalid JSON response: {str(e)}")
    
    def sync_chat_completion(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: float = 0.3,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Synchronous chat completion for compatibility.
        
        Args:
            messages: List of message dictionaries
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            **kwargs: Additional OpenAI parameters
            
        Returns:
            Chat completion response
        """
        # Apply rate limiting
        current_time = time.time()
        time_since_last = current_time - self.last_call_time
        
        if time_since_last < self.call_delay:
            sleep_time = self.call_delay - time_since_last
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f}s")
            time.sleep(sleep_time)
        
        self.last_call_time = time.time()
        
        for attempt in range(self.max_retries + 1):
            try:
                logger.debug(f"Groq API sync call attempt {attempt + 1}/{self.max_retries + 1}")
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **kwargs
                )
                
                logger.debug(f"Groq API sync call successful")
                return response
                
            except Exception as e:
                logger.warning(f"Groq API sync call attempt {attempt + 1} failed: {str(e)}")
                
                if attempt < self.max_retries:
                    # Exponential backoff for retries
                    retry_delay = self.retry_delay * (2 ** attempt)
                    logger.info(f"Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"All Groq API sync retry attempts failed: {str(e)}")
                    raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        return {
            "model": self.model,
            "api_provider": "Groq",
            "base_url": self.base_url,
            "rate_limiting": {
                "call_delay": self.call_delay,
                "max_concurrent": self.max_concurrent,
                "max_retries": self.max_retries,
                "retry_delay": self.retry_delay
            }
        }
    
    async def test_connection(self) -> bool:
        """Test connection to Groq API."""
        try:
            messages = [{"role": "user", "content": "Hello, this is a test message."}]
            response = await self.chat_completion(messages, max_tokens=10)
            
            if response.choices and len(response.choices) > 0:
                logger.info("Groq API connection test successful")
                return True
            else:
                logger.error("Groq API connection test failed: no choices in response")
                return False
                
        except Exception as e:
            logger.error(f"Groq API connection test failed: {str(e)}")
            return False


# Global Groq client instance
_groq_client = None


def get_groq_client() -> GroqClient:
    """Get or create the global Groq client instance."""
    global _groq_client
    if _groq_client is None:
        _groq_client = GroqClient()
    return _groq_client


async def get_async_groq_client() -> GroqClient:
    """Get or create the async Groq client instance."""
    return get_groq_client()
