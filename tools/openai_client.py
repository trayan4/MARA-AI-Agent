"""
OpenAI API client wrapper with retry logic, error handling, and token counting.
"""

import asyncio
import os
import time
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from loguru import logger
import tiktoken

from config import settings

# Load environment variables
load_dotenv()


class OpenAIClient:
    """
    Wrapper for OpenAI API with automatic retry, error handling, and usage tracking.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize OpenAI client.
        
        Args:
            api_key: OpenAI API key. If None, reads from OPENAI_API_KEY env var
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY in .env file "
                "or pass api_key parameter."
            )
        
        # Initialize clients
        self.client = OpenAI(api_key=self.api_key)
        self.async_client = AsyncOpenAI(api_key=self.api_key)
        
        # Load settings
        self.model = settings.llm.model
        self.vision_model = settings.llm.vision_model
        self.temperature = settings.llm.temperature
        self.max_tokens = settings.llm.max_tokens
        self.timeout = settings.llm.timeout
        self.max_retries = settings.llm.max_retries
        self.retry_delay = settings.llm.retry_delay
        
        # Token counter
        self.encoding = tiktoken.encoding_for_model("gpt-4")
        
        # Usage tracking
        self.total_tokens = 0
        self.total_cost = 0.0
        
        logger.info(f"OpenAI client initialized with model: {self.model}")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encoding.encode(text))
    
    def count_messages_tokens(self, messages: List[Dict[str, str]]) -> int:
        """
        Count tokens in a list of messages.
        Approximates the token count for chat completions.
        """
        num_tokens = 0
        for message in messages:
            num_tokens += 4  # Every message follows <im_start>{role/name}\n{content}<im_end>\n
            for key, value in message.items():
                num_tokens += len(self.encoding.encode(str(value)))
                if key == "name":
                    num_tokens += -1  # Role is always 1 token, name is variable
        num_tokens += 2  # Every reply is primed with <im_start>assistant
        return num_tokens
    
    def _calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """
        Calculate approximate cost based on token usage.
        Prices as of January 2026.
        Source: https://openai.com/api/pricing/
        """
        pricing = {
            # GPT-4o models
            "gpt-4o": {"input": 0.0025, "output": 0.01},
            "gpt-4o-2024-11-20": {"input": 0.0025, "output": 0.01},
            "gpt-4o-2024-08-06": {"input": 0.0025, "output": 0.01},
            "gpt-4o-2024-05-13": {"input": 0.005, "output": 0.015},
            
            # GPT-4o mini models
            "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
            "gpt-4o-mini-2024-07-18": {"input": 0.00015, "output": 0.0006},
            
            # GPT-4 Turbo models
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            "gpt-4-turbo-2024-04-09": {"input": 0.01, "output": 0.03},
            "gpt-4-turbo-preview": {"input": 0.01, "output": 0.03},
            
            # GPT-4 models
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4-0613": {"input": 0.03, "output": 0.06},
            
            # Vision models (same as base models)
            "gpt-4-vision-preview": {"input": 0.01, "output": 0.03},
            "gpt-4o-vision": {"input": 0.0025, "output": 0.01},
        }
        
        # Default to GPT-4o pricing if model not found
        prices = pricing.get(model, {"input": 0.0025, "output": 0.01})
        
        cost = (input_tokens / 1000 * prices["input"]) + (output_tokens / 1000 * prices["output"])
        return cost
    
    def _handle_retry(self, attempt: int, error: Exception) -> None:
        """Handle retry logic with exponential backoff."""
        if attempt < self.max_retries:
            wait_time = self.retry_delay * (2 ** attempt)
            logger.warning(
                f"Request failed (attempt {attempt + 1}/{self.max_retries}): {error}. "
                f"Retrying in {wait_time}s..."
            )
            time.sleep(wait_time)
        else:
            logger.error(f"Request failed after {self.max_retries} retries: {error}")
            raise error
    
    async def _handle_retry_async(self, attempt: int, error: Exception) -> None:
        """Handle retry logic with exponential backoff (async version)."""
        if attempt < self.max_retries:
            wait_time = self.retry_delay * (2 ** attempt)
            logger.warning(
                f"Request failed (attempt {attempt + 1}/{self.max_retries}): {error}. "
                f"Retrying in {wait_time}s..."
            )
            await asyncio.sleep(wait_time)
        else:
            logger.error(f"Request failed after {self.max_retries} retries: {error}")
            raise error
    
    def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> ChatCompletion:
        """
        Synchronous chat completion with retry logic.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model to use (defaults to self.model)
            temperature: Sampling temperature (defaults to self.temperature)
            max_tokens: Max tokens to generate (defaults to self.max_tokens)
            **kwargs: Additional parameters to pass to OpenAI API
        
        Returns:
            ChatCompletion response
        """
        model = model or self.model
        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens or self.max_tokens
        
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=self.timeout,
                    **kwargs
                )
                
                # Track usage
                if response.usage:
                    input_tokens = response.usage.prompt_tokens
                    output_tokens = response.usage.completion_tokens
                    total = response.usage.total_tokens
                    
                    self.total_tokens += total
                    cost = self._calculate_cost(model, input_tokens, output_tokens)
                    self.total_cost += cost
                    
                    logger.debug(
                        f"Tokens used - Input: {input_tokens}, Output: {output_tokens}, "
                        f"Total: {total}, Cost: ${cost:.4f}"
                    )
                
                return response
            
            except Exception as e:
                self._handle_retry(attempt, e)
        
        raise RuntimeError("Should not reach here")
    
    async def chat_completion_async(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> ChatCompletion:
        """
        Async chat completion with retry logic.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model to use (defaults to self.model)
            temperature: Sampling temperature (defaults to self.temperature)
            max_tokens: Max tokens to generate (defaults to self.max_tokens)
            **kwargs: Additional parameters to pass to OpenAI API
        
        Returns:
            ChatCompletion response
        """
        model = model or self.model
        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens or self.max_tokens
        
        for attempt in range(self.max_retries + 1):
            try:
                response = await self.async_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=self.timeout,
                    **kwargs
                )
                
                # Track usage
                if response.usage:
                    input_tokens = response.usage.prompt_tokens
                    output_tokens = response.usage.completion_tokens
                    total = response.usage.total_tokens
                    
                    self.total_tokens += total
                    cost = self._calculate_cost(model, input_tokens, output_tokens)
                    self.total_cost += cost
                    
                    logger.debug(
                        f"Tokens used - Input: {input_tokens}, Output: {output_tokens}, "
                        f"Total: {total}, Cost: ${cost:.4f}"
                    )
                
                return response
            
            except Exception as e:
                await self._handle_retry_async(attempt, e)
        
        raise RuntimeError("Should not reach here")
    
    def extract_content(self, response: ChatCompletion) -> str:
        """Extract text content from ChatCompletion response."""
        return response.choices[0].message.content or ""
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get current usage statistics."""
        return {
            "total_tokens": self.total_tokens,
            "total_cost": round(self.total_cost, 4),
            "average_tokens_per_request": (
                round(self.total_tokens / max(1, self.total_tokens // 1000), 2)
            ),
        }
    
    def reset_usage_stats(self) -> None:
        """Reset usage tracking counters."""
        self.total_tokens = 0
        self.total_cost = 0.0
        logger.info("Usage stats reset")


# Singleton instance
_client: Optional[OpenAIClient] = None


def get_openai_client() -> OpenAIClient:
    """Get or create singleton OpenAI client instance."""
    global _client
    if _client is None:
        _client = OpenAIClient()
    return _client