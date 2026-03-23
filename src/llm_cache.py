"""
LLM response caching and optimization for faster AI responses.
Reduces API calls and improves user experience.
"""

import hashlib
import json
import time
from typing import Optional, Dict
import functools

from .llm import chat as llm_chat


class LLMCache:
    """Cache for LLM responses to reduce API calls."""
    
    def __init__(self, max_size: int = 50, ttl_seconds: int = 3600):
        self.cache: Dict[str, Dict] = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
    
    def _get_key(self, predicted_class: str, confidence: float, question: str) -> str:
        """Generate cache key from inputs."""
        content = f"{predicted_class}_{confidence:.3f}_{question.lower().strip()}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, predicted_class: str, confidence: float, question: str) -> Optional[str]:
        """Get cached response if available and not expired."""
        key = self._get_key(predicted_class, confidence, question)
        if key in self.cache:
            cached_item = self.cache[key]
            if time.time() - cached_item['timestamp'] < self.ttl_seconds:
                return cached_item['response']
            else:
                # Remove expired item
                del self.cache[key]
        return None
    
    def set(self, predicted_class: str, confidence: float, question: str, response: str):
        """Cache a response."""
        key = self._get_key(predicted_class, confidence, question)
        
        # Remove oldest item if cache is full
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k]['timestamp'])
            del self.cache[oldest_key]
        
        self.cache[key] = {
            'response': response,
            'timestamp': time.time()
        }


# Global cache instance
_cache = LLMCache()


def fast_chat(predicted_class: str, confidence: float, question: str, api_key: Optional[str] = None) -> str:
    """
    Fast LLM chat with caching. Returns cached response if available.
    
    Args:
        predicted_class: Predicted tumor class
        confidence: Confidence score (0-1)
        question: User's question
        api_key: Optional Groq API key
        
    Returns:
        Assistant response text
    """
    # Check cache first
    cached_response = _cache.get(predicted_class, confidence, question)
    if cached_response:
        return cached_response
    
    # If not in cache, call LLM
    response = llm_chat(predicted_class, confidence, question, api_key)
    
    # Cache the response
    _cache.set(predicted_class, confidence, question, response)
    
    return response


# Common medical questions for instant responses
COMMON_RESPONSES = {
    "what is glioma": "Glioma is a type of tumor that arises from glial cells in the brain or spine. It's the most common primary brain tumor in adults.",
    "what is meningioma": "Meningioma is a tumor that arises from the meninges, the membranes surrounding the brain and spinal cord. It's typically benign but can cause serious symptoms.",
    "what is pituitary": "Pituitary tumors are abnormal growths in the pituitary gland. Most are benign but can affect hormone production and cause various symptoms.",
    "what is no tumor": "No tumor detected means the MRI scan appears normal with no evidence of abnormal growths or masses in the brain tissue.",
}


def get_instant_response(question: str) -> Optional[str]:
    """Get instant response for common questions."""
    question_lower = question.lower().strip()
    for key, response in COMMON_RESPONSES.items():
        if key in question_lower:
            return response
    return None


def chat_with_instant_fallback(predicted_class: str, confidence: float, question: str, api_key: Optional[str] = None) -> str:
    """
    Chat with instant fallback for common questions.
    """
    # Try instant response first
    instant_response = get_instant_response(question)
    if instant_response:
        return instant_response
    
    # Fall back to LLM
    return fast_chat(predicted_class, confidence, question, api_key)
