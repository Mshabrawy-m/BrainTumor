"""
Groq LLM integration for Brain Tumor MRI Q&A.
Uses openai/gpt-oss-120b model.
"""

import os
from typing import Optional

from groq import Groq
from groq.types.chat import ChatCompletion

from .prompts import SYSTEM_PROMPT, build_user_prompt

GROQ_MODEL = "openai/gpt-oss-120b"


def get_groq_client(api_key: Optional[str] = None) -> Groq:
    """Create Groq client. Uses GROQ_API_KEY env var if api_key not provided."""
    key = api_key or os.environ.get("GROQ_API_KEY")
    if not key:
        raise ValueError(
            "Groq API key not found. Set GROQ_API_KEY environment variable "
            "or pass api_key to the chat function."
        )
    return Groq(api_key=key)


def chat(
    predicted_class: str,
    confidence: float,
    question: str,
    api_key: Optional[str] = None
) -> str:
    """
    Send user question with prediction context to Groq LLM.
    
    Args:
        predicted_class: Predicted tumor class
        confidence: Confidence score (0-1)
        question: User's question
        api_key: Optional Groq API key (else uses GROQ_API_KEY)
        
    Returns:
        Assistant response text
    """
    client = get_groq_client(api_key=api_key)
    user_prompt = build_user_prompt(predicted_class, confidence, question)
    
    response: ChatCompletion = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=1,
        max_tokens=8192,
        top_p=1,
        stream=False,
        stop=None
    )
    
    message = response.choices[0].message
    return message.content if message.content else ""
