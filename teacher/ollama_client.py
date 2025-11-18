"""
Ollama API client wrapper for Melvin Kindergarten Teacher.
"""

import requests
from typing import Dict, Any

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3"  # or any model name available in Ollama


def ollama_chat(system_prompt: str, user_prompt: str, model: str = None) -> str:
    """
    Call Ollama chat API with a system + user prompt.
    Return the assistant's content as a string.
    """
    if model is None:
        model = OLLAMA_MODEL
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    }
    
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        
        # Adjust this parsing if Ollama returns streaming or different schema
        msg = data.get("message") or data.get("messages", [{}])[-1]
        content = msg.get("content", "")
        return content
    except requests.exceptions.RequestException as e:
        print(f"Ollama API error: {e}")
        return ""
    except Exception as e:
        print(f"Unexpected error calling Ollama: {e}")
        return ""

