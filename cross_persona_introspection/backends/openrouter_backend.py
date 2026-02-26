"""OpenRouter backend for optional LLM judging."""

import requests
from typing import Optional


class OpenRouterBackend:
    """Minimal OpenRouter chat completion client."""

    API_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, api_key: str, model: str = "anthropic/claude-3-haiku"):
        self.api_key = api_key
        self.model = model

    def chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        """Send a chat completion request. Returns the assistant's response text."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        resp = requests.post(self.API_URL, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        return data["choices"][0]["message"]["content"]
