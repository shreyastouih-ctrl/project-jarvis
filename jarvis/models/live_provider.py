"""Live LLM Provider connecting J.A.R.V.I.S. to ChatGPT, Groq, Gemini, or local Ollama."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional
import httpx

logger = logging.getLogger(__name__)

JARVIS_SYSTEM_PROMPT = (
    "You are J.A.R.V.I.S., the iconic personal artificial intelligence assistant created by Tony Stark in Iron Man. "
    "Speak in a courteous, refined, British butler-esque manner (frequently addressing the user as 'sir'). "
    "Keep responses concise (1 to 2 sentences) and conversational, optimized for spoken voice synthesis. "
    "Do NOT use markdown, emojis, asterisks, or bullet points, as your response is read aloud. "
    "You are strictly J.A.R.V.I.S., loyal, witty, and calm."
)


class LiveLLMProvider:
    """Connects to real LLM providers (OpenAI ChatGPT, Groq, Gemini, or Ollama) with fast fallback."""

    def __init__(self) -> None:
        self._load_env()
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.groq_key = os.getenv("GROQ_API_KEY")
        self.gemini_key = os.getenv("GEMINI_API_KEY")
        self.ollama_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")

    def _load_env(self) -> None:
        """Load .env from project root if present."""
        env_path = Path(".env")
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ[k.strip()] = v.strip().strip("'\"")

    async def generate(self, user_query: str) -> Optional[str]:
        """Generate response via available low-latency model, returning None if no provider available."""
        # Refresh env keys in case .env was updated
        self.groq_key = os.getenv("GROQ_API_KEY")
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.gemini_key = os.getenv("GEMINI_API_KEY")

        # 1. Try Groq (Fastest TTFT, ~120ms)
        if self.groq_key:
            try:
                res = await self._query_groq(user_query)
                if res:
                    return res
            except Exception as err:
                logger.warning("Groq query failed: %s, falling back", err)

        # 2. Try OpenAI (ChatGPT gpt-4o-mini)
        if self.openai_key:
            try:
                res = await self._query_openai(user_query)
                if res:
                    return res
            except Exception as err:
                logger.warning("OpenAI query failed: %s, falling back", err)

        # 3. Try Local Ollama (Zero cost, offline)
        try:
            res = await self._query_ollama(user_query)
            if res:
                return res
        except Exception:
            pass

        return None

    async def _query_groq(self, prompt: str) -> Optional[str]:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.groq_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "llama-3.1-8b-instant",
            "messages": [
                {"role": "system", "content": JARVIS_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 150,
            "temperature": 0.7,
        }
        async with httpx.AsyncClient(timeout=4.0) as client:
            resp = await client.post(url, json=payload, headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                return data["choices"][0]["message"]["content"].strip()
        return None

    async def _query_openai(self, prompt: str) -> Optional[str]:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.openai_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": JARVIS_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 150,
            "temperature": 0.7,
        }
        async with httpx.AsyncClient(timeout=4.0) as client:
            resp = await client.post(url, json=payload, headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                return data["choices"][0]["message"]["content"].strip()
        return None

    async def _query_ollama(self, prompt: str) -> Optional[str]:
        url = f"{self.ollama_url}/api/generate"
        payload = {
            "model": "llama3.2",
            "prompt": f"{JARVIS_SYSTEM_PROMPT}\n\nUser: {prompt}\nJ.A.R.V.I.S.:",
            "stream": False,
            "options": {"num_predict": 100},
        }
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.post(url, json=payload)
            if resp.status_code == 200:
                data = resp.json()
                return data.get("response", "").strip()
        return None
