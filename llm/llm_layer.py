"""
APEX LLM Layer — Unified interface for multiple LLM providers.
Tracks token usage and cost per call for benchmarking.
Layer 3 of the AI Factory model.
"""

import time
from functools import lru_cache
from typing import Optional

from loguru import logger

from app.config import get_settings
from app.models import LLMResponse, TokenUsage


# Per-million-token pricing (input, output) in USD
PRICING = {
    "llama3.2:latest": {"input": 0.0, "output": 0.0},
    "gpt-4o-mini": {"input": 0.15 / 1_000_000, "output": 0.60 / 1_000_000},
    "gpt-4o": {"input": 2.50 / 1_000_000, "output": 10.00 / 1_000_000},
    "gpt-3.5-turbo": {"input": 0.50 / 1_000_000, "output": 1.50 / 1_000_000},
    "gemini-2.0-flash": {"input": 0.10 / 1_000_000, "output": 0.40 / 1_000_000},
    "gemini-1.5-flash": {"input": 0.075 / 1_000_000, "output": 0.30 / 1_000_000},
    "llama-3.3-70b-versatile": {"input": 0.59 / 1_000_000, "output": 0.79 / 1_000_000},
    "llama-3.1-8b-instant": {"input": 0.05 / 1_000_000, "output": 0.08 / 1_000_000},
}

# Default fallback pricing for unknown models
DEFAULT_PRICING = {"input": 1.00 / 1_000_000, "output": 2.00 / 1_000_000}


class LLMLayer:
    """
    Unified LLM interface supporting OpenAI, Google, and Groq providers.
    Every call automatically tracks token usage and cost for benchmarking.
    """

    def __init__(self):
        settings = get_settings()
        self.provider = settings.llm.provider
        self.model = settings.llm.model
        self.temperature = settings.llm.temperature
        self.max_tokens = settings.llm.max_tokens
        self._client = None

        logger.info(f"🤖 LLM Layer initialized: {self.provider}/{self.model}")

    @property
    def client(self):
        """Lazy-initialize the LLM client."""
        if self._client is None:
            settings = get_settings()

            if self.provider == "openai":
                from openai import AsyncOpenAI
                self._client = AsyncOpenAI(api_key=settings.llm.openai_api_key)

            elif self.provider == "groq":
                from groq import AsyncGroq
                self._client = AsyncGroq(api_key=settings.llm.groq_api_key)

            elif self.provider == "google":
                import google.generativeai as genai
                genai.configure(api_key=settings.llm.google_api_key)
                self._client = genai.GenerativeModel(self.model)

            elif self.provider == "ollama":
                import httpx
                self._client = httpx.AsyncClient(
                    base_url=settings.llm.ollama_host,
                    timeout=httpx.Timeout(120.0, connect=10.0),
                )

            else:
                raise ValueError(f"Unsupported LLM provider: {self.provider}")

        return self._client

    async def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
        routing_strategy: str = "default"
    ) -> LLMResponse:
        """
        Generate a response from the LLM.
        V2 UPGRADE: Supports multi-model intelligent routing based on task complexity.
        Returns an LLMResponse with text, token usage, cost, and latency.
        """
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens

        # Smart Routing Logic (Cost/Complexity Optimization)
        target_model = self.model
        if routing_strategy == "fast":
            target_model = "gemini-1.5-flash" if self.provider == "google" else "llama-3.1-8b-instant"
        elif routing_strategy == "complex":
            target_model = "gpt-4o" if self.provider == "openai" else "llama-3.3-70b-versatile"
        
        # Override temporary self.model for this execution
        original_model = self.model
        self.model = target_model

        start = time.perf_counter()

        try:
            if self.provider in ("openai", "groq"):
                response = await self._generate_openai_compatible(
                    prompt, temp, max_tok, system_prompt
                )
            elif self.provider == "google":
                response = await self._generate_google(prompt, temp, max_tok)
            elif self.provider == "ollama":
                response = await self._generate_ollama(
                    prompt, temp, max_tok, system_prompt
                )
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            raise
        finally:
            self.model = original_model # Restore original model

        elapsed_ms = (time.perf_counter() - start) * 1000

        # Calculate cost
        pricing = PRICING.get(self.model, DEFAULT_PRICING)
        cost = (
            response.usage.prompt_tokens * pricing["input"]
            + response.usage.completion_tokens * pricing["output"]
        )

        response.cost_usd = cost
        response.latency_ms = elapsed_ms
        response.model = self.model

        logger.debug(
            f"LLM call: {response.usage.total_tokens} tokens, "
            f"${cost:.6f}, {elapsed_ms:.0f}ms"
        )

        return response

    async def _generate_openai_compatible(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        system_prompt: Optional[str] = None,
    ) -> LLMResponse:
        """Generate using OpenAI or Groq (OpenAI-compatible API)."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        usage = response.usage
        return LLMResponse(
            text=response.choices[0].message.content or "",
            usage=TokenUsage(
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
            ),
        )

    async def _generate_google(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        """Generate using Google's Generative AI (Gemini)."""
        import google.generativeai as genai

        generation_config = genai.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        response = await self.client.generate_content_async(
            prompt,
            generation_config=generation_config,
        )

        # Google doesn't return token counts the same way
        # Estimate from response metadata
        prompt_tokens = getattr(
            response.usage_metadata, "prompt_token_count", len(prompt) // 4
        )
        completion_tokens = getattr(
            response.usage_metadata, "candidates_token_count", len(response.text) // 4
        )

        return LLMResponse(
            text=response.text,
            usage=TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )

    async def _generate_ollama(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        system_prompt: Optional[str] = None,
    ) -> LLMResponse:
        """Generate using a local Ollama model."""
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"System:\n{system_prompt}\n\nUser:\n{prompt}"

        response = await self.client.post(
            "/api/generate",
            json={
                "model": self.model,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            },
        )
        response.raise_for_status()
        data = response.json()
        text = data.get("response", "")

        prompt_tokens = int(data.get("prompt_eval_count") or max(1, len(full_prompt) // 4))
        completion_tokens = int(data.get("eval_count") or max(1, len(text) // 4))

        return LLMResponse(
            text=text,
            usage=TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )

    async def health_check(self) -> dict:
        """Quick health check — test a minimal generation."""
        try:
            start = time.perf_counter()
            response = await self.generate("Say 'ok'.", max_tokens=5, temperature=0)
            latency = (time.perf_counter() - start) * 1000
            return {
                "status": "connected",
                "model": self.model,
                "provider": self.provider,
                "latency_ms": round(latency, 1),
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def get_pricing(self) -> dict:
        """Return pricing for the current model."""
        return PRICING.get(self.model, DEFAULT_PRICING)


# Singleton
_llm_layer: Optional[LLMLayer] = None


def get_llm_layer() -> LLMLayer:
    """Get or create the singleton LLM layer."""
    global _llm_layer
    if _llm_layer is None:
        _llm_layer = LLMLayer()
    return _llm_layer
