"""
APEX Query Decomposer — Breaks complex queries into simpler sub-questions.
Used by CRAG when retrieval is AMBIGUOUS — sub-queries get better individual retrieval.
"""

import json
from typing import Optional
from loguru import logger

from llm.llm_layer import get_llm_layer
from llm.prompt_manager import get_prompt_manager


class SubQuery:
    """A decomposed sub-question with its purpose."""

    def __init__(self, sub_query: str, purpose: str = ""):
        self.sub_query = sub_query
        self.purpose = purpose

    def __repr__(self):
        return f"SubQuery('{self.sub_query[:50]}...')"


class QueryDecomposer:
    """
    Decomposes complex multi-part questions into simpler sub-queries.
    
    This is triggered when CRAG grades a retrieval as AMBIGUOUS (0.3-0.7).
    The sub-queries are individually retrieved and the results merged
    back for a more complete answer.
    """

    async def decompose(self, query: str) -> list[SubQuery]:
        """
        Break a complex query into 2-4 simpler sub-questions.
        Uses LLM to understand the question structure.
        """
        try:
            return await self._llm_decompose(query)
        except Exception as e:
            logger.warning(f"LLM decomposition failed, using fallback: {e}")
            return self._fallback_decompose(query)

    async def _llm_decompose(self, query: str) -> list[SubQuery]:
        """Decompose using LLM with structured JSON output."""
        llm = get_llm_layer()
        pm = get_prompt_manager()

        prompt = pm.get("query_decomposition", query=query)
        response = await llm.generate(prompt, temperature=0.1, max_tokens=512)

        raw = response.text.strip()

        # Remove markdown fences
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(lines[1:-1]) if len(lines) > 2 else raw

        parsed = json.loads(raw)

        sub_queries = []
        for item in parsed:
            if isinstance(item, dict) and "sub_query" in item:
                sub_queries.append(
                    SubQuery(
                        sub_query=item["sub_query"],
                        purpose=item.get("purpose", ""),
                    )
                )

        logger.info(
            f"🔀 Decomposed query into {len(sub_queries)} sub-queries: "
            f"{[sq.sub_query[:40] + '...' for sq in sub_queries]}"
        )
        return sub_queries

    def _fallback_decompose(self, query: str) -> list[SubQuery]:
        """
        Simple fallback: split on conjunctions and question marks.
        Not as smart as LLM decomposition, but ensures we always have sub-queries.
        """
        import re

        # Split on "and", "but", "also", question marks
        parts = re.split(r'\band\b|\bbut\b|\balso\b|\?', query, flags=re.IGNORECASE)
        parts = [p.strip() for p in parts if p.strip() and len(p.strip()) > 10]

        if len(parts) <= 1:
            # If can't split, create a more specific version
            return [
                SubQuery(sub_query=query, purpose="original query"),
                SubQuery(sub_query=f"Define and explain: {query}", purpose="definition"),
            ]

        sub_queries = [SubQuery(sub_query=p + "?", purpose=f"part {i+1}") for i, p in enumerate(parts)]
        return sub_queries[:4]  # Cap at 4


# Singleton
_decomposer: Optional[QueryDecomposer] = None


def get_query_decomposer() -> QueryDecomposer:
    """Get or create the singleton query decomposer."""
    global _decomposer
    if _decomposer is None:
        _decomposer = QueryDecomposer()
    return _decomposer
