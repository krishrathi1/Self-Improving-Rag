"""
APEX Entity Extractor — Extracts named entities from queries and documents.
Uses LLM for extraction with fallback to simple pattern matching.
"""

import json
import re
from typing import Optional
from loguru import logger

from app.models import Entity
from llm.llm_layer import get_llm_layer
from llm.prompt_manager import get_prompt_manager


class EntityExtractor:
    """
    Extracts entities from text using LLM.
    Used in:
    - Query processing: extract entities to use as graph traversal seeds
    - Document ingestion: extract entities for knowledge graph population
    - Self-improvement: extract new entities from high-quality answers
    """

    async def extract(self, text: str) -> list[Entity]:
        """
        Extract entities from text using LLM.
        Falls back to simple regex extraction on failure.
        """
        llm = get_llm_layer()
        if llm.provider == "ollama":
            return self._fallback_extract(text)

        try:
            return await self._llm_extract(text)
        except Exception as e:
            logger.warning(f"LLM entity extraction failed, using fallback: {e}")
            return self._fallback_extract(text)

    async def _llm_extract(self, text: str) -> list[Entity]:
        """Extract entities using LLM with structured JSON output."""
        llm = get_llm_layer()
        pm = get_prompt_manager()

        prompt = pm.get("entity_extraction", text=text)
        response = await llm.generate(prompt, temperature=0.0, max_tokens=1024)

        raw = response.text.strip()

        # Remove markdown fences
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(lines[1:-1]) if len(lines) > 2 else raw

        parsed = json.loads(raw)

        entities = []
        for item in parsed:
            if isinstance(item, dict) and "name" in item:
                entities.append(
                    Entity(
                        name=item["name"].strip(),
                        entity_type=item.get("entity_type", "UNKNOWN"),
                        confidence=item.get("confidence", 0.8),
                    )
                )

        logger.info(f"🔍 Extracted {len(entities)} entities: {[e.name for e in entities[:5]]}")
        return entities

    def _fallback_extract(self, text: str) -> list[Entity]:
        """
        Simple fallback entity extraction using regex.
        Extracts:
        - Capitalized phrases (likely proper nouns)
        - Quoted terms
        - Technical terms after key phrases
        """
        entities = []
        seen = set()

        # Pattern 1: Capitalized multi-word phrases (2-4 words)
        for match in re.finditer(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b', text):
            name = match.group(1).strip()
            if name not in seen and len(name) > 2:
                entities.append(Entity(name=name, entity_type="UNKNOWN", confidence=0.5))
                seen.add(name)

        # Pattern 2: Quoted terms
        for match in re.finditer(r'"([^"]+)"', text):
            name = match.group(1).strip()
            if name not in seen and len(name) > 2:
                entities.append(Entity(name=name, entity_type="CONCEPT", confidence=0.6))
                seen.add(name)

        # Pattern 3: Key noun phrases
        for match in re.finditer(
            r'(?:about|regarding|concerning|of|for)\s+([a-zA-Z\s]{3,30}?)(?:\?|\.|\,|$)',
            text,
            re.IGNORECASE,
        ):
            name = match.group(1).strip()
            if name not in seen and len(name) > 3:
                entities.append(Entity(name=name, entity_type="CONCEPT", confidence=0.4))
                seen.add(name)

        full_query = text[:140].strip()
        if full_query and full_query.lower() not in seen:
            entities.append(Entity(name=full_query, entity_type="QUERY", confidence=0.35))

        return entities[:10]  # Cap at 10 entities


# Singleton
_entity_extractor: Optional[EntityExtractor] = None


def get_entity_extractor() -> EntityExtractor:
    """Get or create the singleton entity extractor."""
    global _entity_extractor
    if _entity_extractor is None:
        _entity_extractor = EntityExtractor()
    return _entity_extractor
