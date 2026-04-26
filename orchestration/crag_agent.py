"""
APEX CRAG Agent — Corrective Retrieval Augmented Generation.
Grades retrieval quality and triggers correction when results are ambiguous.
The CRAG grades are the primary signal driving graph self-improvement.
"""

import json
from typing import Optional
from loguru import logger

from app.models import Chunk, CRAGGrade, CRAGLabel
from llm.llm_layer import get_llm_layer
from llm.prompt_manager import get_prompt_manager


class CRAGAgent:
    """
    Corrective RAG agent that:
    1. Grades the quality of retrieved context against the query
    2. Triggers query decomposition + re-retrieval for ambiguous results
    3. Refuses gracefully when retrieval quality is too low
    
    The CRAG grade (0.0-1.0) is THE signal that drives self-improvement:
    - Feeds back into graph edge weights
    - Determines cache entry eligibility
    - Tracks prompt template performance
    """

    # Grade thresholds
    CORRECT_THRESHOLD = 0.7
    AMBIGUOUS_THRESHOLD = 0.3

    async def grade(self, query: str, chunks: list[Chunk]) -> CRAGGrade:
        """
        Grade the relevance of retrieved chunks to the query.
        Uses LLM-as-Judge with structured JSON output.
        
        Returns:
            CRAGGrade with score (0.0-1.0), label, and reasoning
        """
        if not chunks:
            return CRAGGrade(
                score=0.0,
                label=CRAGLabel.INCORRECT,
                reason="No chunks retrieved — retrieval returned empty results",
            )

        llm = get_llm_layer()
        pm = get_prompt_manager()

        # Take top 5 chunks for grading (avoid token waste)
        chunk_texts = "\n---\n".join([c.text for c in chunks[:5]])

        prompt = pm.get("crag_grading", query=query, chunks=chunk_texts)

        try:
            response = await llm.generate(prompt, temperature=0.0, max_tokens=256)
            parsed = self._parse_grade(response.text)

            score = parsed["score"]
            reason = parsed.get("reason", "")

            if score >= self.CORRECT_THRESHOLD:
                label = CRAGLabel.CORRECT
            elif score >= self.AMBIGUOUS_THRESHOLD:
                label = CRAGLabel.AMBIGUOUS
            else:
                label = CRAGLabel.INCORRECT

            grade = CRAGGrade(score=score, label=label, reason=reason)

            logger.info(
                f"📊 CRAG Grade: {grade.label.value} ({score:.3f}) — {reason[:80]}"
            )
            return grade

        except Exception as e:
            logger.error(f"CRAG grading failed: {e}")
            # Default to AMBIGUOUS on error — safer than assuming correctness
            return CRAGGrade(
                score=0.5,
                label=CRAGLabel.AMBIGUOUS,
                reason=f"Grading error: {str(e)}",
            )

    def _parse_grade(self, response_text: str) -> dict:
        """Parse the JSON response from the grading LLM."""
        text = response_text.strip()

        # Remove markdown code fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1]) if len(lines) > 2 else text

        try:
            parsed = json.loads(text)
            score = float(parsed.get("score", 0.5))
            # Clamp to [0, 1]
            score = max(0.0, min(1.0, score))
            return {"score": score, "reason": parsed.get("reason", "")}
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(f"Could not parse CRAG grade JSON: {e}, raw: {text[:200]}")
            # Try to extract a float from the response
            import re
            match = re.search(r"(\d+\.?\d*)", text)
            if match:
                score = float(match.group(1))
                if score > 1.0:
                    score = score / 10.0  # Handle 0-10 scale
                return {"score": max(0.0, min(1.0, score)), "reason": text[:100]}
            return {"score": 0.5, "reason": "Could not parse grade"}


# Singleton
_crag_agent: Optional[CRAGAgent] = None


def get_crag_agent() -> CRAGAgent:
    """Get or create the singleton CRAG agent."""
    global _crag_agent
    if _crag_agent is None:
        _crag_agent = CRAGAgent()
    return _crag_agent
