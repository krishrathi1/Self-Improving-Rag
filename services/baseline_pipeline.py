"""
APEX Baseline Pipeline — Pipeline 1: Pure LLM (control group).
Simple vector-based RAG without graph augmentation.
This is the comparison baseline for proving GraphRAG's advantages.
"""

import time
from typing import Optional
from loguru import logger

from app.models import PipelineResult, PipelineMetrics
from llm.llm_layer import get_llm_layer
from llm.prompt_manager import get_prompt_manager


class BaselinePipeline:
    """
    Pipeline 1: Baseline LLM-only RAG.
    
    Flow: Query → (optional vector retrieval) → LLM → Answer
    
    No graph, no multi-hop, no CRAG, no self-improvement.
    This is deliberately simple — it's the control group.
    """

    async def run(self, query: str) -> PipelineResult:
        """Execute the baseline pipeline on a single query."""
        start = time.perf_counter()

        llm = get_llm_layer()
        pm = get_prompt_manager()

        # For a fair comparison, we use a basic context prompt
        # In a full implementation, this would do vector-only retrieval
        # For now, we send the query directly to the LLM with a simple prompt
        context = await self._simple_retrieve(query)

        prompt = pm.get(
            "baseline_qa",
            context=context if context else "No specific context available. Answer from your training data.",
            question=query,
        )

        try:
            response = await llm.generate(prompt, max_tokens=640)

            elapsed_ms = (time.perf_counter() - start) * 1000

            result = PipelineResult(
                pipeline="baseline",
                answer=response.text,
                metrics=PipelineMetrics(
                    tokens_used=response.usage.total_tokens,
                    response_time_ms=elapsed_ms,
                    cost_usd=response.cost_usd,
                    retrieval_method="llm_only",
                ),
            )

            logger.info(
                f"✅ Baseline: {response.usage.total_tokens} tokens, "
                f"${response.cost_usd:.6f}, {elapsed_ms:.0f}ms"
            )

            return result

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.error(f"Baseline pipeline error: {e}")
            return PipelineResult(
                pipeline="baseline",
                answer=f"Error in baseline pipeline: {str(e)}",
                metrics=PipelineMetrics(response_time_ms=elapsed_ms),
            )

    async def _simple_retrieve(self, query: str) -> str:
        """
        Simple vector-based retrieval for the baseline.
        Uses local VectorStore for Dense retrieval (and some BM25) instead of TigerGraph
        to represent a true standard baseline RAG.
        """
        try:
            from services.vector_store import get_vector_store
            vs = get_vector_store()
            
            # Simple Dense Search (Standard RAG)
            chunks = await vs.similarity_search(query, k=5)
            
            if not chunks:
                # Fallback to BM25 if Vector store doesn't match
                chunks = await vs.keyword_search(query, k=5)
                
            if chunks:
                texts = [c.text for c in chunks if c.text]
                return "\n\n".join(texts[:5])

        except Exception as e:
            logger.debug(f"Baseline retrieval skipped/failed: {e}")

        return ""


# Singleton
_baseline: Optional[BaselinePipeline] = None


def get_baseline_pipeline() -> BaselinePipeline:
    """Get or create the singleton baseline pipeline."""
    global _baseline
    if _baseline is None:
        _baseline = BaselinePipeline()
    return _baseline
