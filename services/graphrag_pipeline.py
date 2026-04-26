"""
APEX Self-Improving GraphRAG Pipeline — Pipeline 2: The star of the show.
Graph-augmented RAG with CRAG grading and feedback-driven self-improvement.
"""

import time
import uuid
from typing import Optional
from loguru import logger

from app.models import (
    PipelineResult,
    PipelineMetrics,
    CRAGLabel,
    Chunk,
)
from llm.llm_layer import get_llm_layer
from llm.prompt_manager import get_prompt_manager
from graph.tigergraph_layer import get_tigergraph_layer
from orchestration.crag_agent import get_crag_agent
from orchestration.query_router import get_query_router
from orchestration.entity_extractor import get_entity_extractor
from orchestration.query_decomposer import get_query_decomposer
from services.semantic_cache import get_semantic_cache
from services.feedback_loop import get_feedback_loop
from security.guards import input_guard, output_guard


class GraphRAGPipeline:
    """
    Pipeline 2: Self-Improving GraphRAG.
    
    Flow:
    1. Input Guard (injection detection)
    2. Semantic Cache Check
    3. Query Router (classify → strategy selection)
    4. TigerGraph Multi-hop Retrieval
    5. CRAG Grading
       - CORRECT → generate answer
       - AMBIGUOUS → decompose → re-retrieve → re-grade → generate
       - INCORRECT → refuse gracefully
    6. LLM Generation with structured graph context
    7. Output Guard (PII redaction)
    8. ⭐ FEEDBACK LOOP (the self-improvement core)
    
    Every query strengthens the system for the next one.
    """

    async def run(self, query: str) -> PipelineResult:
        """Execute the full self-improving GraphRAG pipeline."""
        start = time.perf_counter()
        query_id = str(uuid.uuid4())

        # --- Step 1: Input Guard ---
        if not input_guard(query):
            return PipelineResult.blocked("Input rejected by safety filter")

        # --- Step 2: Semantic Cache Check ---
        cache = get_semantic_cache()
        cached = await cache.get(query)
        if cached:
            elapsed = time.perf_counter() - start
            logger.info(f"⚡ Cache HIT! 0 tokens, {elapsed*1000:.0f}ms")
            return PipelineResult.from_cache(cached, elapsed)

        # --- Step 3: Query Routing ---
        router = get_query_router()
        route = await router.classify(query)

        # --- Step 4: TigerGraph Multi-hop Retrieval ---
        tg = get_tigergraph_layer()
        graph_context = await tg.multi_hop_retrieve(
            entities=route.entities,
            hops=route.recommended_hops,
            include_vectors=True,
        )

        # --- Step 5: CRAG Grading ---
        crag = get_crag_agent()
        grade = await crag.grade(query, graph_context.chunks)

        # Handle AMBIGUOUS: decompose and retry
        if grade.label == CRAGLabel.AMBIGUOUS:
            logger.info("🔄 CRAG AMBIGUOUS — decomposing query and re-retrieving...")
            decomposer = get_query_decomposer()
            sub_queries = await decomposer.decompose(query)

            extractor = get_entity_extractor()
            additional_chunks = []

            for sq in sub_queries:
                sq_entities = await extractor.extract(sq.sub_query)
                sq_context = await tg.multi_hop_retrieve(
                    entities=sq_entities, hops=2
                )
                additional_chunks.extend(sq_context.chunks)

            # Merge and re-grade
            all_chunks = graph_context.chunks + additional_chunks
            all_chunks = self._deduplicate_chunks(all_chunks)
            graph_context.chunks = all_chunks

            grade = await crag.grade(query, all_chunks)

        # Handle INCORRECT: refuse gracefully
        if grade.label == CRAGLabel.INCORRECT:
            elapsed = time.perf_counter() - start
            result = PipelineResult.refused(query, grade.reason)
            result.metrics.response_time_ms = elapsed * 1000
            return result

        # --- Step 6: LLM Generation with Graph Context ---
        llm = get_llm_layer()
        pm = get_prompt_manager()

        # Format graph context for the prompt
        context_text = "\n\n".join([c.text for c in graph_context.chunks[:7]])
        relationships_text = self._format_relationships(graph_context.relationships)

        prompt = pm.get(
            "graphrag_qa",
            context=context_text if context_text else "No specific passages retrieved.",
            relationships=relationships_text if relationships_text else "No structured relationships found.",
            question=query,
        )

        response = await llm.generate(prompt)

        # --- Step 7: Output Guard ---
        safe_answer = output_guard(response.text)

        elapsed_ms = (time.perf_counter() - start) * 1000

        result = PipelineResult(
            pipeline="graphrag",
            answer=safe_answer,
            metrics=PipelineMetrics(
                tokens_used=response.usage.total_tokens,
                response_time_ms=elapsed_ms,
                cost_usd=response.cost_usd,
                crag_grade=grade.score,
                retrieval_method=f"graph_{route.strategy.value}",
                entities_resolved=graph_context.entities_resolved,
                relationships_traversed=len(graph_context.relationships),
                hops_used=graph_context.hops_used,
            ),
            metadata={
                "route_strategy": route.strategy.value,
                "crag_label": grade.label.value,
                "crag_reason": grade.reason[:200],
            },
        )

        # --- Step 8: ⭐ FEEDBACK LOOP — Self-Improvement Core ---
        try:
            feedback = get_feedback_loop()
            await feedback.process(
                query_id=query_id,
                query=query,
                context=graph_context.chunks,
                answer=safe_answer,
                crag_grade=grade,
                entities=route.entities,
                graph_paths=graph_context.traversal_paths,
                route_strategy=route.strategy.value,
                response_time=time.perf_counter() - start,
                tokens_used=response.usage.total_tokens,
            )
            result.graph_updates_applied = True
            result.cache_entry_created = grade.score >= 0.75
        except Exception as e:
            logger.warning(f"Feedback loop error (non-critical): {e}")

        logger.info(
            f"✅ GraphRAG: {response.usage.total_tokens} tokens, "
            f"${response.cost_usd:.6f}, {elapsed_ms:.0f}ms, "
            f"CRAG: {grade.label.value} ({grade.score:.3f})"
        )

        return result

    def _deduplicate_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """Remove duplicate chunks by ID."""
        seen = set()
        unique = []
        for c in chunks:
            if c.chunk_id not in seen:
                unique.append(c)
                seen.add(c.chunk_id)
        return unique

    def _format_relationships(self, relationships) -> str:
        """Format graph relationships as readable text for the prompt."""
        if not relationships:
            return ""

        lines = []
        for r in relationships[:10]:  # Cap at 10 relationships
            lines.append(
                f"• {r.source} —[{r.relation_type}]→ {r.target} "
                f"(weight: {r.weight:.2f}, confidence: {r.crag_confidence:.2f})"
            )
        return "\n".join(lines)


# Singleton
_graphrag: Optional[GraphRAGPipeline] = None


def get_graphrag_pipeline() -> GraphRAGPipeline:
    """Get or create the singleton GraphRAG pipeline."""
    global _graphrag
    if _graphrag is None:
        _graphrag = GraphRAGPipeline()
    return _graphrag
