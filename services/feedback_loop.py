"""
APEX Feedback Loop — The heart of self-improvement.
Processes every GraphRAG query result and updates the system across 5 channels.
This is what makes APEX SELF-IMPROVING, not just a static GraphRAG comparison.
"""

import uuid
from typing import Optional
from loguru import logger

from app.models import CRAGGrade, Entity, Chunk, GraphPath


class FeedbackLoop:
    """
    The self-improvement engine. After every query, it:
    
    Channel 1: Updates graph edge weights based on CRAG grade
    Channel 2: Caches high-quality responses in semantic cache
    Channel 3: Tracks prompt template performance
    Channel 4: Records query patterns for routing optimization
    Channel 5: Discovers new entities from high-quality answers
    
    The net effect: every query makes the NEXT query faster, cheaper, and more accurate.
    """

    def __init__(self):
        self._total_feedback_events = 0
        self._total_edges_updated = 0
        self._total_entities_discovered = 0
        logger.info("🔄 Feedback Loop initialized")

    async def process(
        self,
        query_id: str,
        query: str,
        context: list[Chunk],
        answer: str,
        crag_grade: CRAGGrade,
        entities: list[Entity],
        graph_paths: list[GraphPath],
        route_strategy: str,
        response_time: float,
        tokens_used: int,
    ):
        """
        Process feedback from a single query execution.
        Runs all 5 channels sequentially (each is fast and non-blocking).
        """
        self._total_feedback_events += 1

        logger.info(
            f"🔄 Feedback processing: CRAG={crag_grade.score:.3f}, "
            f"entities={len(entities)}, edges={sum(len(p.edges) for p in graph_paths)}"
        )

        # --- Channel 1: Graph Edge Weight Update ---
        await self._channel_graph_edges(query_id, graph_paths, crag_grade.score)

        # --- Channel 2: Semantic Cache ---
        await self._channel_cache(query, answer, crag_grade.score)

        # --- Channel 3: Prompt Performance ---
        await self._channel_prompt_tracking(crag_grade.score)

        # --- Channel 4: Query Pattern Learning ---
        await self._channel_pattern_learning(
            query, route_strategy, crag_grade.score, response_time
        )

        # --- Channel 5: Entity Discovery ---
        if crag_grade.score >= 0.7:
            await self._channel_entity_discovery(answer, entities)

        # --- Log metrics ---
        await self._log_metrics(
            query_id, query, crag_grade.score, tokens_used,
            response_time, route_strategy
        )

        logger.success(
            f"✅ Feedback processed: {self._total_feedback_events} total events"
        )

    async def _channel_graph_edges(
        self, query_id: str, graph_paths: list[GraphPath], crag_score: float
    ):
        """Channel 1: Update edge weights based on CRAG grade."""
        edge_ids = []
        for path in graph_paths:
            for edge in path.edges:
                # Use source_target as edge ID
                edge_id = f"{edge.source}_{edge.target}_{edge.relation_type}"
                edge_ids.append(edge_id)

        if edge_ids:
            try:
                from graph.tigergraph_layer import get_tigergraph_layer
                tg = get_tigergraph_layer()
                await tg.apply_feedback(query_id, edge_ids, crag_score)
                self._total_edges_updated += len(edge_ids)
                logger.debug(f"📊 Channel 1: {len(edge_ids)} edge weights updated")
            except Exception as e:
                logger.debug(f"Channel 1 (edge update) skipped: {e}")

    async def _channel_cache(self, query: str, answer: str, crag_score: float):
        """Channel 2: Cache high-quality responses."""
        try:
            from services.semantic_cache import get_semantic_cache
            cache = get_semantic_cache()
            await cache.put(query=query, answer=answer, crag_grade=crag_score)
            logger.debug(f"💾 Channel 2: Cache processed (grade={crag_score:.3f})")
        except Exception as e:
            logger.debug(f"Channel 2 (cache) skipped: {e}")

    async def _channel_prompt_tracking(self, crag_score: float):
        """Channel 3: Track prompt template performance."""
        try:
            from llm.prompt_manager import get_prompt_manager
            pm = get_prompt_manager()
            active_version = pm.active_version("graphrag_qa")
            await pm.track_and_refine("graphrag_qa", active_version, crag_score)
            logger.debug(f"📝 Channel 3: Prompt '{active_version}' tracked")
        except Exception as e:
            logger.debug(f"Channel 3 (prompt tracking) skipped: {e}")

    async def _channel_pattern_learning(
        self, query: str, strategy: str, crag_score: float, response_time: float
    ):
        """Channel 4: Record query pattern for routing optimization."""
        try:
            from orchestration.query_router import get_query_router
            router = get_query_router()
            router.record_outcome(query, strategy, crag_score, response_time)
            logger.debug(
                f"🧭 Channel 4: Pattern recorded (strategy={strategy}, "
                f"grade={crag_score:.3f})"
            )
        except Exception as e:
            logger.debug(f"Channel 4 (pattern learning) skipped: {e}")

    async def _channel_entity_discovery(self, answer: str, known_entities: list[Entity]):
        """
        Channel 5: Discover new entities from high-quality answers.
        The graph GROWS with every successful query.
        """
        try:
            from orchestration.entity_extractor import get_entity_extractor
            from graph.tigergraph_layer import get_tigergraph_layer

            extractor = get_entity_extractor()
            new_entities = await extractor.extract(answer)

            # Filter out already-known entities
            known_names = set(e.name.lower() for e in known_entities)
            discovered = [
                e for e in new_entities
                if e.name.lower() not in known_names and len(e.name) > 2
            ]

            if discovered:
                tg = get_tigergraph_layer()
                await tg.upsert_entities(discovered)

                # Create relationships between known and discovered entities
                if known_entities:
                    await tg.create_relationships(
                        source_entities=known_entities[:3],
                        target_entities=discovered[:5],
                        relation_type="DISCOVERED_VIA_QUERY",
                        initial_weight=0.5,
                    )

                self._total_entities_discovered += len(discovered)
                logger.info(
                    f"🌱 Channel 5: Discovered {len(discovered)} new entities: "
                    f"{[e.name for e in discovered[:5]]}"
                )

        except Exception as e:
            logger.debug(f"Channel 5 (entity discovery) skipped: {e}")

    async def _log_metrics(
        self,
        query_id: str,
        query: str,
        crag_score: float,
        tokens_used: int,
        response_time: float,
        strategy: str,
    ):
        """Log metrics to the evaluation store for dashboard tracking."""
        try:
            from evaluation.metrics_store import get_metrics_store
            store = get_metrics_store()
            await store.log_graphrag_query(
                query_id=query_id,
                query=query,
                crag_score=crag_score,
                tokens_used=tokens_used,
                response_time_ms=response_time * 1000,
                strategy=strategy,
                cache_hit=False,
                edges_updated=self._total_edges_updated,
                entities_discovered=self._total_entities_discovered,
            )
        except Exception as e:
            logger.debug(f"Metrics logging skipped: {e}")

    @property
    def stats(self) -> dict:
        """Get feedback loop statistics for the dashboard."""
        return {
            "total_feedback_events": self._total_feedback_events,
            "total_edges_updated": self._total_edges_updated,
            "total_entities_discovered": self._total_entities_discovered,
        }


# Singleton
_feedback_loop: Optional[FeedbackLoop] = None


def get_feedback_loop() -> FeedbackLoop:
    """Get or create the singleton feedback loop."""
    global _feedback_loop
    if _feedback_loop is None:
        _feedback_loop = FeedbackLoop()
    return _feedback_loop
