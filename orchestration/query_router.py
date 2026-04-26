"""
APEX Query Router — Graph-aware query classification and routing.
Decides which retrieval strategy to use based on entity coverage in the graph.
"""

from typing import Optional
from loguru import logger

from app.models import Entity, RouteDecision, RouteStrategy
from graph.tigergraph_layer import get_tigergraph_layer
from orchestration.entity_extractor import get_entity_extractor
from orchestration.observability import trace_stage


class QueryRouter:
    """
    Graph-aware query router that decides HOW to handle a query:
    
    1. graph_first — Entities exist in graph with high confidence → multi-hop
    2. hybrid_balanced — Partial graph coverage → 1-hop graph + vector search
    3. vector_fallback — No graph coverage → pure vector search
    
    The router also learns from past query patterns:
    if a certain strategy consistently yields higher CRAG grades
    for similar queries, it adapts.
    """

    def __init__(self):
        # Pattern memory: maps query pattern → best strategy
        self._pattern_memory: dict[str, dict] = {}
        logger.info("🧭 Query Router initialized")

    @trace_stage("query_routing")
    async def classify(self, query: str) -> RouteDecision:
        """
        Classify a query and decide the retrieval strategy.
        
        Steps:
        1. Extract candidate entities
        2. Check graph coverage
        3. Check pattern memory for similar past queries
        4. Select strategy
        """
        # 1. Extract entities
        extractor = get_entity_extractor()
        entities = await extractor.extract(query)

        # 2. Check graph coverage
        tg = get_tigergraph_layer()
        coverage = await tg.check_entity_coverage(entities)

        ratio = coverage.get("ratio", 0.0)
        avg_confidence = coverage.get("avg_confidence", 0.0)

        # 3. Check pattern memory
        similar_pattern = self._find_similar_pattern(query)
        if similar_pattern and similar_pattern.get("uses", 0) >= 3:
            # Use learned strategy if we have enough data
            strategy = RouteStrategy(similar_pattern["best_strategy"])
            hops = similar_pattern.get("optimal_hops", 2)
            reasoning = (
                f"Pattern match: similar queries perform best with "
                f"{strategy.value} (avg_crag: {similar_pattern.get('avg_crag', 0):.2f})"
            )
        elif ratio > 0.7 and avg_confidence > 0.5:
            strategy = RouteStrategy.GRAPH_FIRST
            hops = self._adaptive_hop_depth(entities, coverage)
            reasoning = (
                f"High graph coverage ({ratio:.0%}) with good confidence "
                f"({avg_confidence:.2f}) → graph-first with {hops} hops"
            )
        elif ratio > 0.3:
            strategy = RouteStrategy.HYBRID_BALANCED
            hops = 1
            reasoning = (
                f"Partial graph coverage ({ratio:.0%}) → hybrid with 1-hop graph + vector"
            )
        else:
            strategy = RouteStrategy.VECTOR_FALLBACK
            hops = 0
            reasoning = (
                f"Low graph coverage ({ratio:.0%}) → vector-only fallback"
            )

        decision = RouteDecision(
            strategy=strategy,
            recommended_hops=hops,
            entities=entities,
            coverage_ratio=ratio,
            reasoning=reasoning,
        )

        logger.info(f"🧭 Route: {strategy.value} ({reasoning})")
        return decision

    def record_outcome(
        self, query: str, strategy: str, crag_grade: float, response_time: float
    ):
        """
        Record the outcome of a query to learn which strategies work best.
        Part of the self-improvement query pattern learning.
        """
        # Simple fingerprint: lowercase, sorted words
        fingerprint = self._query_fingerprint(query)

        if fingerprint not in self._pattern_memory:
            self._pattern_memory[fingerprint] = {
                "strategies": {},
                "uses": 0,
                "best_strategy": strategy,
                "avg_crag": crag_grade,
            }

        pattern = self._pattern_memory[fingerprint]
        pattern["uses"] += 1

        if strategy not in pattern["strategies"]:
            pattern["strategies"][strategy] = {
                "total_crag": 0.0,
                "total_time": 0.0,
                "count": 0,
            }

        s = pattern["strategies"][strategy]
        s["total_crag"] += crag_grade
        s["total_time"] += response_time
        s["count"] += 1
        avg_crag = s["total_crag"] / s["count"]

        # Update best strategy
        best_avg = 0.0
        for strat_name, strat_data in pattern["strategies"].items():
            if strat_data["count"] > 0:
                strat_avg = strat_data["total_crag"] / strat_data["count"]
                if strat_avg > best_avg:
                    best_avg = strat_avg
                    pattern["best_strategy"] = strat_name

        pattern["avg_crag"] = best_avg

    def _adaptive_hop_depth(self, entities: list[Entity], coverage: dict) -> int:
        """
        Determine optimal hop depth based on entity types and coverage.
        
        Heuristics:
        - Medical entities → 2-3 hops (drug → condition → side_effect)
        - Simple lookups → 1 hop
        - Complex reasoning → 3 hops
        """
        entity_types = [e.entity_type for e in entities]

        if any(t in ("DRUG", "MEDICAL_TERM", "CONDITION") for t in entity_types):
            return 3  # Medical queries benefit from deeper traversal
        elif coverage.get("avg_confidence", 0) > 0.8:
            return 2  # High confidence → moderate depth is enough
        else:
            return 2  # Default

    def _query_fingerprint(self, query: str) -> str:
        """Create a simple fingerprint for query pattern matching."""
        words = sorted(set(query.lower().split()))
        # Take the 5 most significant words (skip common ones)
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "what", "how", "why",
                     "when", "where", "which", "who", "do", "does", "did", "can", "could",
                     "would", "should", "of", "in", "on", "at", "to", "for", "with"}
        significant = [w for w in words if w not in stopwords and len(w) > 2]
        return " ".join(significant[:5])

    def _find_similar_pattern(self, query: str) -> Optional[dict]:
        """Find a similar past query pattern in memory."""
        fingerprint = self._query_fingerprint(query)

        # Exact match
        if fingerprint in self._pattern_memory:
            return self._pattern_memory[fingerprint]

        # Simple subset match
        fp_words = set(fingerprint.split())
        best_match = None
        best_overlap = 0

        for pattern_fp, pattern_data in self._pattern_memory.items():
            pattern_words = set(pattern_fp.split())
            overlap = len(fp_words & pattern_words)
            if overlap >= 3 and overlap > best_overlap:  # At least 3 words in common
                best_overlap = overlap
                best_match = pattern_data

        return best_match

    @property
    def patterns_learned(self) -> int:
        """Number of query patterns learned."""
        return len(self._pattern_memory)


# Singleton
_query_router: Optional[QueryRouter] = None


def get_query_router() -> QueryRouter:
    """Get or create the singleton query router."""
    global _query_router
    if _query_router is None:
        _query_router = QueryRouter()
    return _query_router
