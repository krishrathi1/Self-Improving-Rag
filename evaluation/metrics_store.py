"""
APEX Metrics Store — Persists all comparison and improvement metrics.
Uses SQLite for lightweight storage (upgradeable to PostgreSQL).
Serves the dashboard with aggregated data.
"""

import json
import sqlite3
import os
from datetime import datetime
from typing import Optional
from loguru import logger

from app.models import (
    ComparisonResult,
    MetricsSummary,
    ImprovementBatch,
    PipelineMetrics,
)
from app.config import get_settings


DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "apex_metrics.db")


class MetricsStore:
    """
    SQLite-backed metrics persistence for dashboard data.
    
    Stores:
    - Individual comparison results (baseline vs GraphRAG)
    - GraphRAG query logs with self-improvement metrics
    - Aggregated summaries and improvement curves
    """

    def __init__(self):
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        self._conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_tables()
        logger.info(f"📊 Metrics Store initialized: {DB_PATH}")

    def _init_tables(self):
        """Create tables if they don't exist."""
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS comparisons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                baseline_tokens INTEGER DEFAULT 0,
                baseline_time_ms REAL DEFAULT 0,
                baseline_cost REAL DEFAULT 0,
                baseline_answer TEXT DEFAULT '',
                graphrag_tokens INTEGER DEFAULT 0,
                graphrag_time_ms REAL DEFAULT 0,
                graphrag_cost REAL DEFAULT 0,
                graphrag_answer TEXT DEFAULT '',
                graphrag_crag_grade REAL DEFAULT 0,
                graphrag_cache_hit INTEGER DEFAULT 0,
                token_savings_pct REAL DEFAULT 0,
                speed_improvement_pct REAL DEFAULT 0,
                cost_savings_pct REAL DEFAULT 0,
                accuracy_baseline INTEGER DEFAULT 0,
                accuracy_graphrag INTEGER DEFAULT 0,
                accuracy_winner TEXT DEFAULT 'tie',
                timestamp TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS graphrag_queries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_id TEXT UNIQUE,
                query TEXT NOT NULL,
                crag_score REAL DEFAULT 0,
                tokens_used INTEGER DEFAULT 0,
                response_time_ms REAL DEFAULT 0,
                strategy TEXT DEFAULT '',
                cache_hit INTEGER DEFAULT 0,
                edges_updated INTEGER DEFAULT 0,
                entities_discovered INTEGER DEFAULT 0,
                prompt_version TEXT DEFAULT 'v1',
                timestamp TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS self_improvement_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                details TEXT DEFAULT '',
                timestamp TEXT DEFAULT (datetime('now'))
            );
        """)
        self._conn.commit()

    async def save_comparison(self, comparison: ComparisonResult):
        """Persist a comparison result."""
        try:
            acc_b = 0
            acc_g = 0
            acc_winner = "tie"
            if comparison.accuracy_scores:
                acc_b = comparison.accuracy_scores.answer_a.total
                acc_g = comparison.accuracy_scores.answer_b.total
                acc_winner = comparison.accuracy_scores.winner

            self._conn.execute("""
                INSERT INTO comparisons (
                    query, baseline_tokens, baseline_time_ms, baseline_cost, baseline_answer,
                    graphrag_tokens, graphrag_time_ms, graphrag_cost, graphrag_answer,
                    graphrag_crag_grade, graphrag_cache_hit,
                    token_savings_pct, speed_improvement_pct, cost_savings_pct,
                    accuracy_baseline, accuracy_graphrag, accuracy_winner
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                comparison.query,
                comparison.baseline.metrics.tokens_used,
                comparison.baseline.metrics.response_time_ms,
                comparison.baseline.metrics.cost_usd,
                comparison.baseline.answer[:1000],
                comparison.graphrag.metrics.tokens_used,
                comparison.graphrag.metrics.response_time_ms,
                comparison.graphrag.metrics.cost_usd,
                comparison.graphrag.answer[:1000],
                comparison.graphrag.metrics.crag_grade or 0,
                1 if comparison.graphrag.metrics.cache_hit else 0,
                comparison.token_savings_pct,
                comparison.speed_improvement_pct,
                comparison.cost_savings_pct,
                acc_b, acc_g, acc_winner,
            ))
            self._conn.commit()
        except Exception as e:
            logger.error(f"Failed to save comparison: {e}")

    async def log_graphrag_query(
        self,
        query_id: str,
        query: str,
        crag_score: float,
        tokens_used: int,
        response_time_ms: float,
        strategy: str = "",
        cache_hit: bool = False,
        edges_updated: int = 0,
        entities_discovered: int = 0,
    ):
        """Log a GraphRAG query with its metrics."""
        try:
            self._conn.execute("""
                INSERT OR REPLACE INTO graphrag_queries (
                    query_id, query, crag_score, tokens_used, response_time_ms,
                    strategy, cache_hit, edges_updated, entities_discovered
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                query_id, query, crag_score, tokens_used, response_time_ms,
                strategy, 1 if cache_hit else 0, edges_updated, entities_discovered,
            ))
            self._conn.commit()
        except Exception as e:
            logger.error(f"Failed to log query: {e}")

    async def get_summary(self) -> MetricsSummary:
        """Get aggregated summary for the dashboard."""
        try:
            # Comparison averages
            row = self._conn.execute("""
                SELECT 
                    COUNT(*) as total,
                    AVG(baseline_tokens) as avg_b_tokens,
                    AVG(baseline_time_ms) as avg_b_time,
                    AVG(baseline_cost) as avg_b_cost,
                    AVG(graphrag_tokens) as avg_g_tokens,
                    AVG(graphrag_time_ms) as avg_g_time,
                    AVG(graphrag_cost) as avg_g_cost,
                    AVG(graphrag_crag_grade) as avg_crag,
                    AVG(token_savings_pct) as avg_token_savings,
                    AVG(speed_improvement_pct) as avg_speed_imp,
                    AVG(cost_savings_pct) as avg_cost_savings,
                    AVG(accuracy_baseline) as avg_acc_b,
                    AVG(accuracy_graphrag) as avg_acc_g,
                    SUM(graphrag_cache_hit) as total_cache_hits
                FROM comparisons
            """).fetchone()

            total = row["total"] or 0

            # Self-improvement stats
            si_row = self._conn.execute("""
                SELECT
                    SUM(edges_updated) as total_edges,
                    SUM(entities_discovered) as total_entities,
                    COUNT(CASE WHEN cache_hit = 1 THEN 1 END) as cache_hits,
                    COUNT(*) as total_queries
                FROM graphrag_queries
            """).fetchone()

            cache_hit_rate = 0.0
            if si_row and si_row["total_queries"]:
                cache_hit_rate = (si_row["cache_hits"] or 0) / si_row["total_queries"]

            # Get prompt manager stats
            try:
                from llm.prompt_manager import get_prompt_manager
                pm = get_prompt_manager()
                prompt_versions = pm.total_refinements
                patterns_learned = 0
                try:
                    from orchestration.query_router import get_query_router
                    patterns_learned = get_query_router().patterns_learned
                except Exception:
                    pass
            except Exception:
                prompt_versions = 0
                patterns_learned = 0

            # Improvement curve
            curve = await self.get_improvement_curve()

            return MetricsSummary(
                total_queries=total,
                baseline_avg=PipelineMetrics(
                    tokens_used=int(row["avg_b_tokens"] or 0),
                    response_time_ms=float(row["avg_b_time"] or 0),
                    cost_usd=float(row["avg_b_cost"] or 0),
                ),
                graphrag_avg=PipelineMetrics(
                    tokens_used=int(row["avg_g_tokens"] or 0),
                    response_time_ms=float(row["avg_g_time"] or 0),
                    cost_usd=float(row["avg_g_cost"] or 0),
                    crag_grade=float(row["avg_crag"] or 0),
                ),
                savings={
                    "tokens_pct": float(row["avg_token_savings"] or 0),
                    "latency_pct": float(row["avg_speed_imp"] or 0),
                    "cost_pct": float(row["avg_cost_savings"] or 0),
                },
                self_improvement={
                    "cache_hit_rate": cache_hit_rate,
                    "graph_edges_updated": si_row["total_edges"] or 0 if si_row else 0,
                    "entities_discovered": si_row["total_entities"] or 0 if si_row else 0,
                    "prompt_versions_evolved": prompt_versions,
                    "query_patterns_learned": patterns_learned,
                    "avg_crag_grade": float(row["avg_crag"] or 0),
                    "total_feedback_events": si_row["total_queries"] or 0 if si_row else 0,
                },
                improvement_curve=curve,
            )

        except Exception as e:
            logger.error(f"Failed to get summary: {e}")
            return MetricsSummary()

    async def get_improvement_curve(self, batch_size: int = 10) -> list[ImprovementBatch]:
        """
        Calculate the improvement curve — how GraphRAG metrics improve over time.
        Groups queries into batches and computes rolling averages.
        
        This is THE key visualization proving the self-improvement thesis.
        """
        try:
            rows = self._conn.execute("""
                SELECT graphrag_tokens, graphrag_time_ms, graphrag_cost,
                       graphrag_crag_grade, graphrag_cache_hit,
                       accuracy_graphrag
                FROM comparisons
                ORDER BY id ASC
            """).fetchall()

            if not rows:
                return []

            batches = []
            settings = get_settings()
            bs = settings.evaluation.batch_size

            for i in range(0, len(rows), bs):
                batch_rows = rows[i:i + bs]
                if not batch_rows:
                    break

                n = len(batch_rows)
                avg_tokens = sum(r["graphrag_tokens"] for r in batch_rows) / n
                avg_time = sum(r["graphrag_time_ms"] for r in batch_rows) / n
                avg_cost = sum(r["graphrag_cost"] for r in batch_rows) / n
                avg_crag = sum(r["graphrag_crag_grade"] for r in batch_rows) / n
                cache_hits = sum(1 for r in batch_rows if r["graphrag_cache_hit"])
                avg_accuracy = sum(r["accuracy_graphrag"] for r in batch_rows) / n

                batch = ImprovementBatch(
                    batch_number=len(batches) + 1,
                    avg_tokens=avg_tokens,
                    avg_response_time_ms=avg_time,
                    avg_cost_usd=avg_cost,
                    avg_crag_grade=avg_crag,
                    cache_hit_rate=cache_hits / n,
                    avg_accuracy=avg_accuracy,
                )

                # Calculate deltas from previous batch
                if batches:
                    prev = batches[-1]
                    if prev.avg_tokens > 0:
                        batch.token_delta_pct = (
                            (prev.avg_tokens - avg_tokens) / prev.avg_tokens * 100
                        )
                    if prev.avg_response_time_ms > 0:
                        batch.speed_delta_pct = (
                            (prev.avg_response_time_ms - avg_time) / prev.avg_response_time_ms * 100
                        )
                    batch.accuracy_delta = avg_accuracy - prev.avg_accuracy

                batches.append(batch)

            return batches

        except Exception as e:
            logger.error(f"Failed to compute improvement curve: {e}")
            return []

    async def get_latest(self, n: int = 10) -> list[ComparisonResult]:
        """Get the N most recent comparisons."""
        try:
            rows = self._conn.execute("""
                SELECT * FROM comparisons ORDER BY id DESC LIMIT ?
            """, (n,)).fetchall()

            results = []
            for row in rows:
                results.append(ComparisonResult(
                    query=row["query"],
                    baseline=self._row_to_pipeline_result(row, "baseline"),
                    graphrag=self._row_to_pipeline_result(row, "graphrag"),
                    token_savings_pct=row["token_savings_pct"],
                    speed_improvement_pct=row["speed_improvement_pct"],
                    cost_savings_pct=row["cost_savings_pct"],
                ))
            return results

        except Exception as e:
            logger.error(f"Failed to get latest: {e}")
            return []

    async def get_self_improvement_stats(self) -> dict:
        """Get detailed self-improvement statistics."""
        try:
            from services.feedback_loop import get_feedback_loop
            from services.semantic_cache import get_semantic_cache
            from llm.prompt_manager import get_prompt_manager
            from orchestration.query_router import get_query_router

            feedback = get_feedback_loop()
            cache = get_semantic_cache()
            pm = get_prompt_manager()
            router = get_query_router()

            return {
                "feedback_loop": feedback.stats,
                "cache": {
                    "hit_rate": cache.hit_rate,
                    "total_entries": cache.total_entries,
                },
                "prompts": pm.get_all_stats(),
                "query_patterns_learned": router.patterns_learned,
            }
        except Exception as e:
            return {"error": str(e)}

    async def health_check(self) -> dict:
        """Check database connectivity."""
        try:
            count = self._conn.execute("SELECT COUNT(*) FROM comparisons").fetchone()[0]
            return {"status": "connected", "total_comparisons": count}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def _row_to_pipeline_result(self, row, prefix: str):
        """Convert a database row to a PipelineResult."""
        from app.models import PipelineResult, PipelineMetrics

        return PipelineResult(
            pipeline=prefix,
            answer=row.get(f"{prefix}_answer", ""),
            metrics=PipelineMetrics(
                tokens_used=row.get(f"{prefix}_tokens", 0),
                response_time_ms=row.get(f"{prefix}_time_ms", 0),
                cost_usd=row.get(f"{prefix}_cost", 0),
                crag_grade=row.get("graphrag_crag_grade", 0) if prefix == "graphrag" else None,
                cache_hit=bool(row.get("graphrag_cache_hit", 0)) if prefix == "graphrag" else False,
            ),
        )


# Singleton
_store: Optional[MetricsStore] = None


def get_metrics_store() -> MetricsStore:
    """Get or create the singleton metrics store."""
    global _store
    if _store is None:
        _store = MetricsStore()
    return _store
