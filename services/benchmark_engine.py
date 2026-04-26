"""
APEX Benchmark Engine — Orchestrates side-by-side pipeline comparison.
Layer 4 of the AI Factory model.
"""

from typing import Optional
from datetime import datetime
from loguru import logger

from app.models import (
    PipelineResult,
    ComparisonResult,
    AccuracyComparison,
    AccuracyScore,
)
from services.baseline_pipeline import BaselinePipeline, get_baseline_pipeline
from services.graphrag_pipeline import GraphRAGPipeline, get_graphrag_pipeline
from evaluation.metrics_store import get_metrics_store
from llm.llm_layer import get_llm_layer
from llm.prompt_manager import get_prompt_manager


class BenchmarkEngine:
    """
    Orchestrates side-by-side comparison between baseline and GraphRAG pipelines.
    Computes savings metrics and optionally runs LLM-as-Judge accuracy scoring.
    """

    def compute_comparison(
        self,
        query: str,
        baseline: PipelineResult,
        graphrag: PipelineResult,
    ) -> ComparisonResult:
        """Compute comparison metrics between the two pipeline results."""

        token_savings = self._calc_savings(
            baseline.metrics.tokens_used, graphrag.metrics.tokens_used
        )
        speed_improvement = self._calc_savings(
            baseline.metrics.response_time_ms, graphrag.metrics.response_time_ms
        )
        cost_savings = self._calc_savings(
            baseline.metrics.cost_usd, graphrag.metrics.cost_usd
        )

        return ComparisonResult(
            query=query,
            baseline=baseline,
            graphrag=graphrag,
            token_savings_pct=token_savings,
            speed_improvement_pct=speed_improvement,
            cost_savings_pct=cost_savings,
        )

    async def run_comparison_with_accuracy(
        self, query: str
    ) -> ComparisonResult:
        """
        Run both pipelines and score accuracy using LLM-as-Judge.
        This is the full comparison used for detailed benchmarking.
        """
        import asyncio

        baseline_pipeline = get_baseline_pipeline()
        graphrag_pipeline = get_graphrag_pipeline()

        # Run both pipelines
        baseline_result, graphrag_result = await asyncio.gather(
            baseline_pipeline.run(query),
            graphrag_pipeline.run(query),
        )

        # Compute basic comparison
        comparison = self.compute_comparison(query, baseline_result, graphrag_result)

        # Run LLM-as-Judge accuracy scoring
        try:
            accuracy = await self._score_accuracy(
                query, baseline_result.answer, graphrag_result.answer
            )
            comparison.accuracy_scores = accuracy
        except Exception as e:
            logger.warning(f"Accuracy scoring failed: {e}")

        return comparison

    async def save_comparison(self, comparison: ComparisonResult):
        """Persist comparison result to metrics store."""
        try:
            store = get_metrics_store()
            await store.save_comparison(comparison)
        except Exception as e:
            logger.warning(f"Failed to save comparison: {e}")

    async def _score_accuracy(
        self, query: str, answer_a: str, answer_b: str
    ) -> AccuracyComparison:
        """Use LLM-as-Judge to compare answer quality."""
        import json

        llm = get_llm_layer()
        pm = get_prompt_manager()

        prompt = pm.get(
            "accuracy_judge",
            query=query,
            answer_a=answer_a[:2000],  # Truncate long answers
            answer_b=answer_b[:2000],
        )

        response = await llm.generate(prompt, temperature=0.0, max_tokens=512)

        raw = response.text.strip()
        # Remove markdown fences
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(lines[1:-1]) if len(lines) > 2 else raw

        parsed = json.loads(raw)

        def _parse_score(data: dict) -> AccuracyScore:
            return AccuracyScore(
                accuracy=data.get("accuracy", 5),
                completeness=data.get("completeness", 5),
                relevance=data.get("relevance", 5),
                conciseness=data.get("conciseness", 5),
                total=data.get("total", 20),
            )

        winner_map = {"A": "baseline", "B": "graphrag", "tie": "tie"}
        raw_winner = parsed.get("winner", "tie")

        return AccuracyComparison(
            answer_a=_parse_score(parsed.get("answer_a", {})),
            answer_b=_parse_score(parsed.get("answer_b", {})),
            winner=winner_map.get(raw_winner, "tie"),
            explanation=parsed.get("explanation", ""),
        )

    def _calc_savings(self, baseline_val: float, graphrag_val: float) -> float:
        """Calculate percentage savings: positive = GraphRAG is better."""
        if baseline_val == 0:
            return 0.0
        return round(((baseline_val - graphrag_val) / baseline_val) * 100, 1)


# Singleton
_engine: Optional[BenchmarkEngine] = None


def get_benchmark_engine() -> BenchmarkEngine:
    """Get or create the singleton benchmark engine."""
    global _engine
    if _engine is None:
        _engine = BenchmarkEngine()
    return _engine
