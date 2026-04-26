"""
APEX Metrics Routes — Dashboard data API.
Serves aggregated metrics, improvement curves, and self-improvement stats.
"""

from fastapi import APIRouter
from loguru import logger

from app.models import MetricsSummary, ImprovementBatch, PipelineMetrics
from evaluation.metrics_store import get_metrics_store

router = APIRouter()


@router.get("/metrics/summary", response_model=MetricsSummary)
async def get_metrics_summary():
    """
    Get aggregated metrics for the comparison dashboard.
    Includes baseline vs GraphRAG averages, savings percentages,
    self-improvement stats, and the improvement curve over time.
    """
    store = get_metrics_store()
    summary = await store.get_summary()
    return summary


@router.get("/metrics/improvement-curve", response_model=list[ImprovementBatch])
async def get_improvement_curve():
    """
    Get the improvement curve — how GraphRAG metrics improve over query batches.
    This is the KEY visualization proving the self-improvement thesis.
    """
    store = get_metrics_store()
    curve = await store.get_improvement_curve()
    return curve


@router.get("/metrics/latest")
async def get_latest_metrics():
    """Get the most recent comparison result for the dashboard."""
    store = get_metrics_store()
    latest = await store.get_latest(n=10)
    return [r.model_dump() for r in latest]


@router.get("/metrics/self-improvement")
async def get_self_improvement_stats():
    """
    Get self-improvement specific metrics:
    - Graph edges updated
    - Entities discovered
    - Cache hit rate
    - Prompt versions evolved
    - Query patterns learned
    """
    store = get_metrics_store()
    stats = await store.get_self_improvement_stats()
    return stats
