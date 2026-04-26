"""
APEX Health Check Route — Dependency status monitoring.
"""

import time
from fastapi import APIRouter
from loguru import logger

from app.config import get_settings
from app.models import HealthStatus

router = APIRouter()

_start_time = time.time()


@router.get("/health", response_model=HealthStatus)
async def health_check():
    """
    Check the health of all dependencies:
    - TigerGraph connection
    - Redis connection
    - LLM provider
    - Metrics database
    """
    settings = get_settings()
    dependencies = {}
    overall_status = "healthy"

    def mark_if_unavailable(service_status: dict, *, optional: bool = False):
        nonlocal overall_status
        status = str(service_status.get("status", "")).lower()
        if status in {"connected", "healthy"}:
            return
        if optional and status in {"memory_only", "fallback_memory"}:
            if overall_status == "healthy":
                overall_status = "degraded"
            return
        overall_status = "degraded"

    # Check TigerGraph
    try:
        from graph.tigergraph_layer import get_tigergraph_layer
        tg = get_tigergraph_layer()
        tg_status = await tg.health_check()
        dependencies["tigergraph"] = tg_status
        mark_if_unavailable(tg_status)
    except Exception as e:
        dependencies["tigergraph"] = {"status": "disconnected", "error": str(e)}
        overall_status = "degraded"

    # Check Redis
    try:
        from services.semantic_cache import get_semantic_cache
        cache = get_semantic_cache()
        cache_status = await cache.health_check()
        dependencies["redis"] = cache_status
        mark_if_unavailable(cache_status, optional=True)
    except Exception as e:
        dependencies["redis"] = {"status": "disconnected", "error": str(e)}
        overall_status = "degraded"

    # Check LLM
    try:
        from llm.llm_layer import get_llm_layer
        llm = get_llm_layer()
        llm_status = await llm.health_check()
        dependencies["llm"] = llm_status
        mark_if_unavailable(llm_status)
    except Exception as e:
        dependencies["llm"] = {"status": "disconnected", "error": str(e)}
        overall_status = "degraded"

    # Check Metrics Store
    try:
        from evaluation.metrics_store import get_metrics_store
        store = get_metrics_store()
        store_status = await store.health_check()
        dependencies["metrics_db"] = store_status
        mark_if_unavailable(store_status, optional=True)
    except Exception as e:
        dependencies["metrics_db"] = {"status": "disconnected", "error": str(e)}
        # Metrics DB is non-critical
        if overall_status == "healthy":
            overall_status = "healthy"  # Don't degrade for metrics DB

    return HealthStatus(
        status=overall_status,
        dependencies=dependencies,
        uptime_seconds=time.time() - _start_time,
        version=settings.version,
    )
