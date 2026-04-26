"""
APEX Query Routes — Main query endpoint with SSE streaming.
Handles both single-pipeline and comparison mode queries.
"""

import asyncio
import json
import time
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from loguru import logger

from app.models import (
    QueryRequest,
    PipelineMode,
    ComparisonResult,
    PipelineResult,
    PipelineMetrics,
)
from services.benchmark_engine import BenchmarkEngine, get_benchmark_engine
from services.baseline_pipeline import BaselinePipeline, get_baseline_pipeline
from services.graphrag_pipeline import GraphRAGPipeline, get_graphrag_pipeline

router = APIRouter()


async def _sse_event(event: str, data: dict) -> str:
    """Format a Server-Sent Event."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def _stream_comparison(query: str) -> AsyncGenerator[str, None]:
    """Stream a side-by-side comparison via SSE."""

    engine = get_benchmark_engine()

    # Notify: starting
    yield await _sse_event("status", {"message": "Starting comparison...", "query": query})

    yield await _sse_event("baseline_start", {"pipeline": "baseline", "status": "started"})
    yield await _sse_event("graphrag_start", {"pipeline": "graphrag", "status": "started"})

    async def run_baseline():
        try:
            result = await get_baseline_pipeline().run(query)
            return "baseline", result, None
        except Exception as e:
            logger.error(f"Baseline pipeline error: {e}")
            result = PipelineResult(
                pipeline="baseline",
                answer=f"Error: {str(e)}",
                metrics=PipelineMetrics(),
            )
            return "baseline", result, str(e)

    async def run_graphrag():
        try:
            result = await get_graphrag_pipeline().run(query)
            return "graphrag", result, None
        except Exception as e:
            logger.error(f"GraphRAG pipeline error: {e}")
            result = PipelineResult(
                pipeline="graphrag",
                answer=f"Error: {str(e)}",
                metrics=PipelineMetrics(),
            )
            return "graphrag", result, str(e)

    tasks = {
        asyncio.create_task(run_baseline()),
        asyncio.create_task(run_graphrag()),
    }
    baseline_result = None
    graphrag_result = None

    for finished in asyncio.as_completed(tasks):
        pipeline, result, error = await finished

        if pipeline == "baseline":
            baseline_result = result
            if error:
                yield await _sse_event("baseline_error", {"error": error})
            yield await _sse_event("baseline_complete", {
                "pipeline": "baseline",
                "answer": result.answer,
                "metrics": result.metrics.model_dump(),
            })

        if pipeline == "graphrag":
            graphrag_result = result
            if error:
                yield await _sse_event("graphrag_error", {"error": error})
            yield await _sse_event("graphrag_complete", {
                "pipeline": "graphrag",
                "answer": result.answer,
                "metrics": result.metrics.model_dump(),
                "graph_updates_applied": result.graph_updates_applied,
                "cache_entry_created": result.cache_entry_created,
            })

    # --- Comparison summary ---
    comparison = engine.compute_comparison(query, baseline_result, graphrag_result)
    yield await _sse_event("comparison", {
        "token_savings_pct": comparison.token_savings_pct,
        "speed_improvement_pct": comparison.speed_improvement_pct,
        "cost_savings_pct": comparison.cost_savings_pct,
    })

    # --- Persist metrics ---
    await engine.save_comparison(comparison)

    yield await _sse_event("done", {"message": "Comparison complete"})


@router.post("/query")
async def query(request: QueryRequest):
    """
    Main query endpoint.
    
    Modes:
    - `comparison`: Runs both pipelines and streams results via SSE
    - `baseline`: Runs only the baseline LLM pipeline
    - `graphrag`: Runs only the self-improving GraphRAG pipeline
    """
    logger.info(f"📩 Query received: mode={request.mode}, query={request.query[:80]}...")

    if request.mode == PipelineMode.COMPARISON:
        return StreamingResponse(
            _stream_comparison(request.query),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    elif request.mode == PipelineMode.BASELINE:
        pipeline = get_baseline_pipeline()
        result = await pipeline.run(request.query)
        return result.model_dump()

    elif request.mode == PipelineMode.GRAPHRAG:
        pipeline = get_graphrag_pipeline()
        result = await pipeline.run(request.query)
        return result.model_dump()

    raise HTTPException(status_code=400, detail=f"Unknown mode: {request.mode}")
