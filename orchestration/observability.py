"""
APEX Observability Layer
Provides tracking for latency, tokens, and errors across the pipeline.
"""
import time
from functools import wraps
from loguru import logger

def trace_stage(stage_name: str):
    """
    Decorator to wrap functions with standard observability metrics.
    In a full production environment, this integrates with OpenTelemetry or LangSmith.
    """
    def decorator(func):
        # NOTE: Ideally this uses `asyncio.iscoroutinefunction` for async compatibility
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            try:
                logger.debug(f"[Trace:START] {stage_name}")
                result = await func(*args, **kwargs)
                latency = (time.perf_counter() - start_time) * 1000
                logger.info(f"[Trace:SUCCESS] {stage_name} completed in {latency:.2f}ms")
                return result
            except Exception as e:
                latency = (time.perf_counter() - start_time) * 1000
                logger.error(f"[Trace:FAILED] {stage_name} failed after {latency:.2f}ms: {e}")
                raise
        return async_wrapper
    return decorator
