"""
APEX — FastAPI Application Entry Point.
Configures middleware, lifespan, and mounts all route modules.
"""

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.config import get_settings

# Global start time for uptime tracking
_start_time: float = 0.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.
    Initializes and tears down shared resources (TigerGraph, Redis, LLM).
    """
    global _start_time
    _start_time = time.time()
    settings = get_settings()

    logger.info(f"🚀 Starting {settings.app_name} v{settings.version}")
    logger.info(f"📊 LLM Provider: {settings.llm.provider} / {settings.llm.model}")
    logger.info(f"🐯 TigerGraph: {settings.tigergraph.host}")
    logger.info(f"🔴 Redis: {settings.redis.url}")

    # --- Initialize shared resources ---
    # These will be lazily initialized when first accessed
    # to avoid blocking startup if a service is temporarily unavailable.

    yield

    # --- Cleanup ---
    logger.info("🛑 Shutting down APEX...")


def create_app() -> FastAPI:
    """Factory function to create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.version,
        description=(
            "Self-improving GraphRAG inference system. "
            "Compares baseline LLM against graph-augmented RAG with "
            "real-time benchmarking and a self-improvement feedback loop."
        ),
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # --- CORS ---
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Tighten in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- Register Routes ---
    from routes.query import router as query_router
    from routes.metrics import router as metrics_router
    from routes.health import router as health_router
    from routes.ingest import router as ingest_router

    app.include_router(query_router, prefix="/api", tags=["Query"])
    app.include_router(metrics_router, prefix="/api", tags=["Metrics"])
    app.include_router(health_router, prefix="/api", tags=["Health"])
    app.include_router(ingest_router, prefix="/api", tags=["Ingest"])

    @app.get("/", tags=["Root"])
    async def root():
        return {
            "name": settings.app_name,
            "version": settings.version,
            "docs": "/docs",
            "health": "/api/health",
        }

    return app


# Create the app instance
app = create_app()


def get_uptime() -> float:
    """Get application uptime in seconds."""
    return time.time() - _start_time
