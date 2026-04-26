"""
APEX Data Models — Pydantic schemas for API requests, responses, and internal data flow.
These models enforce type safety across all four layers of the AI Factory architecture.
"""

from pydantic import BaseModel, Field
from typing import Optional, Literal
from datetime import datetime
from enum import Enum


# ============================================
# Enums
# ============================================

class PipelineMode(str, Enum):
    """Which pipeline(s) to run."""
    BASELINE = "baseline"
    GRAPHRAG = "graphrag"
    COMPARISON = "comparison"


class CRAGLabel(str, Enum):
    """CRAG grading labels."""
    CORRECT = "CORRECT"
    AMBIGUOUS = "AMBIGUOUS"
    INCORRECT = "INCORRECT"


class RouteStrategy(str, Enum):
    """Query routing strategies."""
    GRAPH_FIRST = "graph_first"
    HYBRID_BALANCED = "hybrid_balanced"
    VECTOR_FALLBACK = "vector_fallback"


# ============================================
# API Request Models
# ============================================

class QueryRequest(BaseModel):
    """POST /api/query — Run a query through pipeline(s)."""
    query: str = Field(..., min_length=3, max_length=2000, description="The user's question")
    mode: PipelineMode = Field(
        default=PipelineMode.COMPARISON,
        description="Run baseline, graphrag, or both side-by-side"
    )
    dataset: Optional[str] = Field(
        default=None,
        description="Optional dataset/graph name to query against"
    )


class IngestRequest(BaseModel):
    """POST /api/ingest — Ingest a document into the knowledge graph."""
    content: str = Field(..., description="Raw text content to ingest")
    source: str = Field(default="manual", description="Source identifier")
    doc_type: str = Field(default="text", description="Document type: text, pdf, html")
    metadata: dict = Field(default_factory=dict)


class BenchmarkRequest(BaseModel):
    """POST /api/benchmark — Run a batch benchmark."""
    queries: list[str] = Field(..., min_length=1, description="List of queries to benchmark")
    mode: PipelineMode = Field(default=PipelineMode.COMPARISON)


# ============================================
# Internal Data Models
# ============================================

class Entity(BaseModel):
    """An extracted entity from a query or document."""
    name: str
    entity_type: str = "UNKNOWN"
    confidence: float = 1.0
    mention_count: int = 1
    embedding: Optional[list[float]] = None


class Relationship(BaseModel):
    """A relationship between two entities in the knowledge graph."""
    source: str
    target: str
    relation_type: str
    weight: float = 1.0
    crag_confidence: float = 1.0
    traversal_count: int = 0


class Chunk(BaseModel):
    """A text chunk from a document, optionally with embedding."""
    chunk_id: str
    text: str
    token_count: int = 0
    doc_id: Optional[str] = None
    chunk_index: int = 0
    embedding: Optional[list[float]] = None
    score: float = 0.0  # Relevance score from retrieval


class GraphPath(BaseModel):
    """A traversal path through the knowledge graph."""
    edges: list[Relationship] = []
    entities: list[str] = []
    total_weight: float = 0.0


class GraphContext(BaseModel):
    """Full context retrieved from TigerGraph."""
    chunks: list[Chunk] = []
    relationships: list[Relationship] = []
    traversal_paths: list[GraphPath] = []
    entities_resolved: int = 0
    hops_used: int = 0


class CRAGGrade(BaseModel):
    """Result of CRAG retrieval quality grading."""
    score: float = Field(..., ge=0.0, le=1.0)
    label: CRAGLabel
    reason: str = ""
    graded_at: datetime = Field(default_factory=datetime.utcnow)


class TokenUsage(BaseModel):
    """Token usage from an LLM call."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class LLMResponse(BaseModel):
    """Response from the LLM layer."""
    text: str
    usage: TokenUsage
    cost_usd: float = 0.0
    model: str = ""
    latency_ms: float = 0.0


class RouteDecision(BaseModel):
    """Decision from the query router."""
    strategy: RouteStrategy
    recommended_hops: int = 2
    entities: list[Entity] = []
    coverage_ratio: float = 0.0
    reasoning: str = ""


class CacheEntry(BaseModel):
    """A cached query-answer pair."""
    query: str
    answer: str
    confidence: float
    created_at: datetime = Field(default_factory=datetime.utcnow)
    hit_count: int = 0
    ttl_seconds: int = 3600


# ============================================
# API Response Models
# ============================================

class PipelineMetrics(BaseModel):
    """Metrics from a single pipeline run."""
    tokens_used: int = 0
    response_time_ms: float = 0.0
    cost_usd: float = 0.0
    crag_grade: Optional[float] = None
    cache_hit: bool = False
    retrieval_method: str = ""
    entities_resolved: int = 0
    relationships_traversed: int = 0
    hops_used: int = 0


class PipelineResult(BaseModel):
    """Result from running a single pipeline."""
    pipeline: str  # "baseline" or "graphrag"
    answer: str
    metrics: PipelineMetrics
    graph_updates_applied: bool = False
    cache_entry_created: bool = False
    metadata: dict = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    @classmethod
    def blocked(cls, reason: str) -> "PipelineResult":
        """Create a blocked result from safety filters."""
        return cls(
            pipeline="graphrag",
            answer=f"⛔ Query blocked: {reason}",
            metrics=PipelineMetrics(response_time_ms=0),
        )

    @classmethod
    def from_cache(cls, entry: CacheEntry, elapsed: float) -> "PipelineResult":
        """Create a result served from cache (0 LLM tokens)."""
        return cls(
            pipeline="graphrag",
            answer=entry.answer,
            metrics=PipelineMetrics(
                tokens_used=0,
                response_time_ms=elapsed * 1000,
                cost_usd=0.0,
                cache_hit=True,
                crag_grade=entry.confidence,
            ),
            metadata={"cache_confidence": entry.confidence, "cache_hits": entry.hit_count},
        )

    @classmethod
    def refused(cls, query: str, reason: str) -> "PipelineResult":
        """Create a graceful refusal when CRAG grade is INCORRECT."""
        return cls(
            pipeline="graphrag",
            answer=f"I don't have enough reliable information to answer: \"{query}\". "
                   f"Reason: {reason}",
            metrics=PipelineMetrics(crag_grade=0.0),
            metadata={"refused": True, "reason": reason},
        )


class AccuracyScore(BaseModel):
    """LLM-as-Judge accuracy assessment for one answer."""
    accuracy: int = Field(default=0, ge=0, le=10)
    completeness: int = Field(default=0, ge=0, le=10)
    relevance: int = Field(default=0, ge=0, le=10)
    conciseness: int = Field(default=0, ge=0, le=10)
    total: int = Field(default=0, ge=0, le=40)


class AccuracyComparison(BaseModel):
    """Comparison of accuracy between baseline and GraphRAG."""
    answer_a: AccuracyScore  # Baseline
    answer_b: AccuracyScore  # GraphRAG
    winner: Literal["baseline", "graphrag", "tie"] = "tie"
    explanation: str = ""


class ComparisonResult(BaseModel):
    """Full side-by-side comparison result."""
    query: str
    baseline: PipelineResult
    graphrag: PipelineResult
    token_savings_pct: float = 0.0
    speed_improvement_pct: float = 0.0
    cost_savings_pct: float = 0.0
    accuracy_scores: Optional[AccuracyComparison] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ImprovementBatch(BaseModel):
    """Metrics for a batch of queries showing improvement over time."""
    batch_number: int
    avg_tokens: float = 0.0
    avg_response_time_ms: float = 0.0
    avg_cost_usd: float = 0.0
    avg_crag_grade: float = 0.0
    cache_hit_rate: float = 0.0
    avg_accuracy: float = 0.0
    token_delta_pct: Optional[float] = None
    speed_delta_pct: Optional[float] = None
    accuracy_delta: Optional[float] = None


class MetricsSummary(BaseModel):
    """GET /api/metrics/summary — Dashboard data."""
    total_queries: int = 0
    baseline_avg: PipelineMetrics = PipelineMetrics()
    graphrag_avg: PipelineMetrics = PipelineMetrics()
    savings: dict = Field(default_factory=dict)
    self_improvement: dict = Field(default_factory=dict)
    improvement_curve: list[ImprovementBatch] = []


class HealthStatus(BaseModel):
    """GET /api/health — System health."""
    status: Literal["healthy", "degraded", "unhealthy"] = "healthy"
    dependencies: dict = Field(default_factory=dict)
    uptime_seconds: float = 0.0
    version: str = "1.0.0"
