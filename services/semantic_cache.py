"""
APEX Semantic Cache — Confidence-weighted semantic caching with Redis.
Only caches high-quality responses (CRAG grade >= threshold).
Part of the self-improvement loop: cache hit rate increases over time.
"""

import json
import hashlib
import time
from typing import Optional
from datetime import datetime
from loguru import logger

from app.config import get_settings
from app.models import CacheEntry


class SemanticCache:
    """
    Confidence-weighted semantic cache.
    
    Key behaviors:
    - Only caches responses with CRAG grade >= threshold (default 0.75)
    - Cache entries have confidence-scaled TTL (higher grade = longer TTL)
    - Semantic similarity matching (text hash-based for simplicity,
      upgradeable to vector similarity with Redis Stack)
    - Hit rate tracking for dashboard metrics
    
    This is a core self-improvement mechanism:
    As more high-quality responses are cached, the system gets faster
    and cheaper without any LLM calls.
    """

    def __init__(self):
        self._cache: dict[str, CacheEntry] = {}  # In-memory fallback
        self._redis = None
        self._redis_checked = False
        self._hit_count = 0
        self._miss_count = 0
        self._min_confidence = get_settings().evaluation.cache_confidence_threshold
        logger.info(f"💾 Semantic Cache initialized (min confidence: {self._min_confidence})")

    @property
    def redis(self):
        """Lazy-initialize Redis connection."""
        if self._redis is None and self._redis_checked:
            return None
        if self._redis is None:
            try:
                import redis as redis_lib
                settings = get_settings()
                self._redis = redis_lib.Redis.from_url(
                    settings.redis.url,
                    decode_responses=True,
                )
                self._redis.ping()
                self._redis_checked = True
                logger.success("✅ Redis connected for semantic cache")
            except Exception as e:
                logger.warning(f"Redis unavailable, using in-memory cache: {e}")
                self._redis = None
                self._redis_checked = True
        return self._redis

    async def get(self, query: str, threshold: float = 0.85) -> Optional[CacheEntry]:
        """
        Check cache for a semantically similar query.
        
        Currently uses text hash matching (exact match).
        Can be upgraded to vector similarity with Redis Stack.
        """
        cache_key = self._make_key(query)

        # Try Redis first
        if self.redis:
            try:
                data = self.redis.hgetall(f"apex:cache:{cache_key}")
                if data and float(data.get("confidence", 0)) >= self._min_confidence:
                    self._hit_count += 1
                    self.redis.hincrby(f"apex:cache:{cache_key}", "hit_count", 1)

                    entry = CacheEntry(
                        query=data["query"],
                        answer=data["answer"],
                        confidence=float(data["confidence"]),
                        hit_count=int(data.get("hit_count", 1)),
                    )
                    logger.info(f"⚡ Cache HIT (Redis): confidence={entry.confidence:.3f}")
                    return entry
            except Exception as e:
                logger.debug(f"Redis cache get failed: {e}")

        # Fallback to in-memory
        if cache_key in self._cache:
            entry = self._cache[cache_key]
            if entry.confidence >= self._min_confidence:
                self._hit_count += 1
                entry.hit_count += 1
                logger.info(f"⚡ Cache HIT (memory): confidence={entry.confidence:.3f}")
                return entry

        self._miss_count += 1
        return None

    async def put(
        self,
        query: str,
        answer: str,
        crag_grade: float,
    ):
        """
        Cache a query-answer pair if CRAG grade meets threshold.
        Higher grades get longer TTL.
        """
        if crag_grade < self._min_confidence:
            logger.debug(
                f"Cache SKIP: grade {crag_grade:.3f} < threshold {self._min_confidence}"
            )
            return

        cache_key = self._make_key(query)
        ttl = self._calculate_ttl(crag_grade)

        entry = CacheEntry(
            query=query,
            answer=answer,
            confidence=crag_grade,
            ttl_seconds=ttl,
        )

        # Store in Redis
        if self.redis:
            try:
                self.redis.hset(
                    f"apex:cache:{cache_key}",
                    mapping={
                        "query": query,
                        "answer": answer,
                        "confidence": str(crag_grade),
                        "created_at": datetime.utcnow().isoformat(),
                        "hit_count": "0",
                    },
                )
                self.redis.expire(f"apex:cache:{cache_key}", ttl)
                logger.info(f"💾 Cached (Redis): confidence={crag_grade:.3f}, TTL={ttl}s")
            except Exception as e:
                logger.debug(f"Redis cache put failed: {e}")

        # Always store in memory as fallback
        self._cache[cache_key] = entry
        logger.debug(f"💾 Cached (memory): {len(self._cache)} entries total")

    def _make_key(self, query: str) -> str:
        """Create a cache key from the query text."""
        # Normalize: lowercase, strip, collapse whitespace
        normalized = " ".join(query.lower().strip().split())
        return hashlib.sha256(normalized.encode()).hexdigest()[:32]

    def _calculate_ttl(self, crag_grade: float) -> int:
        """
        Calculate TTL based on CRAG grade.
        Higher confidence = longer cache lifetime.
        """
        base_ttl = get_settings().redis.cache_ttl
        # Scale: 0.75 grade → 1x TTL, 1.0 grade → 4x TTL
        multiplier = 1 + 3 * ((crag_grade - 0.75) / 0.25)
        return int(base_ttl * max(1.0, multiplier))

    @property
    def hit_rate(self) -> float:
        """Get the current cache hit rate."""
        total = self._hit_count + self._miss_count
        return self._hit_count / max(total, 1)

    @property
    def total_entries(self) -> int:
        """Get total cached entries."""
        return len(self._cache)

    async def health_check(self) -> dict:
        """Check Redis connectivity."""
        try:
            if self.redis:
                self.redis.ping()
                return {
                    "status": "connected",
                    "entries": self.total_entries,
                    "hit_rate": round(self.hit_rate, 3),
                }
        except Exception as e:
            return {"status": "fallback_memory", "error": str(e)}
        return {
            "status": "memory_only",
            "entries": self.total_entries,
            "hit_rate": round(self.hit_rate, 3),
        }


# Singleton
_cache: Optional[SemanticCache] = None


def get_semantic_cache() -> SemanticCache:
    """Get or create the singleton semantic cache."""
    global _cache
    if _cache is None:
        _cache = SemanticCache()
    return _cache
