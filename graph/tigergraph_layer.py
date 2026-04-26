"""
APEX Graph Layer — TigerGraph integration.
Layer 1 of the AI Factory model: handles Knowledge Graph, Vector Store, and GSQL queries.
All graph operations (retrieval, feedback, entity management) are routed through this class.
"""

import re
import time
from pathlib import Path
from typing import Optional
from loguru import logger

from app.config import get_settings
from app.models import (
    Entity,
    Relationship,
    Chunk,
    GraphContext,
    GraphPath,
)


class TigerGraphLayer:
    """
    Unified interface for all TigerGraph operations.
    
    Responsibilities:
    - Multi-hop entity retrieval
    - Hybrid search (graph traversal + vector similarity)
    - Feedback-driven edge weight updates (self-improvement)
    - Entity and relationship upserts
    - Health monitoring
    """

    def __init__(self):
        settings = get_settings()
        self._conn = None
        self._config = settings.tigergraph
        self._initialized = False
        self._local_chunks: list[Chunk] | None = None
        logger.info(f"🐯 TigerGraph Layer created (host: {self._config.host})")

    @property
    def conn(self):
        """Lazy-initialize TigerGraph connection."""
        if self._conn is None:
            try:
                import pyTigerGraph as tg

                self._conn = tg.TigerGraphConnection(
                    host=self._config.host,
                    graphname=self._config.graph_name,
                    username=self._config.username,
                    password=self._config.password,
                )

                # Try to get/set API token
                if self._config.api_token:
                    self._conn.apiToken = self._config.api_token
                else:
                    try:
                        self._conn.getToken(self._config.password)
                    except Exception:
                        logger.warning("Could not auto-generate TigerGraph API token")

                self._initialized = True
                logger.success(f"✅ Connected to TigerGraph: {self._config.graph_name}")

            except ImportError:
                logger.error("pyTigerGraph not installed. Run: pip install pyTigerGraph")
                raise
            except Exception as e:
                logger.error(f"TigerGraph connection failed: {e}")
                # Return a mock connection for development without TG
                self._conn = MockTigerGraphConnection()
                logger.warning("⚠️ Using mock TigerGraph connection for development")

        return self._conn

    async def multi_hop_retrieve(
        self,
        entities: list[Entity],
        hops: int = 2,
        top_k: int = 20,
        min_edge_weight: float = 0.3,
        include_vectors: bool = True,
    ) -> GraphContext:
        """
        Execute multi-hop retrieval starting from seed entities.
        
        1. Finds matching entities in the graph
        2. Traverses relationships up to `hops` levels
        3. Collects connected chunks
        4. Optionally includes vector similarity search results
        
        Returns a GraphContext with chunks, relationships, and traversal paths.
        """
        start = time.perf_counter()
        entity_names = [e.name for e in entities]

        try:
            # Try to run installed GSQL query
            result = self.conn.runInstalledQuery(
                "multi_hop_retrieve",
                params={
                    "seed_entities": entity_names,
                    "max_hops": hops,
                    "top_k": top_k,
                    "min_edge_weight": min_edge_weight,
                },
            )

            chunks = self._parse_chunks(result)
            relationships = self._parse_relationships(result)
            paths = self._build_paths(relationships, entity_names)

        except Exception as e:
            logger.warning(f"GSQL query failed, falling back to REST API: {e}")
            chunks, relationships, paths = await self._fallback_retrieve(
                entity_names, hops, top_k
            )

        # Optionally merge vector search results
        if include_vectors and entity_names:
            try:
                vector_chunks = await self._vector_search(entity_names[0], top_k=10)
                chunks = self._merge_and_deduplicate(chunks, vector_chunks)
            except Exception as e:
                logger.debug(f"Vector search skipped: {e}")

        elapsed = (time.perf_counter() - start) * 1000
        logger.info(
            f"Graph retrieval: {len(chunks)} chunks, "
            f"{len(relationships)} relationships, {hops} hops, {elapsed:.0f}ms"
        )

        return GraphContext(
            chunks=chunks,
            relationships=relationships,
            traversal_paths=paths,
            entities_resolved=len(entity_names),
            hops_used=hops,
        )

    async def apply_feedback(
        self,
        query_id: str,
        traversed_edges: list[str],
        crag_grade: float,
    ):
        """
        Feed CRAG grades back into graph edge weights.
        This is the CORE self-improvement mechanism for the graph.
        
        - High CRAG grade (>0.7) → strengthen edge weights
        - Low CRAG grade (<0.3) → weaken edge weights
        - Uses exponential moving average for smooth transitions
        """
        if not traversed_edges:
            return

        try:
            self.conn.runInstalledQuery(
                "update_edge_weights",
                params={
                    "query_id": query_id,
                    "traversed_edge_ids": traversed_edges,
                    "crag_grade": crag_grade,
                },
            )
            logger.info(
                f"📊 Edge weights updated: {len(traversed_edges)} edges, "
                f"CRAG grade: {crag_grade:.3f}"
            )
        except Exception as e:
            logger.warning(f"Edge weight update failed (non-critical): {e}")

    async def upsert_entities(self, entities: list[Entity]):
        """
        Add or update entities in the knowledge graph.
        Used during:
        - Initial document ingestion
        - Entity discovery in the self-improvement loop
        """
        for entity in entities:
            try:
                self.conn.upsertVertex(
                    "Entity",
                    entity.name,
                    attributes={
                        "name": entity.name,
                        "entity_type": entity.entity_type,
                        "confidence": entity.confidence,
                        "mention_count": entity.mention_count,
                    },
                )
            except Exception as e:
                logger.debug(f"Entity upsert failed for '{entity.name}': {e}")

        logger.info(f"📌 Upserted {len(entities)} entities")

    async def create_relationships(
        self,
        source_entities: list[Entity],
        target_entities: list[Entity],
        relation_type: str = "RELATED_TO",
        initial_weight: float = 0.5,
    ):
        """Create relationships between entities."""
        for source in source_entities:
            for target in target_entities:
                try:
                    self.conn.upsertEdge(
                        "Entity",
                        source.name,
                        "Relationship",
                        "Entity",
                        target.name,
                        attributes={
                            "relation_type": relation_type,
                            "weight": initial_weight,
                            "crag_confidence": 1.0,
                            "traversal_count": 0,
                        },
                    )
                except Exception as e:
                    logger.debug(
                        f"Relationship creation failed: {source.name} → {target.name}: {e}"
                    )

    async def check_entity_coverage(self, entities: list[Entity]) -> dict:
        """
        Check how many of the given entities exist in the graph.
        Used by the Query Router to decide retrieval strategy.
        """
        found = 0
        total_confidence = 0.0

        for entity in entities:
            try:
                result = self.conn.getVerticesById("Entity", entity.name)
                if result:
                    found += 1
                    total_confidence += result[0].get("confidence", 0.5)
            except Exception:
                pass

        ratio = found / max(len(entities), 1)
        avg_confidence = total_confidence / max(found, 1)

        return {
            "found": found,
            "total": len(entities),
            "ratio": ratio,
            "avg_confidence": avg_confidence,
        }

    async def get_graph_stats(self) -> dict:
        """Get basic graph statistics for the dashboard."""
        try:
            vertices = self.conn.getVertexCount("*")
            edges = self.conn.getEdgeCount("*")
            return {
                "total_vertices": vertices,
                "total_edges": edges,
                "graph_name": self._config.graph_name,
            }
        except Exception as e:
            return {"error": str(e)}

    async def health_check(self) -> dict:
        """Check TigerGraph connectivity."""
        try:
            stats = await self.get_graph_stats()
            if "error" in stats:
                return {"status": "disconnected", **stats}
            return {"status": "connected", **stats}
        except Exception as e:
            return {"status": "disconnected", "error": str(e)}

    # --- Private helpers ---

    def _parse_chunks(self, result: list) -> list[Chunk]:
        """Parse chunk data from GSQL query result."""
        chunks = []
        if result and len(result) > 0:
            raw_chunks = result[0].get("@@top_chunks", result[0].get("top_chunks", []))
            for i, c in enumerate(raw_chunks):
                attrs = c.get("attributes", c)
                chunks.append(
                    Chunk(
                        chunk_id=attrs.get("chunk_id", f"chunk_{i}"),
                        text=attrs.get("text", ""),
                        token_count=attrs.get("token_count", 0),
                        doc_id=attrs.get("doc_id"),
                        chunk_index=attrs.get("chunk_index", i),
                        score=attrs.get("score", 0.5),
                    )
                )
        return chunks

    def _parse_relationships(self, result: list) -> list[Relationship]:
        """Parse relationship data from GSQL query result."""
        relationships = []
        if result and len(result) > 1:
            raw_edges = result[1].get("@@traversed_edges", result[1].get("traversed_edges", []))
            for e in raw_edges:
                attrs = e.get("attributes", e)
                relationships.append(
                    Relationship(
                        source=e.get("from_id", ""),
                        target=e.get("to_id", ""),
                        relation_type=attrs.get("relation_type", "RELATED_TO"),
                        weight=attrs.get("weight", 1.0),
                        crag_confidence=attrs.get("crag_confidence", 1.0),
                        traversal_count=attrs.get("traversal_count", 0),
                    )
                )
        return relationships

    def _build_paths(
        self, relationships: list[Relationship], seed_entities: list[str]
    ) -> list[GraphPath]:
        """Build traversal paths from relationship data."""
        if not relationships:
            return []

        path = GraphPath(
            edges=relationships,
            entities=list(
                set(
                    [r.source for r in relationships]
                    + [r.target for r in relationships]
                )
            ),
            total_weight=sum(r.weight for r in relationships) / max(len(relationships), 1),
        )
        return [path]

    def _merge_and_deduplicate(
        self, graph_chunks: list[Chunk], vector_chunks: list[Chunk]
    ) -> list[Chunk]:
        """Merge graph and vector search results, removing duplicates."""
        seen_ids = set(c.chunk_id for c in graph_chunks)
        merged = list(graph_chunks)

        for vc in vector_chunks:
            if vc.chunk_id not in seen_ids:
                merged.append(vc)
                seen_ids.add(vc.chunk_id)

        return merged

    async def _vector_search(self, query_text: str, top_k: int = 10) -> list[Chunk]:
        """Perform vector similarity search in TigerGraph."""
        try:
            result = self.conn.runInstalledQuery(
                "vector_search",
                params={"query_text": query_text, "top_k": top_k},
            )
            return self._parse_chunks(result)
        except Exception:
            return []

    async def _fallback_retrieve(
        self, entity_names: list[str], hops: int, top_k: int
    ) -> tuple[list[Chunk], list[Relationship], list[GraphPath]]:
        """Fallback retrieval using REST API when GSQL queries aren't installed."""
        chunks = []
        relationships = []

        for name in entity_names:
            try:
                # Get vertex neighbors
                neighbors = self.conn.getVertices("Entity", where=f"name='{name}'")
                if neighbors:
                    edges = self.conn.getEdges("Entity", name)
                    for edge in edges[:top_k]:
                        attrs = edge.get("attributes", {})
                        relationships.append(
                            Relationship(
                                source=name,
                                target=edge.get("to_id", ""),
                                relation_type=attrs.get("relation_type", "RELATED_TO"),
                                weight=attrs.get("weight", 1.0),
                            )
                        )
            except Exception:
                pass

        if not chunks:
            chunks = self._local_document_retrieve(entity_names, top_k)
            relationships = relationships or self._local_relationships(entity_names)

        paths = self._build_paths(relationships, entity_names)
        return chunks, relationships, paths

    def _local_document_retrieve(self, entity_names: list[str], top_k: int) -> list[Chunk]:
        """Retrieve from local project docs when TigerGraph is unavailable."""
        local_chunks = self._load_local_chunks()
        if not local_chunks:
            return []

        query_terms = self._tokenize(" ".join(entity_names))
        if not query_terms:
            query_terms = {"graphrag", "self", "improving", "cache", "crag", "graph"}

        scored = []
        for chunk in local_chunks:
            words = self._tokenize(chunk.text)
            overlap = len(query_terms & words)
            if overlap:
                chunk.score = overlap / max(len(query_terms), 1)
                scored.append(chunk)

        if not scored:
            scored = local_chunks[:top_k]

        return sorted(scored, key=lambda item: item.score, reverse=True)[:top_k]

    def _load_local_chunks(self) -> list[Chunk]:
        """Load PRD and README paragraphs as lightweight fallback chunks."""
        if self._local_chunks is not None:
            return self._local_chunks

        root = Path(__file__).resolve().parents[1]
        docs = [
            root / "PRD_Self_Improving_GraphRAG.md",
            root / "README.md",
        ]
        chunks: list[Chunk] = []

        for doc_path in docs:
            if not doc_path.exists():
                continue
            text = doc_path.read_text(encoding="utf-8", errors="ignore")
            paragraphs = [
                re.sub(r"\s+", " ", part).strip()
                for part in re.split(r"\n\s*\n", text)
                if len(part.strip()) > 120
            ]
            for index, paragraph in enumerate(paragraphs):
                chunks.append(
                    Chunk(
                        chunk_id=f"local:{doc_path.name}:{index}",
                        text=paragraph[:900],
                        token_count=max(1, len(paragraph) // 4),
                        doc_id=doc_path.name,
                        chunk_index=index,
                        score=0.0,
                    )
                )

        self._local_chunks = chunks
        return chunks

    def _local_relationships(self, entity_names: list[str]) -> list[Relationship]:
        """Create synthetic relationships for local fallback explainability."""
        names = entity_names[:6] or ["APEX", "GraphRAG", "CRAG", "Semantic Cache"]
        relationships = []
        for source, target in zip(names, names[1:]):
            relationships.append(
                Relationship(
                    source=source,
                    target=target,
                    relation_type="LOCAL_DOC_RELATED_TO",
                    weight=0.5,
                    crag_confidence=0.5,
                )
            )
        return relationships

    def _tokenize(self, text: str) -> set[str]:
        """Tokenize text for simple local fallback retrieval."""
        stopwords = {
            "the", "and", "for", "with", "that", "this", "from", "what", "how",
            "why", "are", "does", "into", "your", "you", "use", "using",
        }
        return {
            word
            for word in re.findall(r"[a-zA-Z][a-zA-Z0-9_:-]{2,}", text.lower())
            if word not in stopwords
        }


class MockTigerGraphConnection:
    """
    Mock TigerGraph connection for development without a running TG instance.
    Returns empty results — the system degrades gracefully.
    """

    def runInstalledQuery(self, query_name: str, params: dict = None) -> list:
        logger.debug(f"[Mock TG] Query: {query_name}, params: {params}")
        return [{"@@top_chunks": []}, {"@@traversed_edges": []}]

    def upsertVertex(self, *args, **kwargs):
        pass

    def upsertEdge(self, *args, **kwargs):
        pass

    def getVerticesById(self, *args, **kwargs):
        return []

    def getVertices(self, *args, **kwargs):
        return []

    def getEdges(self, *args, **kwargs):
        return []

    def getVertexCount(self, *args):
        return 0

    def getEdgeCount(self, *args):
        return 0


# Singleton
_tg_layer: Optional[TigerGraphLayer] = None


def get_tigergraph_layer() -> TigerGraphLayer:
    """Get or create the singleton TigerGraph layer."""
    global _tg_layer
    if _tg_layer is None:
        _tg_layer = TigerGraphLayer()
    return _tg_layer
