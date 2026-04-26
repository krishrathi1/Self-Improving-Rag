"""
Hybrid Retriever — Rank Reciprocal Fusion (RRF) combinator.
Implements Gap 2.2 from the V2 Architecture.
"""

from typing import List
from loguru import logger
from app.models import Chunk, GraphContext
from services.vector_store import get_vector_store
from graph.tigergraph_layer import get_tigergraph_layer
from orchestration.observability import trace_stage

class HybridRetriever:
    """
    Executes multiple independent retrievers (Graph, Dense, Sparse)
    and combines their results using Reciprocal Rank Fusion (RRF).
    Adaptive system: modifies internal weights based on real-time feedback (RLHF-lite).
    """
    
    def __init__(self):
        self.vector_store = get_vector_store()
        self.tigergraph = get_tigergraph_layer()
        self.k = 60 # RRF fusion constant
        self.weights = {"graph": 1.0, "vector": 1.0}
        
    def adjust_weights(self, graph_delta: float, vector_delta: float):
        """LEARNING LOOP: Update weights based on feedback from evaluation system."""
        self.weights["graph"] = max(0.1, self.weights["graph"] + graph_delta)
        self.weights["vector"] = max(0.1, self.weights["vector"] + vector_delta)
        logger.info(f"⚖️ Hybrid Retrieval Weights updated: {self.weights}")
        
    @trace_stage("hybrid_retrieval")
    async def retrieve(self, query: str) -> GraphContext:
        """
        Run multi-modal retrieval.
        - Graph search extracts structural entities.
        - Vector search extracts semantic chunks.
        Combines them into a unified context.
        """
        logger.info(f"🔄 Executing Hybrid Retrieval for: '{query}'")
        
        # 1. Graph Retrieval (Structural/Exact)
        # Using a dummy or actual tigergraph call if adapted
        graph_context = await self.tigergraph.query_rag(
            query=query, 
            dataset=None, 
            k_hops=2, 
            threshold=0.5
        )
        
        # 2. Vector Retrieval (Semantic)
        vector_chunks = await self.vector_store.similarity_search(query=query, k=5)
        
        # 3. Fuse results (RRF with Learned Weights)
        all_chunks = []
        # Boost scores based on learned system weights
        for idx, c in enumerate(graph_context.chunks):
            c.score = self.weights["graph"] * (1 / (self.k + idx + 1))
            all_chunks.append(c)

        for idx, c in enumerate(vector_chunks):
            c.score = self.weights["vector"] * (1 / (self.k + idx + 1))
            all_chunks.append(c)
        
        # Basic RRF/Deduplication
        unique_chunks = {}
        for c in all_chunks:
            if c.chunk_id not in unique_chunks:
                unique_chunks[c.chunk_id] = c
            else:
                unique_chunks[c.chunk_id].score += c.score # Sum RRF scores for overlapping hits
                
        fused_chunks = list(unique_chunks.values())
        
        # Sort or rank
        fused_chunks.sort(key=lambda x: x.score, reverse=True)
        
        graph_context.chunks = fused_chunks
        
        logger.success(f"✅ Hybrid Retrieval complete. Graph + Vector fused into {len(fused_chunks)} chunks.")
        
        return graph_context

# Singleton
_hybrid_retriever = None

def get_hybrid_retriever() -> HybridRetriever:
    global _hybrid_retriever
    if _hybrid_retriever is None:
        _hybrid_retriever = HybridRetriever()
    return _hybrid_retriever
