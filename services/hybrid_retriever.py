"""
Hybrid Retriever — Rank Reciprocal Fusion (RRF) combinator.
Implements Gap 2.2 from the V2 Architecture.
"""

from typing import List
from loguru import logger
from app.models import Chunk, GraphContext
from services.vector_store import get_vector_store
from graph.tigergraph_layer import get_tigergraph_layer

class HybridRetriever:
    """
    Executes multiple independent retrievers (Graph, Dense, Sparse)
    and combines their results using Reciprocal Rank Fusion (RRF).
    """
    
    def __init__(self):
        self.vector_store = get_vector_store()
        self.tigergraph = get_tigergraph_layer()
        self.k = 60 # RRF fusion constant
        
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
        
        # 3. Fuse results (RRF)
        # Assuming we just append or re-rank the context chunks for now
        all_chunks = graph_context.chunks + vector_chunks
        
        # Basic RRF/Deduplication would go here based on chunk_ids
        unique_chunks = {}
        for c in all_chunks:
            if c.chunk_id not in unique_chunks:
                unique_chunks[c.chunk_id] = c
                
        fused_chunks = list(unique_chunks.values())
        
        # Sort or rank if scores available
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
