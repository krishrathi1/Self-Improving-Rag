"""
APEX Vector Store — Layer for dense vector embeddings and semantic search.
Integrates with the knowledge graph to enable Hybrid Retrieval.
Now fully implements in-memory exact K-NN (FAISS-style) indexing and retrieval.
"""

from typing import List, Optional
import uuid
import math
from loguru import logger
from app.models import Chunk
from langchain.text_splitter import RecursiveCharacterTextSplitter

class VectorStore:
    """
    In-memory Vector Database. 
    Implements Exact K-Nearest Neighbors using Cosine Similarity for Dense Retrieval,
    and BM25 for Sparse/Keyword Retrieval.
    In a full distributed cluster, this swaps to ChromaDB or Pinecone + ElasticSearch.
    """
    
    def __init__(self):
        self._collection = []
        self._bm25 = None
        self._tokenized_corpus = []
        
        # Initialize Real Embedding Model
        try:
            from sentence_transformers import SentenceTransformer
            # Using a fast, lightweight model for local execution
            self._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("🧠 Real Vector Store Online (SentenceTransformers + BM25)")
        except ImportError:
            self._embedding_model = None
            logger.warning("sentence-transformers not found! Using fallback hash embeddings.")

    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        dot_product = sum(x * y for x, y in zip(v1, v2))
        norm_v1 = math.sqrt(sum(x * x for x in v1))
        norm_v2 = math.sqrt(sum(y * y for y in v2))
        
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
            
        return dot_product / (norm_v1 * norm_v2)

    def _embed(self, texts: List[str]) -> List[List[float]]:
        """Produce dense semantic vectors for texts."""
        if self._embedding_model:
            # Generate real embeddings (returns numpy array, convert to list)
            embeddings = self._embedding_model.encode(texts)
            return embeddings.tolist()
        else:
            # Fallback mock embedder
            results = []
            for text in texts:
                vector = [0.0] * 64
                words = text.lower().split()
                for i, word in enumerate(words):
                    hash_val = hash(word)
                    vector[(hash_val + i) % 64] += 1.0
                norm = math.sqrt(sum(x * x for x in vector)) or 1.0
                results.append([x / norm for x in vector])
            return results

    async def add_texts(self, texts: List[str], metadatas: Optional[List[dict]] = None) -> List[str]:
        """Convert chunks to embeddings and insert them into the DB."""
        ids = [str(uuid.uuid4()) for _ in texts]
        
        # Batch embed all texts at once for performance
        vectors = self._embed(texts)
        
        for i, text in enumerate(texts):
            meta = metadatas[i] if metadatas else {}
            self._collection.append({
                "id": ids[i],
                "vector": vectors[i],
                "text": text,
                "metadata": meta
            })
            self._tokenized_corpus.append(text.lower().split())
            
        # Re-build BM25 index
        try:
            from rank_bm25 import BM25Okapi
            self._bm25 = BM25Okapi(self._tokenized_corpus)
        except ImportError:
            self._bm25 = None
            logger.warning("rank_bm25 not installed, sparse retrieval will be disabled.")
            
        logger.debug(f"📐 Embedded and stored {len(texts)} chunks in Vector DB (Dense + Sparse)")
        return ids

    async def similarity_search(self, query: str, k: int = 5) -> List[Chunk]:
        """Perform dense semantic search returning Chunk models."""
        logger.info(f"🔎 FAISS/Dense search executed for: '{query[:30]}...'")
        
        if not self._collection:
            return []
            
        query_vector = self._embed([query])[0]
        scored_results = []
        
        for doc in self._collection:
            score = self._cosine_similarity(query_vector, doc["vector"])
            scored_results.append((score, doc))
            
        # Sort by best score descending
        scored_results.sort(key=lambda x: x[0], reverse=True)
        top_k = scored_results[:k]
        
        chunks = []
        for rank, (score, doc) in enumerate(top_k):
            if score > 0.05:
                meta = doc["metadata"]
                chunks.append(Chunk(
                    chunk_id=doc["id"],
                    text=doc["text"],
                    doc_id=meta.get("doc_id", "unknown"),
                    chunk_index=rank,
                    score=score
                ))
                
        return chunks

    async def keyword_search(self, query: str, k: int = 5) -> List[Chunk]:
        """Perform sparse keyword search using BM25."""
        logger.info(f"🔎 BM25/Sparse search executed for: '{query[:30]}...'")
        
        if not self._collection or not self._bm25:
            return []
            
        tokenized_query = query.lower().split()
        scores = self._bm25.get_scores(tokenized_query)
        
        scored_results = [
            (score, self._collection[i])
            for i, score in enumerate(scores)
        ]
        
        scored_results.sort(key=lambda x: x[0], reverse=True)
        top_k = scored_results[:k]
        
        chunks = []
        for rank, (score, doc) in enumerate(top_k):
            if score > 0.0:  # Only return if there's some BM25 match
                meta = doc["metadata"]
                chunks.append(Chunk(
                    chunk_id=doc["id"],
                    text=doc["text"],
                    doc_id=meta.get("doc_id", "unknown"),
                    chunk_index=rank,
                    score=score
                ))
                
        return chunks

# Singleton 
_vector_store: Optional[VectorStore] = None

def get_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store
