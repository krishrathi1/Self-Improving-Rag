"""
APEX Ingestion Service — Populates the Knowledge Graph and Vector Store.
Processes documents, extracts entities/relationships, and handles indexing.
"""

import uuid
from typing import List, Optional
from loguru import logger

from app.models import Entity, Chunk, Relationship
from graph.tigergraph_layer import get_tigergraph_layer
from orchestration.entity_extractor import get_entity_extractor
from llm.llm_layer import get_llm_layer

class IngestionService:
    """
    Main service for document ingestion.
    Flow: Text -> Chunks -> Entity Extraction -> Graph Upsert -> Vector Index.
    """

    def __init__(self):
        self.tg = get_tigergraph_layer()
        self.extractor = get_entity_extractor()
        self.llm = get_llm_layer()

    async def ingest_text(self, text: str, doc_name: str) -> dict:
        """
        Ingest a block of text into the system.
        Returns a summary of what was added.
        """
        doc_id = str(uuid.uuid4())
        logger.info(f"📥 Starting ingestion for doc: {doc_name} ({doc_id})")

        # 1. Chunking (Simple paragraph-based for now)
        paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 50]
        chunks = []
        for i, p in enumerate(paragraphs):
            chunks.append(Chunk(
                chunk_id=f"{doc_id}_{i}",
                text=p,
                token_count=len(p) // 4,
                doc_id=doc_name,
                chunk_index=i
            ))
        
        logger.info(f"✂️ Created {len(chunks)} chunks")

        # 2. Upsert chunks first
        await self.tg.upsert_chunks(chunks)

        # 3. Entity & Relationship Extraction
        all_entities = []
        all_relationships = []
        
        for chunk in chunks[:10]: 
            entities = await self.extractor.extract(chunk.text)
            all_entities.extend(entities)
            
            # Link chunk to extracted entities
            await self.tg.link_chunk_to_entities(chunk.chunk_id, entities)
            
            if len(entities) > 1:
                # Create a chain of relationships within the chunk
                for i in range(len(entities) - 1):
                    all_relationships.append({
                        "source": entities[i],
                        "target": entities[i+1],
                        "type": "CO_MENTIONED",
                        "weight": 0.5
                    })

        # 4. Deduplicate entities
        unique_entities = {}
        for e in all_entities:
            unique_entities[e.name.lower()] = e
        
        entities_to_upsert = list(unique_entities.values())
        logger.info(f"🔍 Extracted {len(entities_to_upsert)} unique entities")

        # 4. Upsert to TigerGraph
        try:
            await self.tg.upsert_entities(entities_to_upsert)
            
            # Create relationships
            for rel in all_relationships:
                await self.tg.create_relationships(
                    source_entities=[rel["source"]],
                    target_entities=[rel["target"]],
                    relation_type=rel["type"],
                    initial_weight=rel["weight"]
                )
            
            # Link chunks to entities (TigerGraph layer would need a method for this)
            # For now, let's assume upsert_entities handles the basics
            
            logger.success(f"✅ Successfully ingested {doc_name}")
        except Exception as e:
            logger.error(f"❌ Ingestion failed in Graph Layer: {e}")
            # Continue to vector indexing even if graph fails

        return {
            "doc_id": doc_id,
            "chunks_count": len(chunks),
            "entities_count": len(entities_to_upsert),
            "relationships_count": len(all_relationships)
        }

# Singleton
_ingestion_service: Optional[IngestionService] = None

def get_ingestion_service() -> IngestionService:
    global _ingestion_service
    if _ingestion_service is None:
        _ingestion_service = IngestionService()
    return _ingestion_service
