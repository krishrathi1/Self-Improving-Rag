"""
APEX TigerGraph Schema — GSQL definitions for the APEX knowledge graph.
Run these commands in TigerGraph's GraphStudio or via GSQL shell.
"""

SCHEMA_GSQL = """
// ============================================
// APEX Knowledge Graph Schema
// ============================================

// --- Vertex Types ---

CREATE VERTEX Document (
    PRIMARY_ID doc_id STRING,
    title STRING,
    source_type STRING DEFAULT "text",
    content_hash STRING,
    ingested_at DATETIME
) WITH primary_id_as_attribute="true"

CREATE VERTEX Chunk (
    PRIMARY_ID chunk_id STRING,
    text STRING,
    token_count INT DEFAULT 0,
    chunk_index INT DEFAULT 0,
    embedding LIST<DOUBLE>
) WITH primary_id_as_attribute="true"

CREATE VERTEX Entity (
    PRIMARY_ID entity_id STRING,
    name STRING,
    entity_type STRING DEFAULT "UNKNOWN",
    confidence FLOAT DEFAULT 1.0,
    mention_count INT DEFAULT 1,
    last_accessed DATETIME
) WITH primary_id_as_attribute="true"

CREATE VERTEX Community (
    PRIMARY_ID community_id STRING,
    summary STRING,
    entity_count INT DEFAULT 0,
    coherence_score FLOAT DEFAULT 0.0
) WITH primary_id_as_attribute="true"

// --- Edge Types ---

CREATE DIRECTED EDGE doc_has_chunk (
    FROM Document, TO Chunk
) WITH REVERSE_EDGE="chunk_in_doc"

CREATE DIRECTED EDGE chunk_mentions_entity (
    FROM Chunk, TO Entity
) WITH REVERSE_EDGE="entity_in_chunk"

CREATE UNDIRECTED EDGE entity_relationship (
    FROM Entity, TO Entity,
    relation_type STRING DEFAULT "RELATED_TO",
    weight FLOAT DEFAULT 1.0,
    crag_confidence FLOAT DEFAULT 1.0,
    traversal_count INT DEFAULT 0,
    last_strengthened DATETIME
)

CREATE DIRECTED EDGE entity_in_community (
    FROM Entity, TO Community
) WITH REVERSE_EDGE="community_has_entity"

// --- Graph ---
CREATE GRAPH APEX_KG (
    Document, Chunk, Entity, Community,
    doc_has_chunk, chunk_in_doc,
    chunk_mentions_entity, entity_in_chunk,
    entity_relationship,
    entity_in_community, community_has_entity
)
"""

QUERIES_GSQL = """
// ============================================
// APEX GSQL Queries
// ============================================

// Query 1: Multi-Hop Entity Retrieval
// Finds seed entities, traverses N hops, collects connected chunks
CREATE QUERY multi_hop_retrieve(
    SET<STRING> seed_entities,
    INT max_hops = 2,
    INT top_k = 20,
    FLOAT min_edge_weight = 0.3
) FOR GRAPH APEX_KG {

    SetAccum<EDGE> @@traversed_edges;
    SetAccum<VERTEX> @@visited_entities;
    HeapAccum<Chunk>(20, token_count ASC) @@top_chunks;

    // Step 1: Find seed entities
    Seeds = {Entity.*};
    Matched = SELECT s FROM Seeds:s
              WHERE s.name IN seed_entities;

    // Step 2: Multi-hop traversal with confidence filtering
    Current = Matched;
    FOREACH i IN RANGE[1, max_hops] DO
        Next = SELECT t FROM Current:s -(entity_relationship:e)- Entity:t
               WHERE e.weight >= min_edge_weight
               ACCUM @@traversed_edges += e,
                     @@visited_entities += t,
                     e.traversal_count = e.traversal_count + 1;
        Current = Next;
    END;

    // Step 3: Collect chunks from visited entities
    AllEntities = @@visited_entities UNION Matched;
    Chunks = SELECT c FROM AllEntities:e -(entity_in_chunk:m)- Chunk:c
             ACCUM @@top_chunks += c;

    PRINT @@top_chunks AS top_chunks;
    PRINT @@traversed_edges AS traversed_edges;
}


// Query 2: Feedback-Driven Edge Weight Update
// Uses exponential moving average to update edge confidence
CREATE QUERY update_edge_weights(
    STRING query_id,
    SET<STRING> traversed_edge_ids,
    FLOAT crag_grade
) FOR GRAPH APEX_KG {

    FLOAT alpha = 0.3;  // Learning rate for EMA

    AllEdges = {Entity.*};
    Updated = SELECT s FROM AllEdges:s -(entity_relationship:e)- Entity:t
              WHERE e.eid IN traversed_edge_ids
              ACCUM
                  e.crag_confidence = (1.0 - alpha) * e.crag_confidence + alpha * crag_grade,
                  e.weight = e.weight * (0.8 + 0.4 * crag_grade),
                  e.last_strengthened = now();

    PRINT Updated.size() AS edges_updated;
}


// Query 3: Entity Confidence Decay
// Periodically decays unused entity confidence to reduce noise
CREATE QUERY decay_unused_entities(
    INT days_threshold = 7,
    FLOAT decay_factor = 0.95
) FOR GRAPH APEX_KG {

    Stale = SELECT e FROM Entity:e
            WHERE datetime_diff(now(), e.last_accessed) > days_threshold * 86400
            ACCUM e.confidence = e.confidence * decay_factor;

    PRINT Stale.size() AS decayed_count;
}


// Query 4: Get Entity Neighborhood
// Returns direct neighbors and their relationship types
CREATE QUERY get_entity_neighbors(
    STRING entity_name,
    INT max_neighbors = 50
) FOR GRAPH APEX_KG {

    HeapAccum<Entity>(50, confidence DESC) @@top_neighbors;

    Seed = {Entity.*};
    Start = SELECT s FROM Seed:s WHERE s.name == entity_name;

    Neighbors = SELECT t FROM Start:s -(entity_relationship:e)- Entity:t
                ACCUM @@top_neighbors += t;

    PRINT @@top_neighbors AS neighbors;
}


// Query 5: Community Summary Retrieval
// Gets entities in a community for contextual retrieval
CREATE QUERY get_community_context(
    STRING community_id_param,
    INT max_entities = 20
) FOR GRAPH APEX_KG {

    HeapAccum<Entity>(20, confidence DESC) @@community_entities;

    Comm = {Community.*};
    Target = SELECT c FROM Comm:c WHERE c.community_id == community_id_param;

    Entities = SELECT e FROM Target:c -(community_has_entity:m)- Entity:e
               ACCUM @@community_entities += e;

    PRINT @@community_entities AS entities;
}
"""

# Python helper to print these for manual execution
if __name__ == "__main__":
    print("=" * 60)
    print("APEX Knowledge Graph — Schema Definition")
    print("=" * 60)
    print(SCHEMA_GSQL)
    print("\n" + "=" * 60)
    print("APEX Knowledge Graph — GSQL Queries")
    print("=" * 60)
    print(QUERIES_GSQL)
