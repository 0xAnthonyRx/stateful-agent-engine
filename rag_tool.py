import os
import re
from typing import Any, Dict, List
from dotenv import load_dotenv
from flashrank import Ranker, RerankRequest
from langchain_core.tools import tool
import litellm
from pgvector import Vector
from pgvector.psycopg import register_vector
import psycopg
from psycopg.rows import dict_row

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/vectordb")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini/gemini-embedding-001")
EMBEDDING_DIM = 768

# Initialize the local cross-encoder from local cache
ranker = Ranker(model_name="ms-marco-TinyBERT-L-2-v2", cache_dir="/tmp/flashrank_cache")


def sanitize_tsquery(query: str) -> str:
    clean = re.sub(r"[^\w\s]", " ", query)
    tokens = [t.strip() for t in clean.split() if len(t.strip()) > 1]
    stop_verbs = {"find", "show", "get", "search", "where", "what", "how", "who"}
    meaningful = [t for t in tokens if t.lower() not in stop_verbs]
    if not meaningful:
        meaningful = tokens
    return " | ".join(meaningful) if meaningful else ""


def get_embedding(query: str) -> Vector:
    response = litellm.embedding(
        model=EMBEDDING_MODEL,
        input=[query],
        dimensions=EMBEDDING_DIM,
        num_retries=3,
    )
    return Vector(response.data[0]["embedding"])


@tool
def query_farm_knowledge_base(query: str) -> str:
    """
    Searches the official farm operations knowledge base, standard operating procedures (SOPs), 
    equipment manuals, biosecurity protocols, and invoices using hybrid RRF search.
    Use this tool whenever you need verified operating procedures, maintenance specs, 
    or historical supplier invoices.
    """
    query_vector = get_embedding(query)
    ts_query = sanitize_tsquery(query)
    keyword_condition = (
        "to_tsquery('english', %(ts_query)s)" if ts_query else "plainto_tsquery('english', %(query)s)"
    )

    sql = f"""
    WITH vector_candidates AS (
        SELECT id, content, metadata,
               ROW_NUMBER() OVER (ORDER BY embedding <=> %(vec)s::vector ASC) AS rank
        FROM documents
        ORDER BY embedding <=> %(vec)s::vector ASC
        LIMIT 10
    ),
    keyword_candidates AS (
        SELECT id, content, metadata,
               ROW_NUMBER() OVER (ORDER BY ts_rank(tsv, {keyword_condition}) DESC) AS rank
        FROM documents
        WHERE tsv @@ {keyword_condition}
        ORDER BY ts_rank(tsv, {keyword_condition}) DESC
        LIMIT 10
    )
    SELECT COALESCE(v.id, k.id) AS id,
           COALESCE(v.content, k.content) AS content,
           COALESCE(v.metadata, k.metadata) AS metadata,
           ROUND((
               COALESCE(1.0 / (60 + v.rank), 0.0) +
               COALESCE(1.0 / (60 + k.rank), 0.0)
           )::numeric, 5) AS rrf_score
    FROM vector_candidates v
    FULL OUTER JOIN keyword_candidates k ON v.id = k.id
    ORDER BY rrf_score DESC
    LIMIT 5;
    """

    with psycopg.connect(DATABASE_URL, row_factory=dict_row) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute(
                sql,
                {"vec": query_vector, "ts_query": ts_query, "query": query},
            )
            candidates = cur.fetchall()

    if not candidates:
        return "No relevant records found in the knowledge base."

    # Stage 2: Cross-Encoder Reranking
    passages = [
        {"id": c["metadata"].get("doc_id", str(c["id"])), "text": c["content"]}
        for c in candidates
    ]
    rerank_req = RerankRequest(query=query, passages=passages)
    reranked = ranker.rerank(rerank_req)

    # Filter out anything below 0.20 confidence floor
    high_confidence = [p for p in reranked if p["score"] >= 0.20][:2]

    if not high_confidence:
        return "No high-confidence documents matched your query criteria."

    output_lines = []
    for item in high_confidence:
        output_lines.append(f"[{item['id']}] (Relevance Score: {round(item['score'], 3)}):\n{item['text']}")

    return "\n\n".join(output_lines)
