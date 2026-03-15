# =============================================================================
# memory/store.py — ChromaDB wrapper with embed + rerank pipeline
# =============================================================================

import json
import uuid
import httpx
import chromadb
from chromadb.config import Settings
from datetime import datetime
from typing import Optional

from config import (
    OLLAMA_BASE_URL,
    EMBED_MODEL,
    RERANKER_MODEL,
    CHROMA_DB_PATH,
    CHROMA_COLLECTION,
    MEMORY_RETRIEVAL_TOP_K,
    MEMORY_RERANK_TOP_N,
    MEMORY_MIN_RELEVANCE,
    ATOMIC_CACHE_COLLECTION,
    ATOMIC_CACHE_MIN_RELEVANCE,
    ATOMIC_CACHE_MIN_SCORE,
    REQUEST_TIMEOUT,
    VERBOSE,
)


# -----------------------------------------------------------------------------
# Shared ChromaDB client — one client, two collections
# -----------------------------------------------------------------------------

_chroma_client: Optional[chromadb.PersistentClient] = None

def _get_client():
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(
            path=CHROMA_DB_PATH,
            settings=Settings(anonymized_telemetry=False),
        )
    return _chroma_client

def _get_collection():
    return _get_client().get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

def _get_atomic_collection():
    return _get_client().get_or_create_collection(
        name=ATOMIC_CACHE_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )


# -----------------------------------------------------------------------------
# Embedding via mxbai-embed-large
# -----------------------------------------------------------------------------

async def _embed(text: str) -> list[float]:
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        resp = await client.post(
            f"{OLLAMA_BASE_URL}/api/embed",
            json={"model": EMBED_MODEL, "input": text},
        )
        resp.raise_for_status()
        return resp.json()["embeddings"][0]


# -----------------------------------------------------------------------------
# Reranking via bge-reranker-v2-m3
# -----------------------------------------------------------------------------

async def _rerank(query: str, candidates: list[dict]) -> list[dict]:
    scored = []
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        for candidate in candidates:
            prompt = (
                f"Query: {query}\n\n"
                f"Passage: {candidate['text']}\n\n"
                "Rate how relevant this passage is to the query on a scale of 0 to 1. "
                "Respond with only a number."
            )
            try:
                resp = await client.post(
                    f"{OLLAMA_BASE_URL}/api/generate",
                    json={
                        "model": RERANKER_MODEL,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": 0.0, "num_predict": 8},
                    },
                )
                resp.raise_for_status()
                raw   = resp.json().get("response", "0").strip()
                score = float(raw.split()[0])
                score = max(0.0, min(1.0, score))
            except Exception:
                score = 0.0

            if score >= MEMORY_MIN_RELEVANCE:
                scored.append({**candidate, "relevance": score})

    scored.sort(key=lambda x: x["relevance"], reverse=True)
    return scored[:MEMORY_RERANK_TOP_N]


# -----------------------------------------------------------------------------
# Session store
# -----------------------------------------------------------------------------

async def store_result(
    query: str,
    solutions: list[dict],
    iteration_count: int,
):
    collection = _get_collection()
    embedding  = await _embed(query)

    doc_text = f"Query: {query}\n\n"
    for i, sol in enumerate(solutions, 1):
        doc_text += (
            f"Solution {i} [{sol['hat']} hat] (score: {sol['score']:.1f}/10):\n"
            f"{sol['content']}\n\n"
        )

    solutions_json = json.dumps(solutions)
    metadata = {
        "query":          query,
        "timestamp":      datetime.utcnow().isoformat(),
        "iterations":     iteration_count,
        "top_score":      max(s["score"] for s in solutions),
        "solution_count": len(solutions),
        "solutions_json": solutions_json if len(solutions_json) < 500_000 else "[]",
    }

    collection.add(
        ids=[str(uuid.uuid4())],
        embeddings=[embedding],
        documents=[doc_text],
        metadatas=[metadata],
    )


async def retrieve_hints(query: str) -> list[dict]:
    collection = _get_collection()
    if collection.count() == 0:
        return []

    embedding = await _embed(query)
    k         = min(MEMORY_RETRIEVAL_TOP_K, collection.count())
    results   = collection.query(
        query_embeddings=[embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    candidates = [
        {"text": doc, "metadata": meta}
        for doc, meta in zip(results["documents"][0], results["metadatas"][0])
    ]

    if not candidates:
        return []

    return await _rerank(query, candidates)


def format_hints_for_context(hints: list[dict]) -> str:
    if not hints:
        return ""

    lines = [
        "--- Relevant past solutions (from memory) ---",
        "These are high-scoring solutions to similar problems. "
        "Use them as inspiration, not as answers to copy.\n",
    ]
    for i, hint in enumerate(hints, 1):
        lines.append(
            f"[Past example {i} | relevance: {hint['relevance']:.2f}]\n"
            f"{hint['text']}\n"
        )
    lines.append("--- End of memory hints ---\n")
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Atomic question cache
# -----------------------------------------------------------------------------

async def store_atomic_solutions(
    sub_questions: list[str],
    solutions: list[dict],
):
    if not solutions or not sub_questions:
        return

    top_solution = max(solutions, key=lambda s: s.get("score", 0.0))
    if top_solution.get("score", 0.0) < ATOMIC_CACHE_MIN_SCORE:
        return

    collection = _get_atomic_collection()

    for question in sub_questions:
        embedding = await _embed(question)
        collection.add(
            ids=[str(uuid.uuid4())],
            embeddings=[embedding],
            documents=[top_solution["content"]],
            metadatas={
                "question":  question,
                "hat":       top_solution["hat"],
                "score":     top_solution["score"],
                "timestamp": datetime.utcnow().isoformat(),
            },
        )

    if VERBOSE:
        print(f"  [CACHE] Stored {len(sub_questions)} atomic solution(s)")


async def retrieve_cached_solutions(
    sub_questions: list[str],
) -> dict[str, str  | None]:
    collection = _get_atomic_collection()

    if collection.count() == 0:
        return {q: None for q in sub_questions}

    results = {}
    for question in sub_questions:
        embedding = await _embed(question)
        k         = min(3, collection.count())
        search    = collection.query(
            query_embeddings=[embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        docs      = search["documents"][0]
        metadatas = search["metadatas"][0]

        if not docs:
            results[question] = None
            continue

        candidates = [
            {"text": doc, "metadata": meta}
            for doc, meta in zip(docs, metadatas)
        ]
        reranked = await _rerank(question, candidates)

        if reranked and reranked[0]["relevance"] >= ATOMIC_CACHE_MIN_RELEVANCE:
            results[question] = reranked[0]["text"]
            if VERBOSE:
                print(
                    f"  [CACHE HIT] '{question[:60]}' "
                    f"(relevance: {reranked[0]['relevance']:.2f}, "
                    f"score: {reranked[0]['metadata'].get('score', 0):.1f}/10)"
                )
        else:
            results[question] = None
            if VERBOSE:
                print(f"  [CACHE MISS] '{question[:60]}'")

    return results