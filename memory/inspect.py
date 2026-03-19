# =============================================================================
# memory/inspect.py — ChromaDB inspection utility
# =============================================================================
# Usage:
#   python -m memory.inspect                    # show summary
#   python -m memory.inspect sessions           # list all sessions
#   python -m memory.inspect cache              # list atomic cache entries
#   python -m memory.inspect search "query"     # search sessions by query
#   python -m memory.inspect session <id>       # show full session detail
#   python -m memory.inspect stats              # show collection statistics

import sys
import json
from datetime import datetime

import chromadb
from chromadb.config import Settings

from config import (
    CHROMA_DB_PATH,
    CHROMA_COLLECTION,
    ATOMIC_CACHE_COLLECTION,
)


# -----------------------------------------------------------------------------
# Client
# -----------------------------------------------------------------------------

def _get_client():
    return chromadb.PersistentClient(
        path=CHROMA_DB_PATH,
        settings=Settings(anonymized_telemetry=False),
    )


def _get_collection(name: str):
    try:
        return _get_client().get_collection(name=name)
    except Exception:
        return None


# -----------------------------------------------------------------------------
# Formatters
# -----------------------------------------------------------------------------

def _divider(char: str = "─", width: int = 60):
    print(char * width)


def _header(text: str):
    _divider("=")
    print(f"  {text}")
    _divider("=")


def _format_timestamp(ts: str) -> str:
    try:
        dt = datetime.fromisoformat(ts)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return ts


# -----------------------------------------------------------------------------
# Commands
# -----------------------------------------------------------------------------

def cmd_summary():
    """Show a high-level summary of both collections."""
    _header("ChromaDB Summary")

    for name, label in [
        (CHROMA_COLLECTION, "Session store"),
        (ATOMIC_CACHE_COLLECTION, "Atomic cache"),
    ]:
        col = _get_collection(name)
        if col is None:
            print(f"  {label:<20} — not found (no data yet)")
            continue

        count = col.count()
        print(f"  {label:<20} — {count} entries")

        if count > 0:
            results = col.get(limit=count, include=["metadatas"])
            metas   = results["metadatas"]

            if name == CHROMA_COLLECTION:
                scores = [m.get("top_score", 0) for m in metas]
                avg    = sum(scores) / len(scores) if scores else 0
                print(f"  {'Avg top score':<20} — {avg:.1f}/10")
                timestamps = [m.get("timestamp", "") for m in metas if m.get("timestamp")]
                if timestamps:
                    latest = max(timestamps)
                    print(f"  {'Latest session':<20} — {_format_timestamp(latest)}")

            elif name == ATOMIC_CACHE_COLLECTION:
                scores = [m.get("score", 0) for m in metas]
                avg    = sum(scores) / len(scores) if scores else 0
                print(f"  {'Avg cache score':<20} — {avg:.1f}/10")
                hats   = {}
                for m in metas:
                    h = m.get("hat", "unknown")
                    hats[h] = hats.get(h, 0) + 1
                print(f"  {'By hat':<20} — {dict(sorted(hats.items()))}")

    print()


def cmd_sessions(limit: int = 20):
    """List recent sessions."""
    _header(f"Sessions (last {limit})")

    col = _get_collection(CHROMA_COLLECTION)
    if col is None or col.count() == 0:
        print("  No sessions stored yet.")
        return

    results = col.get(
        limit=min(limit, col.count()),
        include=["metadatas"],
    )

    ids   = results["ids"]
    metas = results["metadatas"]

    # Sort by timestamp descending
    paired = sorted(
        zip(ids, metas),
        key=lambda x: x[1].get("timestamp", ""),
        reverse=True,
    )

    for i, (id_, meta) in enumerate(paired, 1):
        query     = meta.get("query", "")[:60]
        score     = meta.get("top_score", 0)
        iters     = meta.get("iterations", 0)
        timestamp = _format_timestamp(meta.get("timestamp", ""))
        short_id  = id_[:8]

        print(f"  [{i:>2}] {short_id}  {timestamp}  score={score:.1f}  iter={iters}")
        print(f"        {query}{'...' if len(meta.get('query','')) > 60 else ''}")
        _divider(char="·", width=60)

    print()


def cmd_cache(limit: int = 20):
    """List atomic cache entries."""
    _header(f"Atomic Cache (last {limit})")

    col = _get_collection(ATOMIC_CACHE_COLLECTION)
    if col is None or col.count() == 0:
        print("  No cache entries yet.")
        return

    results = col.get(
        limit=min(limit, col.count()),
        include=["metadatas", "documents"],
    )

    ids   = results["ids"]
    metas = results["metadatas"]
    docs  = results["documents"]

    paired = sorted(
        zip(ids, metas, docs),
        key=lambda x: x[1].get("timestamp", ""),
        reverse=True,
    )

    for i, (id_, meta, doc) in enumerate(paired, 1):
        question  = meta.get("question", "")[:60]
        score     = meta.get("score", 0)
        hat       = meta.get("hat", "?")
        timestamp = _format_timestamp(meta.get("timestamp", ""))
        short_id  = id_[:8]
        preview   = doc[:80].replace("\n", " ")

        print(f"  [{i:>2}] {short_id}  {timestamp}  score={score:.1f}  hat={hat}")
        print(f"        Q: {question}{'...' if len(meta.get('question','')) > 60 else ''}")
        print(f"        A: {preview}...")
        _divider(char="·", width=60)

    print()


def cmd_search(query: str):
    """Search sessions by query text (simple substring match)."""
    _header(f"Search: '{query}'")

    col = _get_collection(CHROMA_COLLECTION)
    if col is None or col.count() == 0:
        print("  No sessions stored yet.")
        return

    results = col.get(
        limit=col.count(),
        include=["metadatas", "documents"],
    )

    ids   = results["ids"]
    metas = results["metadatas"]
    docs  = results["documents"]

    matches = [
        (id_, meta, doc)
        for id_, meta, doc in zip(ids, metas, docs)
        if query.lower() in meta.get("query", "").lower()
        or query.lower() in doc.lower()
    ]

    if not matches:
        print(f"  No sessions matching '{query}'")
        return

    print(f"  Found {len(matches)} match(es)\n")
    for id_, meta, doc in matches:
        print(f"  ID:    {id_[:8]}")
        print(f"  Query: {meta.get('query', '')}")
        print(f"  Score: {meta.get('top_score', 0):.1f}  |  Iterations: {meta.get('iterations', 0)}")
        print(f"  Time:  {_format_timestamp(meta.get('timestamp', ''))}")
        _divider(char="·", width=60)

    print()


def cmd_session(id_prefix: str):
    """Show full detail for a session by ID prefix."""
    _header(f"Session: {id_prefix}")

    col = _get_collection(CHROMA_COLLECTION)
    if col is None or col.count() == 0:
        print("  No sessions stored yet.")
        return

    results = col.get(
        limit=col.count(),
        include=["metadatas", "documents"],
    )

    match = None
    for id_, meta, doc in zip(results["ids"], results["metadatas"], results["documents"]):
        if id_.startswith(id_prefix):
            match = (id_, meta, doc)
            break

    if not match:
        print(f"  No session found with ID starting '{id_prefix}'")
        return

    id_, meta, doc = match
    print(f"  Full ID:    {id_}")
    print(f"  Query:      {meta.get('query', '')}")
    print(f"  Score:      {meta.get('top_score', 0):.1f}/10")
    print(f"  Iterations: {meta.get('iterations', 0)}")
    print(f"  Time:       {_format_timestamp(meta.get('timestamp', ''))}")
    print()
    _divider()
    print("  SOLUTIONS:")
    _divider()

    solutions_json = meta.get("solutions_json", "[]")
    try:
        solutions = json.loads(solutions_json)
        for i, sol in enumerate(solutions, 1):
            print(f"\n  [{i}] {sol.get('hat', '?').upper()} HAT  score={sol.get('score', 0):.1f}/10")
            print(f"  {sol.get('content', '')[:300]}...")
    except Exception:
        print(doc)

    print()


def cmd_stats():
    """Show detailed statistics."""
    _header("Statistics")

    col = _get_collection(CHROMA_COLLECTION)
    if col and col.count() > 0:
        results = col.get(limit=col.count(), include=["metadatas"])
        metas   = results["metadatas"]
        scores  = [m.get("top_score", 0) for m in metas]
        iters   = [m.get("iterations", 0) for m in metas]

        print("  SESSION STORE")
        print(f"    Total sessions:    {len(metas)}")
        print(f"    Avg score:         {sum(scores)/len(scores):.2f}/10")
        print(f"    Max score:         {max(scores):.1f}/10")
        print(f"    Min score:         {min(scores):.1f}/10")
        print(f"    Avg iterations:    {sum(iters)/len(iters):.1f}")
        score_dist = {"0-4": 0, "4-7": 0, "7-9": 0, "9-10": 0}
        for s in scores:
            if s < 4:   score_dist["0-4"] += 1
            elif s < 7: score_dist["4-7"] += 1
            elif s < 9: score_dist["7-9"] += 1
            else:       score_dist["9-10"] += 1
        print(f"    Score distribution: {score_dist}")
    else:
        print("  SESSION STORE — empty")

    print()

    col = _get_collection(ATOMIC_CACHE_COLLECTION)
    if col and col.count() > 0:
        results = col.get(limit=col.count(), include=["metadatas"])
        metas   = results["metadatas"]
        scores  = [m.get("score", 0) for m in metas]
        hats    = {}
        for m in metas:
            h = m.get("hat", "unknown")
            hats[h] = hats.get(h, 0) + 1

        print("  ATOMIC CACHE")
        print(f"    Total entries:     {len(metas)}")
        print(f"    Avg score:         {sum(scores)/len(scores):.2f}/10")
        print(f"    By hat:            {dict(sorted(hats.items()))}")
    else:
        print("  ATOMIC CACHE — empty")

    print()


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

def main():
    args = sys.argv[1:]

    if not args or args[0] == "summary":
        cmd_summary()
    elif args[0] == "sessions":
        limit = int(args[1]) if len(args) > 1 else 20
        cmd_sessions(limit)
    elif args[0] == "cache":
        limit = int(args[1]) if len(args) > 1 else 20
        cmd_cache(limit)
    elif args[0] == "search" and len(args) > 1:
        cmd_search(" ".join(args[1:]))
    elif args[0] == "session" and len(args) > 1:
        cmd_session(args[1])
    elif args[0] == "stats":
        cmd_stats()
    else:
        print("Usage:")
        print("  python -m memory.inspect                  — summary")
        print("  python -m memory.inspect sessions [N]     — list sessions")
        print("  python -m memory.inspect cache [N]        — list cache entries")
        print("  python -m memory.inspect search <query>   — search sessions")
        print("  python -m memory.inspect session <id>     — session detail")
        print("  python -m memory.inspect stats            — statistics")


if __name__ == "__main__":
    main()