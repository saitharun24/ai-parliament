# =============================================================================
# hats/decomposer.py — Atomic query decomposer
# =============================================================================
# Breaks a user query into its smallest independent sub-questions.
# The decomposer decides how many sub-questions exist — no artificial cap.
# Simple queries may return just 1. Complex multi-part queries return as many
# atomic components as genuinely exist.
#
# Output feeds directly into solver prompts — each hat must address every
# sub-question, giving the critic and judge a precise checklist to work from.

import json
import re
import asyncio
import httpx

from hats.utils import build_options, get_semaphore, strip_think

from config import (
    REASONING_MODEL,
    OLLAMA_BASE_URL,
    REQUEST_TIMEOUT,
    MAX_RETRIES,
    RETRY_DELAY,
    VERBOSE,
)

_DECOMPOSER_SYSTEM = """You are a query decomposer. Your job is to break a user query \
into its smallest, fully independent atomic sub-questions.

An atomic sub-question is one that:
- Asks exactly ONE thing
- Can be fully answered without knowing the answer to any other sub-question
- Would produce a meaningfully different answer from the other sub-questions

Rules:
- ALWAYS split compound queries — if the query contains multiple sentences, multiple \
questions, or uses words like "also", "and", "how", "why", "what", "should" in \
separate clauses, each clause is a separate sub-question
- Split even if the sub-questions are related — relatedness is fine, dependency is not
- Only return a single item if the query genuinely asks exactly one thing
- Order sub-questions logically: definitions and foundations before comparisons and implications

Examples:
Query: "How does photosynthesis work, why does it matter for the ecosystem, and should we be concerned about declining plant life?"
Output: [
  "How does photosynthesis work?",
  "Why is photosynthesis important for the ecosystem?",
  "Should we be concerned about declining plant life and its effect on photosynthesis?"
]
 
Query: "What is machine learning, how did it originate, what are its main types, how does it differ from traditional programming, and what are the future implications for jobs?"
Output: [
  "What is machine learning?",
  "How did machine learning originate?",
  "What are the main types of machine learning?",
  "How does machine learning differ from traditional programming?",
  "What are the future implications of machine learning for jobs?"
]

Respond ONLY with a valid JSON array of strings — no markdown, no explanation."""

def _parse_decomposer_response(raw: str) -> list[str] | None:
    """
    Extract sub-questions from the decomposer response.
    Tries JSON array first, falls back to plain text line parsing.
    """
    # Strip markdown fences
    cleaned = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()

    # Attempt 1: JSON array
    match = re.search(r"\[.*?\]", cleaned, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            if isinstance(result, list) and all(isinstance(q, str) for q in result):
                return [q.strip() for q in result if q.strip()]
        except json.JSONDecodeError:
            pass

    # Attempt 2: numbered list fallback e.g. "1. question\n2. question"
    lines = cleaned.splitlines()
    numbered = []
    for line in lines:
        line = line.strip()
        match = re.match(r'^[\d]+[.)]\s+(.+)', line)
        if match:
            numbered.append(match.group(1).strip())
    if len(numbered) >= 2:
        if VERBOSE:
            print("  [DECOMPOSER] JSON parse failed — using numbered list fallback")
        return numbered

    # Attempt 3: bullet list fallback e.g. "- question\n- question"
    bullets = []
    for line in lines:
        line = line.strip()
        match = re.match(r'^[-*•]\s+(.+)', line)
        if match:
            bullets.append(match.group(1).strip())
    if len(bullets) >= 2:
        if VERBOSE:
            print("  [DECOMPOSER] JSON parse failed — using bullet list fallback")
        return bullets

    if VERBOSE:
        print("  [DECOMPOSER] WARNING: all parse attempts failed — treating query as atomic")
    return None

async def decompose(query: str) -> list[str]:
    """
    Decompose a query into atomic sub-questions.

    Returns a list of sub-question strings.
    If the query is already atomic or parse fails, returns [query] unchanged.
    """
    raw = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            async with get_semaphore():
                async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
                    resp = await client.post(
                        f"{OLLAMA_BASE_URL}/api/chat",
                        json={
                            "model": REASONING_MODEL,
                            "messages": [
                                {"role": "system", "content": _DECOMPOSER_SYSTEM},
                                {"role": "user",
                                 "content": "How does photosynthesis work, why does it matter for the ecosystem, and should we be concerned about declining plant life?"},
                                {"role": "assistant",
                                 "content": '["How does A work?", "Why does A matter for B?", "Should we be concerned about declining A and its effect on B?"]'},
                                {"role": "user",
                                 "content": "What is machine learning, how did it originate, what are its main types, how does it differ from traditional programming, and what are the future implications for jobs?"},
                                {"role": "assistant",
                                 "content": '["What is A?", "How did A originate?", "What are the main types of A?", "How does A differ from B?", "What are the future implications of A for C?"]'},
                                {"role": "user",   "content": query},
                            ],
                            "stream": False,
                            "options": build_options("decomposer"),
                        },
                    )
                    resp.raise_for_status()
                    raw = resp.json()["message"]["content"].strip()
            break

        except (httpx.ReadTimeout, httpx.ConnectError):
            if attempt < MAX_RETRIES:
                if VERBOSE:
                    print(f"  [DECOMPOSER] Timeout (attempt {attempt}/{MAX_RETRIES}) — retrying in {RETRY_DELAY}s...")
                await asyncio.sleep(RETRY_DELAY)
            else:
                raw = None  # fallback to atomic on final failure

    sub_questions = (_parse_decomposer_response(raw) if raw else None) or [query]

    if VERBOSE:
        print(f"  [DECOMPOSER] {len(sub_questions)} sub-question(s):")
        for i, q in enumerate(sub_questions, 1):
            print(f"    {i}. {q}")

    return sub_questions


def format_sub_questions(sub_questions: list[str]) -> str:
    """
    Format sub-questions into a structured block for injection into hat prompts.
    Each hat receives this instead of the raw query — they must address every item.
    """
    if len(sub_questions) == 1:
        # Atomic query — no special formatting needed
        return sub_questions[0]

    lines = ["Answer ALL of the following sub-questions completely:\n"]
    for i, q in enumerate(sub_questions, 1):
        lines.append(f"[{i}] {q}")
    lines.append(
        "\nYour response must address each sub-question explicitly and in order. "
        "Label each answer with its number e.g. [1], [2] etc."
    )
    return "\n".join(lines)