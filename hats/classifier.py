# =============================================================================
# hats/classifier.py — Fast query domain classifier
# =============================================================================
# Single ultra-short LLM call (256 tokens max, temp 0.0) that returns one word.
# This determines which De Bono solver hats are activated for the query.
# Cost: ~1–2 seconds. Saves time by skipping irrelevant hats.

import asyncio
import httpx

from hats.utils import build_options, get_semaphore, strip_think

from config import (
    HAT_THINK,
    REASONING_MODEL,
    OLLAMA_BASE_URL,
    SUPPORTED_DOMAINS,
    DOMAIN_HAT_SELECTION,
    REQUEST_TIMEOUT,
    MAX_RETRIES,
    RETRY_DELAY,
    VERBOSE,
)

_CLASSIFIER_SYSTEM = (
    "You are a query classifier. Classify the user's query into exactly one of these "
    "categories: coding, math, writing, research, general.\n"
    "Respond with ONE word only — the category name. No explanation, no punctuation."
)


async def classify_query(query: str) -> tuple[str, list[str]]:
    """
    Classify the query domain and return the active solver hat list.

    Returns:
        (domain, active_hats)
        domain:      one of SUPPORTED_DOMAINS
        active_hats: list of hat names to run as solvers (excludes black/blue)
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
                                {"role": "system", "content": _CLASSIFIER_SYSTEM},
                                {"role": "user",   "content": query},
                            ],
                            "stream": False,
                            "options": build_options("classifier"),
                        },
                    )
                    resp.raise_for_status()
                    raw = strip_think(
                        resp.json()["message"]["content"],
                        thinking_enabled=HAT_THINK["classifier"],
                    ).strip().lower()
            break

        except (httpx.ReadTimeout, httpx.ConnectError):
            if attempt < MAX_RETRIES:
                if VERBOSE:
                    print(f"  [CLASSIFIER] Timeout (attempt {attempt}/{MAX_RETRIES}) — retrying in {RETRY_DELAY}s...")
                await asyncio.sleep(RETRY_DELAY)
            else:
                raw = "general"  # fallback on final failure

    # Extract first word and validate
    domain = raw.split()[0] if raw else "general"
    if domain not in SUPPORTED_DOMAINS:
        domain = "general"

    active_hats = DOMAIN_HAT_SELECTION[domain]

    if VERBOSE:
        print(f"  [CLASSIFIER] Domain: {domain} → hats: {active_hats}")

    return domain, active_hats