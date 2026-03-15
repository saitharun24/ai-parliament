# =============================================================================
# hats/solver.py — De Bono solver hat runner (parallel)
# =============================================================================
# Runs only the domain-relevant solver hats simultaneously via asyncio.gather().
# Each hat gets its own system prompt = base persona + domain-specific injection.

import asyncio
import json
import httpx

from hats.utils import strip_think, build_options, get_semaphore
from memory.learnings import load_hat_learnings
from config import (
    REASONING_MODEL,
    OLLAMA_BASE_URL,
    HAT_THINK,
    HAT_PERSONAS,
    DOMAIN_INJECTIONS,
    REQUEST_TIMEOUT,
    MAX_RETRIES,
    RETRY_DELAY,
    STREAM_OUTPUT,
    VERBOSE,
)

def _build_system_prompt(hat: str, domain: str) -> str:
    """Combine base persona with domain-specific instruction."""
    base      = HAT_PERSONAS[hat]
    safe_domain = domain if (hat, domain) in DOMAIN_INJECTIONS else "general"
    injection = DOMAIN_INJECTIONS[(hat, safe_domain)]
    learnings = load_hat_learnings(hat)
    return f"{base}\n\nDomain focus: {injection}{learnings}"


async def _call_hat(
    hat: str,
    domain: str,
    query: str,
    memory_context: str = "",
    critique: str = "",
    previous_solution: str = "",
) -> dict:
    """
    Call a single De Bono solver hat.
    FIX mode activates when critique + previous_solution are provided.
    """
    system_prompt = _build_system_prompt(hat, domain)

    if critique and previous_solution:
        user_message = (
            (f"{memory_context}\n\n" if memory_context else "") +
            f"Problem:\n{query}\n\n"
            f"Your previous solution:\n{previous_solution}\n\n"
            f"Issues found by the critic (Black Hat — in telegraphic notation):\n{critique}\n\n"
            "Revise your solution to address every issue. "
            "State explicitly what you changed and why."
            "Critic markers: MISSING: (gap to fill) | WRONG: (incorrect — fix it) | WEAK: (strengthen this point).\n"
            "Your revised output must also use telegraphic notation."
        )
    else:
        user_message = (
            (f"{memory_context}\n\n" if memory_context else "") +
            f"Problem:\n{query}"
        )

    if VERBOSE:
        mode = "FIX" if critique else "INIT"
        print(f"  [{hat.upper()} HAT | {mode}] Generating...")

    response_text = ""

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            async with get_semaphore():
                async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
                    if STREAM_OUTPUT and not critique:
                        async with client.stream(
                                "POST",
                                f"{OLLAMA_BASE_URL}/api/chat",
                                json={
                                    "model": REASONING_MODEL,
                                    "messages": [
                                        {"role": "system", "content": system_prompt},
                                        {"role": "user", "content": user_message},
                                    ],
                                    "stream": True,
                                    "options": build_options(hat),
                                },
                        ) as resp:
                            resp.raise_for_status()
                            if VERBOSE:
                                print(f"  [{hat}] ", end="", flush=True)
                            async for line in resp.aiter_lines():
                                if not line.strip():
                                    continue
                                try:
                                    chunk = json.loads(line)
                                    token = chunk.get("message", {}).get("content", "")
                                    response_text += token
                                    if VERBOSE:
                                        print(token, end="", flush=True)
                                    if chunk.get("done"):
                                        break
                                except Exception:
                                    continue
                            if VERBOSE:
                                print()
                    else:
                        resp = await client.post(
                            f"{OLLAMA_BASE_URL}/api/chat",
                            json={
                                "model": REASONING_MODEL,
                                "messages": [
                                    {"role": "system", "content": system_prompt},
                                    {"role": "user", "content": user_message},
                                ],
                                "stream": False,
                                "options": build_options(hat),
                            },
                        )
                        resp.raise_for_status()
                        response_text = resp.json()["message"]["content"]
                        if VERBOSE:
                            print(f"  [{hat}] Done ({len(response_text)} chars)")
                break  # success — exit retry loop

        except (httpx.ReadTimeout, httpx.ConnectError) as e:
            if attempt < MAX_RETRIES:
                if VERBOSE:
                    print(
                        f"  [{hat.upper()} HAT] Timeout (attempt {attempt}/{MAX_RETRIES}) — retrying in {RETRY_DELAY}s...")
                await asyncio.sleep(RETRY_DELAY)
            else:
                raise

    return {
        "hat":     hat,
        "domain": domain,
        "content": strip_think(response_text, thinking_enabled=HAT_THINK[hat]),
        "score":   0.0,
    }


async def run_solvers(
    query: str,
    active_hats: list[str],
    domain: str,
    memory_context: str = "",
    critiques: dict | None = None,
    previous_solutions: dict | None = None,
) -> list[dict]:
    """
    Run all active solver hats in parallel.
    active_hats comes from the classifier — only domain-relevant hats fire.
    """
    critiques          = critiques or {}
    previous_solutions = previous_solutions or {}

    tasks = [
        _call_hat(
            hat=hat,
            domain=domain,
            query=query,
            memory_context=memory_context,
            critique=critiques.get(hat, ""),
            previous_solution=previous_solutions.get(hat, ""),
        )
        for hat in active_hats
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Sequential execution — Ollama processes one request at a time on a single GPU.
    # asyncio.gather() appears parallel but Ollama serialises internally, causing
    # timeouts as requests queue up. Sequential is identical speed with zero timeout risk.
    solutions = []
    for hat, result in zip(active_hats, results):
        if isinstance(result, Exception):
            print(f"  [WARNING] {hat} hat failed: {repr(result)}")
            solutions.append({"hat": hat, "domain": domain, "content": f"[Error: {result}]", "score": 0.0})
        else:
            solutions.append(result)

    return solutions