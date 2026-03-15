# =============================================================================
# hats/critic.py — Black Hat (Critic) — now parallel
# =============================================================================
# Runs all solution critiques simultaneously via asyncio.gather().
# If ALL critiques return NO ISSUES FOUND → signals orchestrator to skip fix pass.

import asyncio
import httpx

from hats.utils import strip_think, build_options, get_semaphore
from config import (
    HAT_THINK,
    REASONING_MODEL,
    OLLAMA_BASE_URL,
    REQUEST_TIMEOUT,
    MAX_RETRIES,
    RETRY_DELAY,
    VERBOSE,
)

# Telegraphic notation reference:
# Causation: A → B | Chain: A → B → C | Bidirectional: A ←→ B
# Change: ↑↓ ↑↑↓↓ | Proportional: ∝ | Approx: ~ | Variable: ±
# Logic: if X → Y | ∴ (therefore) | ∵ (because) | NOT | & | |
# Certainty: CONFIRMED: | LIKELY: | POSSIBLE: | UNKNOWN: | DISPUTED: | NOTE:
# Temporal: ST: (short-term) | LT: (long-term) | hist: | curr:
# Structure: [1][2] sub-questions | :: section sep | // contrast/alternative
# Classification: Term: def | Types: A;B;C | e.g. | cf. | vs. | w/ | w/o
# Evaluate substance not style — notation is intentional compression

_BLACK_HAT_SYSTEM = (
    "You are the Black Hat thinker in De Bono's Six Thinking Hats framework. "
    "You are the voice of critical judgment, caution, and risk assessment. "
    "Your sole job is to find what is wrong, risky, incomplete, or logically flawed "
    "in the proposed solution.\n\n"
    "Solutions use structured telegraphic notation — evaluate SUBSTANCE not style:\n"
    "A → B (cause) | A → B → C (chain) | A ←→ B (bidirectional) | A ≈ B (similar) | A ≠ B (distinct)\n"
    "A ⊂ B (subset) | A + B → C (combined) | A × B (scaled by)\n"
    "↑ ↓ ↑↑ ↓↓ (change) | ∝ (proportional) | ~ (approx) | ± (variable)\n"
    "if X → Y (conditional) | ∴ (therefore) | ∵ (because) | NOT (negation) | & (both) | | (either)\n"
    "CONFIRMED/LIKELY/POSSIBLE/UNKNOWN/DISPUTED/NOTE: (certainty) | ST/LT/hist/curr (temporal)\n"
    "Term: def | Types: A;B;C | e.g. | cf. | vs. | esp. | re. | w/ | w/o | // (contrast) | :: (section sep) | [1][2][3] (sub-questions)\n\n"
    "Think briefly, then write your critique immediately.\n\n"
    "For every issue found:\n"
    "- State the problem precisely\n"
    "- Explain why it is a problem\n"
    "- Suggest what a correct approach would look like\n\n"
    "Do NOT praise anything. Do NOT acknowledge what is correct. Find problems only.\n"
    "Be surgical and specific — vague criticism is useless.\n"
    "Structure output as a numbered list of issues.\n"
    "If the solution is genuinely flawless, write exactly: NO ISSUES FOUND\n"
    "Be as concise as possible — one short sentence per issue without losing the meaning. No padding.\n"
    "Use telegraphic notation in your critique where possible:\n"
    "→ (causes) | ←→ (bidirectional) | ↑↓ (change) | ∴ (therefore) | ∵ (because)\n"
    "MISSING: (gap in solution) | WRONG: (incorrect claim) | WEAK: (needs strengthening)\n"
    "Your critique is internal — the summarizer will expand the final answer.\n"
    "Write in compressed form, maximum signal, minimum words."
)


async def _critique_one(
    query: str,
    solution: dict,
    judge_findings: str = "",
) -> dict:
    """Critique a single solution. Returns solution dict with 'critique' added."""
    hat     = solution["hat"]
    domain = solution.get("domain", "general")
    content = solution["content"]

    if judge_findings:
        user_message = (
            f"Problem:\n{query}\n\n"
            f"Solution (from {hat} — {hat.upper()} Hat):\n{content}\n\n"
            f"The Blue Hat (Judge) found these issues you missed:\n{judge_findings}\n\n"
            "Produce a complete revised critique incorporating both your original findings "
            "and the judge's missed issues. Use telegraphic notation throughout."
        )
    else:
        user_message = (
            f"Problem:\n{query}\n\n"
            f"Solution to critique (from {hat} — {hat.upper()} Hat):\n{content}"
        )

    if VERBOSE:
        mode = "RE-CHECK" if judge_findings else "CRITIQUE"
        print(f"  [BLACK HAT | {mode}] → {hat} hat solution...")

    critique = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            async with get_semaphore():
                async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
                    resp = await client.post(
                        f"{OLLAMA_BASE_URL}/api/chat",
                        json={
                            "model": REASONING_MODEL,
                            "messages": [
                                {"role": "system", "content": _BLACK_HAT_SYSTEM},
                                {"role": "user", "content": user_message},
                            ],
                            "stream": False,
                            "options": build_options("black"),
                        },
                    )
                    resp.raise_for_status()
                    critique = strip_think(
                        resp.json()["message"]["content"],
                        thinking_enabled=HAT_THINK["black"],
                    )
                    break

        except (httpx.ReadTimeout, httpx.ConnectError) as e:
            if attempt < MAX_RETRIES:
                if VERBOSE:
                    print(
                        f"  [BLACK HAT] Timeout on {hat} (attempt {attempt}/{MAX_RETRIES}) — retrying in {RETRY_DELAY}s...")
                await asyncio.sleep(RETRY_DELAY)
            else:
                raise

    # Safety net — should never reach here if raise works correctly
    if critique is None:
        critique = "[Critic failed to produce output]"

    clean = "NO ISSUES FOUND" in critique.upper()
    if VERBOSE:
        print(f"    [{hat}] → {'Clean' if clean else 'Issues found'}")

    return {**solution, "critique": critique, "critique_clean": clean}


async def run_critic(
    query: str,
    solutions: list[dict],
    judge_findings: dict | None = None,
) -> tuple[list[dict], bool]:
    """
    Critique all solutions in parallel.

    Returns:
        (critiqued_solutions, all_clean)
        all_clean = True if EVERY solution got NO ISSUES FOUND
                  → orchestrator skips the fix pass entirely
    """
    judge_findings = judge_findings or {}

    tasks = [
        _critique_one(
            query=query,
            solution=sol,
            judge_findings=judge_findings.get(sol["hat"], ""),
        )
        for sol in solutions
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    critiqued = []
    for sol, result in zip(solutions, results):
        if isinstance(result, Exception):
            print(f"  [WARNING] Critic failed for {sol['hat']}: {repr(result)}")
            critiqued.append({**sol, "critique": "[Critic error]", "critique_clean": False})
        else:
            critiqued.append(result)

    all_clean = all(s.get("critique_clean", False) for s in critiqued)

    if all_clean and VERBOSE:
        print("  [BLACK HAT] All solutions clean — fix pass skipped")

    return critiqued, all_clean