# =============================================================================
# hats/judge.py — Blue Hat (Judge) — parallel
# =============================================================================

import re
import json
import asyncio
import httpx

from hats.utils import strip_think, build_options, get_semaphore
from config import (
    HAT_THINK,
    REASONING_MODEL,
    OLLAMA_BASE_URL,
    JUDGE_RUBRIC,
    REQUEST_TIMEOUT,
    MAX_RETRIES,
    RETRY_DELAY,
    SCORE_EXIT_THRESHOLD,
    TOP_N_SOLUTIONS,
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

JUDGE_SYSTEM_PROMPT = """You are the Blue Hat thinker in De Bono's Six Thinking Hats framework.
You are the impartial judge responsible for process control, evaluation, and synthesis.

Solutions use structured telegraphic notation — evaluate SUBSTANCE not style:
A → B (cause) | A → B → C (chain) | A ←→ B (bidirectional) | A ≈ B (similar) | A ≠ B (distinct)
A ⊂ B (subset of) | A + B → C (combined) | A × B (scaled by)
↑ ↓ ↑↑ ↓↓ (change) | ∝ (proportional) | ~ (approx) | ± (variable)
if X → Y (conditional) | ∴ (therefore) | ∵ (because) | NOT (negation) | & (both) | | (either)
CONFIRMED/LIKELY/POSSIBLE/UNKNOWN/DISPUTED/NOTE: (certainty) | ST/LT/hist/curr (temporal)
MISSING: (gap) | WRONG: (incorrect) | WEAK: (needs strengthening) (critic issue markers)
Term: def | Types: A;B;C | e.g. | cf. | vs. | esp. | re. | w/ | w/o | // (contrast) | :: (section sep) | [1][2][3] (sub-questions)

Think briefly, then output ONLY a JSON object — no markdown, no extra text after the JSON.

Score the solution on these dimensions (0-10 each):
- correctness:  Is the solution actually right and does it fully solve the problem?
- completeness: Does it address all parts and edge cases of the problem?
- clarity:      Is it well-explained, logically structured, and easy to follow?
- originality:  Does it bring a novel or insightful angle?

Check whether the Black Hat critic's findings are complete. List any issues missed.

Weighted final score: correctness 40%, completeness 25%, clarity 20%, originality 15%.

Your scoring is internal — the summarizer expands the final answer. Keep reasoning compressed, maximum signal, minimum words.

Your entire response after thinking must be this JSON and nothing else:
{
  "scores": {
    "correctness":  <float 0-10>,
    "completeness": <float 0-10>,
    "clarity":      <float 0-10>,
    "originality":  <float 0-10>
  },
  "final_score": <float 0-10>,
  "critic_missed": "<issues the critic missed, or NONE>",
  "reasoning": "<one sentence max without losing the key insight — use telegraphic notation>"
}"""


def _compute_weighted_score(scores: dict) -> float:
    """Recompute final score from raw dimensions — never trust model's self-reported value."""
    return round(
        sum(scores.get(dim, 0.0) * weight for dim, weight in JUDGE_RUBRIC.items()),
        2,
    )


def _parse_judge_response(raw: str) -> dict:
    """
    Extract JSON from the judge response.
    strip_think() already removed the <think> block upstream — this handles
    any remaining markdown fences or stray backticks.
    """
    cleaned = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    match   = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    if VERBOSE:
        print(f"  [BLUE HAT] WARNING: JSON parse failed. Raw:\n{raw[:300]}")
    return {
        "scores":        {"correctness": 0, "completeness": 0, "clarity": 0, "originality": 0},
        "final_score":   0.0,
        "critic_missed": "PARSE ERROR — could not extract judge scores",
        "reasoning":     raw[:500],
    }


async def _judge_solution(query: str, solution: dict) -> dict:
    """Score a single critiqued solution."""
    hat      = solution["hat"]
    content  = solution["content"]
    critique = solution.get("critique", "No critique available.")

    user_message = (
        f"Problem:\n{query}\n\n"
        f"Solution (from {hat} — {hat.upper()} Hat):\n{content}\n\n"
        f"Black Hat critic findings:\n{critique}\n\n"
        "Evaluate and respond with JSON only."
    )

    if VERBOSE:
        print(f"  [BLUE HAT] Scoring {hat} hat...")

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
                                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                                {"role": "user", "content": user_message},
                            ],
                            "stream": False,
                            "options": build_options("blue"),
                        },
                    )
                    resp.raise_for_status()
                    raw = strip_think(
                        resp.json()["message"]["content"],
                        thinking_enabled=HAT_THINK["blue"],
                    )
            break  # success — exit retry loop

        except (httpx.ReadTimeout, httpx.ConnectError) as e:
            if attempt < MAX_RETRIES:
                if VERBOSE:
                    print(
                        f"  [BLUE HAT] Timeout on {hat} (attempt {attempt}/{MAX_RETRIES}) — retrying in {RETRY_DELAY}s...")
                await asyncio.sleep(RETRY_DELAY)
            else:
                raise

    if raw is None: raw = ""
    parsed   = _parse_judge_response(raw)
    weighted = _compute_weighted_score(parsed.get("scores", {}))
    parsed["final_score"] = weighted

    if VERBOSE:
        print(f"    [{hat}] → {weighted:.1f}/10")

    return {
        **solution,
        "scores":        parsed.get("scores", {}),
        "score":         weighted,
        "critic_missed": parsed.get("critic_missed", "NONE"),
        "reasoning":     parsed.get("reasoning", ""),
    }


async def run_judge(query: str, solutions: list[dict]) -> dict:
    """
    Judge all solutions in parallel, then rank and return results.

    Returns:
    {
        "ranked":         list[dict],  # all solutions sorted by score desc
        "top":            list[dict],  # top-N solutions for summarizer
        "should_exit":    bool,        # True if top scores all meet threshold
        "judge_findings": dict,        # { hat_name: missed_issues } for critic re-check
    }
    """
    tasks   = [_judge_solution(query, sol) for sol in solutions]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    judged = []
    for sol, result in zip(solutions, results):
        if isinstance(result, Exception):
            print(f"  [WARNING] Judge failed for {sol['hat']}: {repr(result)}")
            judged.append({
                **sol,
                "scores": {}, "score": 0.0,
                "critic_missed": "JUDGE ERROR", "reasoning": "",
            })
        else:
            judged.append(result)

    ranked = sorted(judged, key=lambda x: x["score"], reverse=True)
    top    = ranked[:TOP_N_SOLUTIONS]

    should_exit    = all(s["score"] >= SCORE_EXIT_THRESHOLD for s in top)
    judge_findings = {
        sol["hat"]: sol["critic_missed"]
        for sol in ranked
        if sol.get("critic_missed", "NONE").strip().upper() != "NONE"
    }

    if VERBOSE:
        print(f"\n  [BLUE HAT] Rankings:")
        for i, sol in enumerate(ranked, 1):
            print(f"    {i}. {sol['hat']:12s}  score={sol['score']:.1f}/10")
        if should_exit:
            print(f"  [BLUE HAT] Threshold ({SCORE_EXIT_THRESHOLD}) met — early exit")
        if judge_findings:
            print(f"  [BLUE HAT] Missed issues found for: {list(judge_findings.keys())}")

    return {
        "ranked":         ranked,
        "top":            top,
        "should_exit":    should_exit,
        "judge_findings": judge_findings,
    }