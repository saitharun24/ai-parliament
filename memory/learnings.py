# =============================================================================
# memory/learnings.py — Model-driven per-hat learning system
# =============================================================================
# At the end of each good session (top score >= ATOMIC_CACHE_MIN_SCORE),
# the model reviews all critiques and judge reasoning for each hat and
# intelligently updates that hat's .md file by:
#   - Adding genuinely new critique patterns
#   - Adding genuinely new quality signals
#   - Consolidating similar or redundant existing entries
#   - Leaving the file unchanged if nothing new was learned
#
# One LLM call per hat — focused and context-aware.
#
# File structure:
#   memory/learnings/white.md
#   memory/learnings/red.md
#   memory/learnings/yellow.md
#   memory/learnings/green.md
#   memory/learnings/black.md
#   memory/learnings/blue.md

import asyncio
import httpx
from pathlib import Path

from config import (
    REASONING_MODEL,
    OLLAMA_BASE_URL,
    ATOMIC_CACHE_MIN_SCORE,
    REQUEST_TIMEOUT,
    VERBOSE,
)

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

_LEARNINGS_DIR = Path("./memory/learnings")


def _md_path(hat: str) -> Path:
    return _LEARNINGS_DIR / f"{hat}.md"


def _ensure_dir():
    _LEARNINGS_DIR.mkdir(parents=True, exist_ok=True)


def _read_md(hat: str) -> str:
    path = _md_path(hat)
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _write_md(hat: str, content: str):
    _ensure_dir()
    _md_path(hat).write_text(content.strip() + "\n", encoding="utf-8")


# -----------------------------------------------------------------------------
# Learning synthesizer prompt
# -----------------------------------------------------------------------------

def _build_synthesizer_prompt(
    hat: str,
    domain: str,
    critiques: list[str],
    judge_reasonings: list[str],
    existing_content: str,
    is_good_session: bool,
) -> str:
    critiques_block  = "\n".join(f"- {c}" for c in critiques) if critiques else "None"
    reasonings_block = "\n".join(f"- {r}" for r in judge_reasonings) if judge_reasonings else "None (poor session — no high-scoring solutions)"
    existing_block   = existing_content.strip() if existing_content.strip() else "Empty — no learnings yet."
    session_note     = (
        "This was a GOOD session — update both critique patterns and quality signals."
        if is_good_session else
        "This was a POOR session — update critique patterns only. Do NOT add quality signals."
    )

    return f"""You are updating the learnings file for the {hat.upper()} Hat thinker in a reasoning system.

This file is injected into the {hat} hat's system prompt on future queries to help it produce better solutions.

DOMAIN: {domain}
SESSION TYPE: {session_note}

CRITIQUES FOUND IN THIS SESSION (issues the {hat} hat's solutions had):
{critiques_block}

JUDGE REASONING FOR HIGH-SCORING SOLUTIONS FROM THE {hat.upper()} HAT:
{reasonings_block}

CURRENT LEARNINGS FILE CONTENT:
{existing_block}

Your task:
1. Review the session data above
2. Compare against the existing learnings file
3. Produce an UPDATED version of the learnings file that:
   - Adds genuinely NEW critique patterns not already covered
   - Adds genuinely NEW quality signals if this was a good session
   - CONSOLIDATES any existing entries that are similar or redundant into single cleaner entries
   - REMOVES nothing unless directly superseded by a consolidated entry
   - Keeps entries concise — one clear sentence each
   - Write entries in CLEAR, FULLY EXPANDED language — these are injected into solver
     system prompts and must be immediately actionable, not compressed shorthand
   - Input critiques may be in telegraphic notation — expand ALL symbols into full natural
     language when writing learning entries:
     → (causes/leads to) | ←→ (bidirectional) | ↑ ↓ (increase/decrease)
     ↑↑ ↓↓ (sharp increase/decrease) | ~ (approximately) | ± (variable/uncertain)
     ∝ (proportional to) | ≈ (similar to) | ≠ (distinct from) | ⊂ (subset/type of)
     A + B → C (combined produces) | A × B (scaled by)
     ∴ (therefore) | ∵ (because) | if X → Y (conditional: if X then Y)
     NOT (negation) | & (both apply) | | (either applies)
     // (contrast/alternative) | :: (section separator) | [1][2][3] (sub-question labels)
     CONFIRMED/LIKELY/POSSIBLE/UNKNOWN/DISPUTED/NOTE: → expand as certainty qualifiers
     MISSING: → gap in solution | WRONG: → incorrect claim | WEAK: → needs strengthening
     ST/LT (short/long-term) | hist/curr (historical/current)
     Term: (definition) | Types: A;B;C (list of types)
     w/ w/o esp. re. cf. vs. e.g. → with/without/especially/regarding/compare/versus/example
   - If nothing new was learned and no consolidation needed, return the existing content unchanged

Structure the output as markdown:
## {domain}
### Recurring critique patterns
- <pattern>

### Quality signals from high-scoring solutions
- <signal>

Respond with ONLY the updated markdown content — no explanation, no preamble, no code fences."""


# -----------------------------------------------------------------------------
# Per-hat learning update
# -----------------------------------------------------------------------------

async def _update_hat_learnings(
    hat: str,
    domain: str,
    critiques: list[str],
    judge_reasonings: list[str],
    is_good_session: bool,
) -> bool:
    """
    Make one LLM call to update a single hat's learnings file.
    Returns True if the file was updated, False if unchanged.
    """
    existing = _read_md(hat)
    prompt   = _build_synthesizer_prompt(
        hat=hat,
        domain=domain,
        critiques=critiques,
        judge_reasonings=judge_reasonings,
        existing_content=existing,
        is_good_session=is_good_session,
    )

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        resp = await client.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": REASONING_MODEL,
                "messages": [
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
                "options": {
                    "temperature": 0.2,
                    "num_predict": 1024,
                    "think":       False,
                    "num_gpu":     38,
                },
            },
        )
        resp.raise_for_status()
        updated = resp.json()["message"]["content"].strip()

    # Strip any accidental code fences the model added
    import re
    updated = re.sub(r"```(?:markdown|md)?", "", updated).strip().rstrip("`").strip()

    if not updated or updated == existing.strip():
        if VERBOSE:
            print(f"  [LEARNINGS] {hat} hat — no changes")
        return False

    _write_md(hat, updated)
    if VERBOSE:
        print(f"  [LEARNINGS] {hat} hat — updated")
    return True


# -----------------------------------------------------------------------------
# Public API — called at end of session
# -----------------------------------------------------------------------------

async def update_learnings_from_session(
    domain: str,
    solutions: list[dict],
    critiqued_solutions: list[dict],
    top_score: float,
):
    """
    Update learnings files after every session.

    Good session (score >= threshold):
        Updates both critique patterns AND quality signals.
    Poor session (score < threshold):
        Updates critique patterns only — poor sessions are valuable for
        learning what to avoid, even if nothing scored well enough to cache.

    solutions:           ranked list from judge (hat, content, score, reasoning)
    critiqued_solutions: list with critique text per hat
    top_score:           best judge score in this session
    """
    is_good_session = top_score >= ATOMIC_CACHE_MIN_SCORE

    if VERBOSE:
        session_type = "good" if is_good_session else "poor"
        print(f"\n  [LEARNINGS] Updating hat learnings ({session_type} session, top score: {top_score:.1f})...")

    # Build per-hat critique and reasoning collections
    hat_critiques:  dict[str, list[str]] = {}
    hat_reasonings: dict[str, list[str]] = {}

    for sol in critiqued_solutions:
        hat      = sol["hat"]
        critique = sol.get("critique", "")
        if critique and "NO ISSUES FOUND" not in critique.upper():
            hat_critiques.setdefault(hat, []).append(critique)

    # Only collect quality signals for good sessions
    if is_good_session:
        for sol in solutions:
            hat       = sol["hat"]
            reasoning = sol.get("reasoning", "")
            score     = sol.get("score", 0.0)
            if reasoning and score >= ATOMIC_CACHE_MIN_SCORE:
                hat_reasonings.setdefault(hat, []).append(f"[{score:.1f}] {reasoning}")

    # Update each active hat that has something to learn
    active_hats = list({s["hat"] for s in solutions} | {s["hat"] for s in critiqued_solutions})
    tasks = []
    for hat in active_hats:
        if hat in ("cache",):
            continue
        critiques  = hat_critiques.get(hat, [])
        reasonings = hat_reasonings.get(hat, [])
        if not critiques and not reasonings:
            continue
        tasks.append(
            _update_hat_learnings(
                hat=hat,
                domain=domain,
                critiques=critiques,
                judge_reasonings=reasonings,
                is_good_session=is_good_session,
            )
        )

    if tasks:
        await asyncio.gather(*tasks)
    elif VERBOSE:
        print("  [LEARNINGS] Nothing to update this session")


# -----------------------------------------------------------------------------
# Public API — Read
# -----------------------------------------------------------------------------

def load_hat_learnings(hat: str) -> str:
    """
    Load learnings for a given hat as a context block for injection
    into the solver system prompt. Returns empty string if file is empty.
    """
    content = _read_md(hat)
    if not content.strip():
        return ""

    return (
        f"\n\n--- Learnings from past sessions ---\n"
        f"These patterns and quality signals were accumulated from previous queries. "
        f"Use them to avoid known pitfalls and aim for the quality standard described.\n\n"
        f"{content.strip()}\n"
        f"--- End of learnings ---"
    )

def get_learnings_summary() -> str:
    """Return a one-line summary per hat for display at session end."""
    lines = []
    for hat in ["white", "red", "yellow", "green", "black", "blue"]:
        path    = _md_path(hat)
        exists  = path.exists()
        lines_n = len(path.read_text(encoding="utf-8").splitlines()) if exists else 0
        lines.append(
            f"  {hat:<12} {'exists' if exists else 'empty':<8}  {lines_n} lines"
        )
    return "\n".join(lines)