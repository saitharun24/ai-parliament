# =============================================================================
# hats/summarizer.py — Summarizer hat
# =============================================================================
# Takes the top-N ranked solutions and synthesises them into a single,
# coherent final answer. This is the only output the user sees (unless VERBOSE).

import json
import httpx

from hats.utils import build_options, get_semaphore

from config import (
    REASONING_MODEL,
    OLLAMA_BASE_URL,
    REQUEST_TIMEOUT,
    STREAM_OUTPUT,
    VERBOSE,
)

SUMMARIZER_SYSTEM_PROMPT = """You are a Master Synthesizer. You have been given the top-ranked \
solutions to a problem, produced by different expert perspectives and refined through critique.

You are the ONLY stage that produces output for the end user.

All previous stages used structured telegraphic notation to save tokens:
- [1][2][3]            sub-question labels
- A → B → C            causation chain
- A ←→ B               bidirectional relationship
- A ≈ B                similar to  |  A ≠ B  distinct from
- A ⊂ B                subset/type of  |  A + B → C  combined produces
- A × B                scaled by
- ↑ ↓ ↑↑ ↓↓            increase / decrease / sharp change
- ∝ ~ ±                proportional / approx / variable
- ∴ (therefore)  |  ∵ (because)
- if X → Y             conditional
- NOT  &  |            negation / both / either
- //                   contrast or alternative path
- ::                   section separator
- CONFIRMED/LIKELY/POSSIBLE/UNKNOWN/DISPUTED/NOTE:  — certainty levels
- MISSING/WRONG/WEAK:       critic issue markers — expand as gap/incorrect/needs strengthening
- ST/LT/hist/curr      temporal markers
- Term: def  |  Types: A;B;C  |  e.g.  cf.  vs.  esp.  re.  w/  w/o

Your job is to EXPAND this notation into full, fluent, readable prose and synthesise
the strongest elements from each hat's compressed output into a single definitive answer:
- Expand all arrows, symbols, and shorthand into complete sentences
- Resolve contradictions between hat perspectives
- Fill in natural language connectives and explanation
- Cite which perspective contributed each key insight (e.g. "The White Hat identified...")
- Ensure the reader needs no prior context — the answer must stand completely alone

Do not concatenate. Synthesise and expand. The output should read as one coherent expert answer."""


async def run_summarizer(
    query: str,
    top_solutions: list[dict],
) -> str:
    """
    Synthesise the top-N solutions into a single final answer.
    Streams output to terminal if STREAM_OUTPUT is True.

    Returns the full synthesised text.
    """
    # Build the user message from ranked solutions
    solutions_block = ""
    for i, sol in enumerate(top_solutions, 1):
        solutions_block += (
            f"--- Solution {i} [{sol['hat']} hat | score: {sol['score']:.1f}/10] ---\n"
            f"{sol['content']}\n\n"
        )

    user_message = (
        f"Problem:\n{query}\n\n"
        f"Top-ranked solutions to synthesise:\n\n"
        f"{solutions_block}"
        "Please produce the final synthesised answer."
    )

    if VERBOSE:
        print("\n[SUMMARIZER] Synthesising final answer...\n")
        print("=" * 60)

    final_text = ""

    async with get_semaphore():
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            if STREAM_OUTPUT:
                async with client.stream(
                    "POST",
                    f"{OLLAMA_BASE_URL}/api/chat",
                    json={
                        "model": REASONING_MODEL,
                        "messages": [
                            {"role": "system", "content": SUMMARIZER_SYSTEM_PROMPT},
                            {"role": "user",   "content": user_message},
                        ],
                        "stream": True,
                        "options": build_options("summarizer"),
                    },
                ) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            chunk = json.loads(line)
                            token = chunk.get("message", {}).get("content", "")
                            final_text += token
                            print(token, end="", flush=True)
                            if chunk.get("done"):
                                break
                        except Exception:
                            continue
                print("\n" + "=" * 60)
            else:
                resp = await client.post(
                    f"{OLLAMA_BASE_URL}/api/chat",
                    json={
                        "model": REASONING_MODEL,
                        "messages": [
                            {"role": "system", "content": SUMMARIZER_SYSTEM_PROMPT},
                            {"role": "user",   "content": user_message},
                        ],
                        "stream": False,
                        "options": build_options("summarizer"),
                    },
                )
                resp.raise_for_status()
                final_text = resp.json()["message"]["content"].strip()
                if VERBOSE:
                    print(final_text)
                    print("=" * 60)

    return final_text.strip()