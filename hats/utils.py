# =============================================================================
# hats/utils.py — Shared utilities for hat response processing
# =============================================================================

import re
import asyncio
from config import OLLAMA_MAX_CONCURRENT, HAT_TEMPERATURES, HAT_MAX_TOKENS, HAT_THINK, NUM_GPU_LAYERS

# Global semaphore — limits concurrent Ollama requests across ALL hats.
# asyncio.gather() still fires all tasks simultaneously but each task waits
# for the semaphore before making its HTTP call. With OLLAMA_MAX_CONCURRENT=1
# requests queue in Python with proper per-request timeouts instead of
# piling into Ollama's internal queue and triggering cascading timeouts.
_ollama_semaphore: asyncio.Semaphore | None = None
def get_semaphore() -> asyncio.Semaphore:
    """Return the global Ollama concurrency semaphore."""
    global _ollama_semaphore
    if _ollama_semaphore is None:
        _ollama_semaphore = asyncio.Semaphore(OLLAMA_MAX_CONCURRENT)
    return _ollama_semaphore


def strip_think(text: str, thinking_enabled: bool = True) -> str:
    """
    Handle deepseek-r1's <think>...</think> chain-of-thought block.

    If thinking_enabled is False, think tags are never emitted by the model
    so we return the text as-is — no regex overhead, no risk of accidentally
    stripping content that happens to contain the word 'think'.

    If thinking_enabled is True, deepseek has two behaviours:
      A) <think>reasoning...</think>\\n\\nActual response  — normal case
      B) <think>reasoning...JSON...</think>               — entire output inside think

    Strategy for True:
      1. Try content AFTER </think> — if non-empty, that is the clean response
      2. If empty (case B), extract content FROM INSIDE <think> instead.
         The actual output is buried after the reasoning text inside the block.
    """
    if not thinking_enabled:
        return text.strip()

    if "<think>" not in text:
        return text.strip()

    # Case A: extract what comes after </think>
    after = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if after:
        return after

    # Case B: nothing after </think> — pull the inner content
    match = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()

    return text.strip()


def build_options(hat: str, extra: dict | None = None) -> dict:
    """
    Build the Ollama options dict for a given hat.
    Centralises temperature, token budget, think flag, and GPU layer split
    so all hats stay in sync with config automatically.

    extra: optional overrides (e.g. {"temperature": 0.0} for one-off calls)
    """

    options = {
        "temperature": HAT_TEMPERATURES[hat],
        "num_predict": HAT_MAX_TOKENS[hat],
        "think": HAT_THINK[hat],
        "num_gpu": NUM_GPU_LAYERS,
    }
    if extra:
        options.update(extra)
    return options