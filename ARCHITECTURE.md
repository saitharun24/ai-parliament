# Reasoning System — Architecture Log

A complete record of all architectural decisions, performance optimizations, and quality improvements made to the De Bono Six Thinking Hats reasoning system built on `deepseek-r1:8b` + Ollama.

---

## Architecture

### De Bono Six Thinking Hats Framework
Makes use of De Bono's structured Six Thinking Hats framework. White, Red, Yellow, and Green hats act as solvers. Black hat is the critic. Blue hat is the judge. Each hat has a distinct epistemic role, producing genuinely diverse solutions rather than variations of the same approach.

### Domain-Aware Hat Selection
A lightweight classifier call detects query domain (coding, math, writing, research, general) and activates only domain-relevant solver hats. Red hat is excluded from coding and math domains where gut feel adds less signal than structured reasoning. Saves 1-2 hat calls per query on analytical domains.

### Query Decomposition
A decomposer step breaks compound queries into atomic sub-questions before the solver hats run. Each hat must address every sub-question explicitly and label answers `[1]`, `[2]`, `[3]` etc. Gives the critic and judge a precise checklist to evaluate against rather than a vague blob. Includes three fallback parsers (JSON array → numbered list → bullet list) for robustness.

### Atomic Question Cache
ChromaDB stores solutions at the sub-question level rather than session level. On future queries, each sub-question is checked independently against the cache using embedding similarity + reranker threshold (0.75). Cache hits skip the entire solver→critic→judge pipeline for that sub-question. Fully cached queries go directly to the summarizer. System gets progressively faster as the cache fills.

### Semaphore-Controlled Concurrency
`asyncio.gather()` launches all hat tasks simultaneously but a global semaphore (`OLLAMA_MAX_CONCURRENT = 1`) ensures only one request hits Ollama at a time. This queues requests in Python rather than inside Ollama's internal queue, giving each request its full `REQUEST_TIMEOUT` budget and eliminating cascading timeout failures. Preserves the parallel task structure while solving the single-GPU serialisation problem.

### GPU/CPU Layer Split
`NUM_GPU_LAYERS` controls how many of the model's layers run on GPU vs CPU RAM. Set to 38 (all layers on GPU) for RTX 4060 8GB. Tunable — reduce if VRAM pressure causes slowdowns.

---

## Performance

### Per-Hat Token Budgets
Each hat has a tuned `num_predict` budget based on its output type. `deepseek-r1:8b` consumes 400-600 tokens on its `<think>` block before producing output. Budgets account for this overhead. Solver hats at 2048 (multi-sub-question answers), black/blue at 3072 (complex input prompts cause longer think blocks), classifier at 512 (single word output), decomposer at 1536.

### Retry Logic with Backoff
All Ollama API calls in solver, critic, judge, classifier, decomposer, and summarizer include retry logic with configurable `MAX_RETRIES` and `RETRY_DELAY`. On `ReadTimeout` or `ConnectError`, the request retries with a delay rather than failing permanently and returning an empty/error solution.

### `think` Parameter Per Hat
`HAT_THINK` controls whether deepseek-r1's chain-of-thought reasoning block is enabled per hat. The `strip_think()` utility handles both output patterns: JSON/text after `</think>` (normal case) and content inside `</think>` (token-budget truncation case).

### Early Exit on Score Threshold
The refinement loop exits early if all top solutions score above `SCORE_EXIT_THRESHOLD` (7.5). Avoids unnecessary iterations when the first pass already produces high-quality solutions.

### Skip Fix Pass When Clean
If the critic returns `NO ISSUES FOUND` for all solutions, the fix pass (another full solver round) is skipped entirely. Saves 4 solver calls (~280 seconds) on clean first-pass solutions.

### Per-Step Timing Instrumentation
Every pipeline step is timed and a breakdown table is printed at session end showing seconds and percentage per step. Enables data-driven optimization by identifying actual bottlenecks rather than guessing.

---

## Quality

### Refined Hat Personas
Each solver hat persona includes:
- Explicit `WHAT YOU DO` and `WHAT YOU NEVER DO` sections
- Epistemic classification system for White hat (CONFIRMED FACT / STRONG EVIDENCE / WEAK EVIDENCE / UNKNOWN)
- Benefit classification for Yellow hat (CERTAIN / LIKELY / POSSIBLE)
- Minimum idea count requirement for Green hat (3+ meaningfully different ideas)
- Instinct-protection rules for Red hat (never justify with logic)

### Internal Communication Compression
Solver, critic, and judge prompts include an explicit instruction that their output is internal — processed by downstream stages before reaching the user. They write in dense, compressed form with maximum signal and minimum padding. The summarizer is the only user-facing stage and explicitly owns expansion of compressed content into readable prose.

### Structured Telegraphic Notation Protocol
All inter-hat communication uses a defined telegraphic notation — a compact symbolic language that preserves full informational content while reducing token count by an estimated 40-60% compared to prose. Every stage in the pipeline (solvers, critic, judge, summarizer, learnings) has the complete notation reference embedded in its system prompt, ensuring consistent interpretation across all stages.
 
**Full notation set:**
 
| Category | Symbols | Meaning |
|---|---|---|
| Structure | `[1][2][3]` `::` `//` | Sub-question labels, section separator, contrast/alternative |
| Causation | `→` `←→` `A+B→C` | Causes, bidirectional, combined produces |
| Similarity | `≈` `≠` `⊂` `×` | Similar, distinct, subset of, scaled by |
| Change | `↑` `↓` `↑↑` `↓↓` `~` `±` `∝` | Increase, decrease, sharp change, approx, variable, proportional |
| Logic | `∴` `∵` `if→` `NOT` `&` `\|` | Therefore, because, conditional, negation, both, either |
| Certainty | `CONFIRMED:` `LIKELY:` `POSSIBLE:` `UNKNOWN:` `DISPUTED:` `NOTE:` | Six certainty levels |
| Critic markers | `MISSING:` `WRONG:` `WEAK:` | Gap in solution, incorrect claim, needs strengthening |
| Temporal | `ST:` `LT:` `hist:` `curr:` | Short-term, long-term, historical, current |
| Classification | `Term:` `Types:` `e.g.` `cf.` `vs.` `esp.` `re.` `w/` `w/o` | Definition, list, example, compare, versus, especially, regarding, with, without |
 
**Communication flow:**
```
Solver    →  Critic    telegraphic notation
Critic    →  Judge     telegraphic notation (+ MISSING:/WRONG:/WEAK: markers)
Judge     →  Critic    telegraphic notation (critic_missed field)
Critic    →  Solver    telegraphic notation (fix pass)
All       →  Summarizer telegraphic notation → expanded to full prose
All       →  Learnings  telegraphic notation → expanded to clear actionable language
```
 
The summarizer and learnings system both receive the full notation reference and are explicitly instructed to expand every symbol into natural language. The learnings `.md` files are always written in clear expanded language regardless of how compressed the input was, ensuring solver prompts remain readable.

### Brevity Instructions for Black and Blue Hats
Black hat (critic): one concise sentence per issue without losing meaning, no padding, internal format.
Blue hat (judge): one sentence max for reasoning field without losing the key insight, compressed scoring.

### Domain-Specific Injection
Each solver hat receives a domain-specific instruction suffix at runtime (e.g. White hat in coding domain: "Focus on: language specs, algorithm complexity, known bugs, API contracts, benchmark data"). Keeps hat behaviour relevant to the query type without bloating the base persona.

### Memory Hints via Reranker
Past sessions are retrieved via ChromaDB cosine similarity then re-scored by `bge-reranker-v2-m3` before injection as context hints into solver prompts. Two-stage retrieve-then-rerank eliminates false positives from pure embedding similarity. Minimum relevance threshold (0.4) prevents noise injection.

### Judge Scores Recomputed
The judge's `final_score` field is always recomputed from raw dimension scores using `JUDGE_RUBRIC` weights rather than trusting the model's self-reported value. Prevents score drift from the model's own arithmetic errors.

### Structured Scoring Rubric
Judge evaluates on four weighted dimensions: correctness (40%), completeness (25%), clarity (20%), originality (15%). Weights tunable in `config.py`.

---

## Memory & Learning

### Shared ChromaDB Client
A single `chromadb.PersistentClient` instance serves both the session store and atomic cache collections. Previously two separate clients were instantiated per call, creating redundant connections to the same path.

### Model-Driven Learning System
At the end of every session the model updates per-hat `.md` learning files by reviewing all critiques and judge reasoning. The model checks existing file content, adds only genuinely new insights, consolidates similar entries, and writes in clear expanded language (not compressed shorthand). Learning files are injected into solver system prompts on future queries.

- **Every session**: critique patterns updated (even poor sessions — failures are valuable)
- **Good sessions only** (score ≥ 7.0): quality signals also updated
- **Deduplication**: model checks existing content before adding, consolidates redundant entries
- **Expansion**: learning entries written in clear actionable language despite compressed pipeline inputs

### ChromaDB Inspector
`memory/inspect.py` provides a CLI for browsing stored data:
```
python -m memory.inspect                  — summary
python -m memory.inspect sessions [N]     — list recent sessions
python -m memory.inspect cache [N]        — list atomic cache entries
python -m memory.inspect search <query>   — search by topic
python -m memory.inspect session <id>     — full session detail
python -m memory.inspect stats            — score distributions
```

### Solutions JSON Size Guard
`solutions_json` metadata stored in ChromaDB is capped at 500,000 characters before storage to prevent ChromaDB's 1MB metadata value limit from being exceeded silently.

---

## Robustness

### `strip_think` Utility
Centralised utility in `hats/utils.py` handles deepseek-r1's `<think>` block in two cases:
- **Case A**: content after `</think>` — extract post-think response
- **Case B**: empty after `</think>` (token budget truncated think block) — extract content from inside `<think>` instead

Applied at every Ollama response capture point. `thinking_enabled` parameter short-circuits for hats with `think: False`.

### `build_options` Utility
Centralised function in `hats/utils.py` builds the Ollama options dict for any hat from config. All hats stay in sync with temperature, token budget, think flag, and GPU layer settings automatically. Single place to change any parameter.

### `None` Safety Nets
Critic and judge both guard against `None` response text before processing. If all retries fail, a placeholder string is used rather than crashing with `AttributeError` on `.upper()` or `json.loads(None)`.

### Fallback Domain
Classifier always falls back to `"general"` if the model returns an unrecognised domain or the call fails entirely. Pipeline never aborts due to classification failure.

### Atomic Cache Always Returns
`retrieve_cached_solutions` returns `{q: None for q in sub_questions}` on empty collection rather than raising. Pipeline handles a fully empty cache gracefully on first run.

---

## Configuration

All tuneable parameters live in `config.py`. No magic numbers elsewhere. Key settings:

| Parameter | Default | Effect |
|---|---|---|
| `MAX_ITERATIONS` | 1 | Refinement loop iterations |
| `SCORE_EXIT_THRESHOLD` | 7.5 | Early exit score |
| `TOP_N_SOLUTIONS` | 3 | Solutions passed to summarizer |
| `OLLAMA_MAX_CONCURRENT` | 1 | Max concurrent Ollama requests |
| `NUM_GPU_LAYERS` | 38 | GPU layer split |
| `ATOMIC_CACHE_MIN_RELEVANCE` | 0.75 | Cache hit threshold |
| `ATOMIC_CACHE_MIN_SCORE` | 7.0 | Min score to cache a solution |
| `REQUEST_TIMEOUT` | 300 | Per-call timeout in seconds |
| `MAX_RETRIES` | 3 | Retry attempts on timeout |
| `RETRY_DELAY` | 5 | Seconds between retries |