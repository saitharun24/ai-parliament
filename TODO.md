# TODO

Active development items, ideas, and known issues for AI Parliament.
Items are grouped by area. Check off completed items and move to DONE at the bottom.

---

## Performance

- [ ] Investigate Option B (per-sub-question pipeline) once hardware is upgraded
- [ ] Profile judge think block length — explore prompt trimming to reduce judge time further
- [ ] Test `deepseek-r1:1.5b` as solver hat model with `deepseek-r1:8b` reserved for critic/judge only
- [ ] Benchmark telegraphic notation token savings vs baseline prose across 20+ queries

---

## Quality

- [ ] Evaluate hybrid critic/judge approach — solvers answer all sub-questions together, critic/judge evaluate per sub-question by splitting `[1][2][3]` labelled sections
- [ ] Tune `SCORE_EXIT_THRESHOLD` based on accumulated session data — check if 7.5 is calibrated correctly for `deepseek-r1:8b`
- [ ] Add a fifth solver hat for specific domains (e.g. a Devil's Advocate hat for research/writing)
- [ ] Review and refine hat personas after 50+ sessions using learnings files as feedback

---

## Memory & Learning

- [ ] Add expiry for time-sensitive questions that slipped into cache before `_is_timeless` was implemented
- [ ] Build a learning file viewer into `memory/inspect.py` — show current `.md` content per hat
- [ ] Evaluate quality of learnings after 20+ sessions — check if patterns are actionable
- [ ] Add a `--reset-learnings` flag to `main.py` for when learnings files need to be cleared

---

## Architecture

- [ ] Explore llama.cpp backend as Ollama replacement for true CPU/GPU layer parallelism on Windows
- [ ] Add `MAX_ITERATIONS = 2` support back with a smarter exit condition based on delta between iteration scores
- [ ] Consider adding a Blue Hat meta-review step that evaluates the overall reasoning process, not just individual solutions

---

## Robustness

- [ ] Add end-to-end test suite — at minimum one test per hat, one for cache hit/miss, one for decomposer
- [ ] Handle Ollama model-not-loaded gracefully — detect and surface clear error if model isn't pulled
- [ ] Add `--dry-run` flag to test pipeline without making Ollama calls

---

## Developer Experience

- [ ] Write a `SETUP.md` with step-by-step setup for Windows, Mac, and Linux
- [ ] Add `--verbose` / `--quiet` CLI flags to override `VERBOSE` in config without editing the file
- [ ] Add a `--domain` CLI flag to manually override the classifier for known query types
- [ ] Package as a proper Python module with `pyproject.toml`

---

## Done

- [x] Implemented semaphore-controlled concurrency — eliminated cascading timeouts
- [x] Added atomic question cache with embedding + reranker pipeline
- [x] Added query decomposition with three fallback parsers
- [x] Implemented model-driven per-hat learning system
- [x] Added structured telegraphic notation protocol across all pipeline stages
- [x] Added `_is_timeless` classifier — prevents time-sensitive answers from being cached
- [x] Moved learnings update to background task — user gets answer without waiting
- [x] Added `memory/inspect.py` — CLI browser for ChromaDB session store and cache
- [x] Added per-step timing breakdown at session end
- [x] Fixed reranker comparing question vs solution — now compares question vs question