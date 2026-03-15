# AI Parliament

> A local reasoning system that thinks before it answers — using De Bono's Six Thinking Hats, structured self-critique, and persistent memory to punch well above its weight class.

Built on `deepseek-r1:8b` + Ollama. Runs entirely on your machine. No API keys. No cloud.

---

## What makes this different

Most local LLM setups send your question to a model and get one answer back. AI Parliament sends your question through a structured multi-stage reasoning pipeline where different cognitive personas independently attack the problem, a critic hunts for flaws in every solution, a judge scores and ranks them, and a synthesizer weaves the best elements into a single definitive answer.

The result is an 8B parameter model that regularly produces reasoning quality that surprises you.

---

## How it works

```
Your query
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  CLASSIFIER  — detects domain, selects active hats  │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  DECOMPOSER  — breaks query into atomic questions   │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  ATOMIC CACHE  — serves cached answers instantly    │
│  (uncached questions proceed to solver hats)        │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│  WHITE   │  │   RED    │  │  YELLOW  │  │  GREEN   │
│  Facts   │  │  Instinct│  │ Optimism │  │Creativity│
│ temp 0.1 │  │ temp 0.8 │  │ temp 0.6 │  │ temp 0.9 │
└──────────┘  └──────────┘  └──────────┘  └──────────┘
    │              │              │              │
    └──────────────┴──────────────┴──────────────┘
                          │
                          ▼
            ┌─────────────────────────┐
            │  BLACK HAT  — Critic    │
            │  Finds every flaw       │
            └─────────────────────────┘
                          │
                          ▼
            ┌─────────────────────────┐
            │  BLUE HAT  — Judge      │
            │  Scores and ranks       │
            └─────────────────────────┘
                          │
                 (loop if needed)
                          │
                          ▼
            ┌─────────────────────────┐
            │  SUMMARIZER             │
            │  Synthesises top answers│
            └─────────────────────────┘
                          │
                          ▼
                   Final answer
                          │
                          ▼
         ┌────────────────────────────┐
         │  MEMORY  — ChromaDB        │
         │  Stores session + cache    │
         │  Updates learning files    │
         └────────────────────────────┘
```

---

## The Six Thinking Hats

Based on Edward de Bono's framework. Each hat is a distinct cognitive mode — not just a different personality, but a fundamentally different way of approaching a problem.

| Hat | Role | Temperature | What it does |
|-----|------|-------------|--------------|
| ⚪ White | Solver | 0.1 | Facts only. Distinguishes CONFIRMED FACT / STRONG EVIDENCE / WEAK EVIDENCE / UNKNOWN |
| 🔴 Red | Solver | 0.8 | Pure instinct. Never justifies with logic. Surfaces what data misses |
| 🟡 Yellow | Solver | 0.6 | Rigorous optimism. CERTAIN / LIKELY / POSSIBLE benefits with mechanisms |
| 🟢 Green | Solver | 0.9 | Lateral thinking. Minimum 3 genuinely different ideas. Borrows from unrelated fields |
| ⚫ Black | Critic | 0.2 | Finds every flaw. Never praises. Concise, surgical, specific |
| 🔵 Blue | Judge | 0.1 | Scores on correctness, completeness, clarity, originality. Outputs structured JSON |

---

## Unique features

### Query decomposition
Complex multi-part queries are split into atomic sub-questions before the hats see them. Each hat answers every sub-question explicitly. The critic and judge evaluate against a precise checklist rather than a vague blob.

### Atomic question cache
Solutions are stored at the sub-question level in ChromaDB. Future queries that share sub-questions with past queries skip the entire solver pipeline for those sub-questions and serve cached answers instantly. The system gets faster with every query.

### Two-stage memory retrieval
Past sessions are retrieved by embedding similarity then re-scored by `bge-reranker-v2-m3` (a cross-encoder reranker) before injection as context. Pure embedding similarity has too many false positives — the reranker cuts noise and only passes genuinely relevant past solutions.

### Model-driven learning files
After every session the model itself updates per-hat `.md` learning files by reviewing critiques and judge reasoning. It adds only genuinely new insights, consolidates similar entries, and writes in clear actionable language. Poor sessions update critique patterns (what to avoid). Good sessions additionally update quality signals (what excellence looks like). The files are injected into solver prompts on future queries — the system gets smarter with every run.

### Structured telegraphic notation protocol
All inter-hat communication uses a defined compact symbolic language — not just "be brief" instructions, but a fully specified notation that every stage understands. Solvers write in dense telegraphic form, the critic and judge evaluate and respond in the same notation, and only the summarizer expands everything into readable prose for the user.
 
```
[1] Oil↑ → prod.costs↑ → margins↓ → earnings est.↓ → sell-off;
ST: sentiment-driven // LT: structural re-pricing ∝ oil dependency.
UNKNOWN: exact elasticity. CONFIRMED: hist. correlation >0.6.
```
 
Key symbols: `→` (causes) `←→` (bidirectional) `↑↓` (change) `∴` (therefore) `∵` (because) `//` (contrast) `CONFIRMED/LIKELY/POSSIBLE/UNKNOWN/DISPUTED` (certainty) `ST/LT` (temporal) `MISSING:/WRONG:/WEAK:` (critic issue markers)
 
Reduces inter-hat token usage by an estimated 40-60% compared to prose, directly shortening think blocks and reducing the risk of JSON parse failures in the judge.

### Internal communication compression
Solver, critic, and judge communicate in dense compressed form — maximum signal, minimum tokens. The summarizer is the only user-facing stage and explicitly owns expansion into readable prose. This reduces token consumption throughout the pipeline, shortens think blocks, and lowers the risk of mid-sentence truncation.

### Semaphore-controlled concurrency
`asyncio.gather()` launches all hat tasks simultaneously but a semaphore ensures only one request hits Ollama at a time. This queues in Python rather than inside Ollama's internal queue, giving each request its full timeout budget and eliminating cascading timeout failures — while preserving the parallel task structure for future multi-GPU upgrades.

### Per-step timing breakdown
Every pipeline step is timed. A breakdown table prints at session end showing seconds and percentage per step. Data-driven optimization rather than guessing.

---

## Hardware requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM | 8GB | 12GB+ |
| RAM | 16GB | 32GB |
| GPU | Any CUDA | RTX 4060+ |

Tested on: RTX 4060 8GB, AMD Ryzen, 16GB RAM, Windows 11.

---

## Setup

**1. Install Ollama**
Download from [ollama.com](https://ollama.com) and start it:
```bash
ollama serve
```

**2. Pull the required models**
```bash
ollama pull deepseek-r1:8b
ollama pull mxbai-embed-large
ollama pull qllama/bge-reranker-v2-m3:latest
```

**3. Install Python dependencies**
```bash
pip install -r requirements.txt
```

**4. Run**
```bash
# Interactive mode
python main.py

# Single query
python main.py "Why do stock markets fall when oil prices rise?"
```

---

## Inspect your memory store

```bash
python -m memory.inspect                        # summary
python -m memory.inspect stats                  # score distributions
python -m memory.inspect sessions               # list recent sessions
python -m memory.inspect cache                  # list cached sub-questions
python -m memory.inspect search "oil prices"    # search by topic
python -m memory.inspect session a3f9b2         # full session detail
```

---

## Project structure

```
ai-parliament/
├── main.py                  # CLI entry point
├── orchestrator.py          # Pipeline controller
├── config.py                # All tuneable parameters
├── requirements.txt
├── README.md
├── OPTIMIZATIONS.md         # Full optimization log
│
├── hats/
│   ├── classifier.py        # Domain detection
│   ├── decomposer.py        # Query decomposition
│   ├── solver.py            # White/Red/Yellow/Green hat runner
│   ├── critic.py            # Black hat (critic)
│   ├── judge.py             # Blue hat (judge)
│   ├── summarizer.py        # Final synthesis
│   └── utils.py             # Shared utilities (semaphore, strip_think, build_options)
│
└── memory/
    ├── store.py             # ChromaDB session store + atomic cache
    ├── learnings.py         # Model-driven per-hat learning system
    ├── inspect.py           # CLI inspector for ChromaDB
    └── learnings/           # Per-hat .md learning files (auto-generated)
        ├── white.md
        ├── red.md
        ├── yellow.md
        ├── green.md
        ├── black.md
        └── blue.md
```

---

## Key configuration

All parameters in `config.py`. No magic numbers elsewhere.

```python
MAX_ITERATIONS         = 1      # Refinement loop iterations
SCORE_EXIT_THRESHOLD   = 7.5    # Early exit if top solutions score above this
TOP_N_SOLUTIONS        = 3      # Solutions passed to summarizer
OLLAMA_MAX_CONCURRENT  = 1      # Max concurrent Ollama requests
NUM_GPU_LAYERS         = 38     # GPU layer split (set to model layer count for full GPU)
ATOMIC_CACHE_MIN_SCORE = 7.0    # Minimum judge score to cache a solution
REQUEST_TIMEOUT        = 300    # Per-call timeout in seconds
MAX_RETRIES            = 3      # Retry attempts on timeout
```

---

## Model

`deepseek-r1:8b` — chosen for its built-in chain-of-thought reasoning (`<think>` blocks) which makes the critic and judge significantly more reliable than general-purpose models of the same size. The reasoning traces are stripped from inter-hat communication to save tokens but are used by black and blue hats where careful evaluation matters most.

---

## License

MIT