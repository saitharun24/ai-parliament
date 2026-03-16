# =============================================================================
# orchestrator.py — Main pipeline controller (De Bono edition)
# =============================================================================
# Pipeline order:
#   1. Classify domain → select active hats
#   2. Retrieve + rerank memory hints
#   3. Decompose query into atomic sub-questions
#   4. Solver hats (parallel, domain-filtered)
#   5. Refinement loop: Black Hat → Blue Hat → fix pass (repeat up to MAX_ITERATIONS)
#   6. Summarise top-N solutions
#   7. Store session in memory

from datetime import datetime
import asyncio

from config import MAX_ITERATIONS, VERBOSE, TOP_N_SOLUTIONS
from hats.classifier  import classify_query
from hats.decomposer  import decompose, format_sub_questions
from hats.solver      import run_solvers
from hats.critic      import run_critic
from hats.judge       import run_judge
from hats.summarizer  import run_summarizer
from memory.learnings import get_learnings_summary, update_learnings_from_session
from memory.store     import (
    retrieve_hints, format_hints_for_context, store_result,
    retrieve_cached_solutions, store_atomic_solutions,
)


def _header(text: str):
    print(f"\n{'='*60}\n  {text}\n{'='*60}")

def _section(text: str):
    print(f"\n{'─'*50}\n  {text}\n{'─'*50}")

def _tick() -> datetime:
    return datetime.now()

def _tock(label: str, t: datetime):
    elapsed = (datetime.now() - t).total_seconds()
    print(f"  ⏱  {label}: {elapsed:.1f}s")
    return elapsed


async def run(query: str) -> str:
    """Full De Bono reasoning pipeline for a single query."""
    start = datetime.now()
    timings: dict[str, float] = {}

    _header(f"REASONER  {start.strftime('%H:%M:%S')}")
    print(f"  Query: {query[:80]}{'...' if len(query) > 80 else ''}")

    # ------------------------------------------------------------------
    # Step 1: Classify domain → select active hats
    # ------------------------------------------------------------------
    _section("Step 1 — Domain classification")
    t = _tick()

    domain, active_hats = await classify_query(query)

    timings["classifier"] = _tock("classifier", t)

    # ------------------------------------------------------------------
    # Step 2: Retrieve + rerank memory hints
    # ------------------------------------------------------------------
    _section("Step 2 — Memory retrieval")
    t = _tick()
    hints = await retrieve_hints(query)
    timings["memory_retrieval"] = _tock("memory retrieval", t)
    if hints:
        print(f"  {len(hints)} relevant past session(s) retrieved and reranked")
    else:
        print("  No relevant past sessions — starting fresh")
    memory_context = format_hints_for_context(hints)

    # ------------------------------------------------------------------
    # Step 3: Decompose query into atomic sub-questions
    # ------------------------------------------------------------------
    _section("Step 3 — Query decomposition")
    t = _tick()
    sub_questions    = await decompose(query)
    structured_query = format_sub_questions(sub_questions)
    timings["decomposer"] = _tock("decomposer", t)

    # ------------------------------------------------------------------
    # Step 4: Check atomic cache for each sub-question
    # Sub-questions with cached solutions skip the hat pipeline entirely.
    # Only uncached sub-questions go through solvers → critic → judge.
    # ------------------------------------------------------------------
    _section("Step 4 — Atomic cache lookup")
    t = _tick()
    cache_results = await retrieve_cached_solutions(sub_questions)

    cached_questions = {q: s for q, s in cache_results.items() if s is not None}
    uncached_questions = [q for q, s in cache_results.items() if s is None]

    timings["cache_lookup"] = _tock("cache lookup", t)

    if cached_questions:
        print(f"  {len(cached_questions)}/{len(sub_questions)} sub-question(s) served from cache")
    if uncached_questions:
        print(f"  {len(uncached_questions)}/{len(sub_questions)} sub-question(s) need full pipeline")

    # If ALL sub-questions are cached, skip to summarizer directly
    if not uncached_questions:
        _section("Step 5 — All sub-questions cached — skipping to summarizer")
        cached_solutions = [
            {"hat": "cache", "content": content, "score": 10.0}
            for content in cached_questions.values()
        ]
        t = _tick()
        final_answer = await run_summarizer(structured_query, cached_solutions)
        timings["summarizer"] = _tock("summarizer", t)
        _section("Step 6 — Storing session in memory")
        await store_result(
            query=query,
            solutions=cached_solutions,
            iteration_count=0,
        )
        elapsed = (datetime.now() - start).total_seconds()
        _header(f"DONE (fully cached) — {elapsed:.1f}s  |  domain: {domain}  |  sub-questions: {len(sub_questions)}")
        return final_answer

    # Build structured query from uncached questions only
    partial_query = format_sub_questions(uncached_questions)

    # ------------------------------------------------------------------
    # Step 5: Initial solver pass (parallel, domain-filtered hats only)
    # partial_query flows through the entire pipeline from here —
    # every hat, critic, judge, and summarizer works against the
    # decomposed checklist, not the raw query.
    # ------------------------------------------------------------------
    _section(f"Step 5 — Solver hats: {active_hats}")
    t = _tick()
    solutions = await run_solvers(
        query=partial_query,
        active_hats=active_hats,
        domain=domain,
        memory_context=memory_context,
    )
    timings["solvers"] = _tock("solvers", t)
    previous_solutions = {s["hat"]: s["content"] for s in solutions}

    # ------------------------------------------------------------------
    # Refinement loop — operates on uncached sub-questions only
    # ------------------------------------------------------------------
    final_judge_result = None
    last_critiqued = []
    iteration = 0

    for iteration in range(1, MAX_ITERATIONS + 1):
        _section(f"Iteration {iteration}/{MAX_ITERATIONS}")

        # Black Hat: critique all solutions in parallel
        _section("  Black Hat (Critic)")
        t = _tick()
        critiqued, all_clean = await run_critic(partial_query, solutions)
        last_critiqued = critiqued
        timings[f"critic_iter{iteration}"] = _tock(f"critic iter {iteration}", t)

        # Skip fix pass entirely if no issues found
        if all_clean:
            _section("  Fix pass SKIPPED — no issues found")
            break

        # Blue Hat: judge and score
        _section("  Blue Hat (Judge)")
        t = _tick()
        judge_result       = await run_judge(partial_query, critiqued)
        final_judge_result = judge_result
        timings[f"judge_iter{iteration}"] = _tock(f"judge iter {iteration}", t)

        # If judge found things critic missed → re-check critic
        if judge_result["judge_findings"]:
            _section("  Black Hat re-check (Blue Hat found missed issues)")
            solutions_to_recheck = [
                s for s in critiqued
                if s["hat"] in judge_result["judge_findings"]
            ]
            t = _tick()
            rechecked, _ = await run_critic(
                structured_query,
                solutions_to_recheck,
                judge_findings=judge_result["judge_findings"],
            )
            timings[f"critic_recheck_iter{iteration}"] = _tock(f"critic recheck iter {iteration}", t)
            recheck_map        = {s["hat"]: s for s in rechecked}
            critiqued          = [recheck_map.get(s["hat"], s) for s in critiqued]
            t = _tick()
            judge_result       = await run_judge(partial_query, critiqued)
            final_judge_result = judge_result
            timings[f"judge_recheck_iter{iteration}"] = _tock(f"judge recheck iter {iteration}", t)

        # Early exit if score threshold met
        if judge_result["should_exit"]:
            _section(f"  Early exit — threshold met at iteration {iteration}")
            break

        # Fix pass — only if issues were found AND not last iteration
        if not all_clean and iteration < MAX_ITERATIONS:
            _section("  Fix pass — solvers revise (parallel)")
            critiques_for_fix = {
                s["hat"]: s.get("critique", "") for s in critiqued
                if not s.get("critique_clean", False)
            }
            t = _tick()
            solutions = await run_solvers(
                query=partial_query,
                active_hats=active_hats,
                domain=domain,
                memory_context=memory_context,
                critiques=critiques_for_fix,
                previous_solutions=previous_solutions,
            )
            timings[f"fix_pass_iter{iteration}"] = _tock(f"fix pass iter {iteration}", t)
            previous_solutions = {s["hat"]: s["content"] for s in solutions}

    if final_judge_result is None:
        _header("ERROR — no judge result produced. Check MAX_ITERATIONS in config.")
        return ""

    # ------------------------------------------------------------------
    # Step 6: Merge cached solutions with new solutions, then summarise
    # Cached solutions are prepended so the summarizer has full context.
    # ------------------------------------------------------------------
    _section(f"Step 6 — Summarising top {TOP_N_SOLUTIONS} solutions")
    top_solutions = final_judge_result["top"]

    # Prepend cached solutions so summarizer sees the complete picture
    if cached_questions:
        cached_as_solutions = [
            {"hat": "cache", "content": content, "score": 10.0}
            for content in cached_questions.values()
        ]
        top_solutions = cached_as_solutions + top_solutions

    t = _tick()
    final_answer  = await run_summarizer(structured_query, top_solutions)
    timings["summarizer"] = _tock("summarizer", t)

    # ------------------------------------------------------------------
    # Step 7: Store in memory
    # Original query stored — not structured_query — so future semantic
    # retrieval matches by intent, not by decomposed prompt format.
    # ------------------------------------------------------------------
    _section("Step 7 — Storing session in memory")
    t = _tick()
    ranked: list[dict] = final_judge_result["ranked"]
    await store_result(
        query=query,
        solutions=final_judge_result["ranked"],
        iteration_count=iteration,
    )

    # Store each uncached sub-question with its best solution for future cache hits
    await store_atomic_solutions(
        sub_questions=uncached_questions,
        solutions=final_judge_result["ranked"],
    )
    timings["storage"] = _tock("storage", t)
    print("  Stored in ChromaDB")

    # Update learnings from this session if it was a good one
    top_score = ranked[0]["score"] if ranked else 0.0
    t = _tick()
    asyncio.ensure_future(update_learnings_from_session(
        domain=domain,
        solutions=ranked,
        critiqued_solutions=last_critiqued,
        top_score=top_score,
    ))
    timings["learnings"] = 0.0
    timings["learnings"] = _tock("learnings update", t)

    elapsed = (datetime.now() - start).total_seconds()
    _print_timings(timings, elapsed)

    if VERBOSE:
        print(f"\n  LEARNINGS STATUS:\n{get_learnings_summary()}")

    _header(
        f"DONE — {elapsed:.1f}s  |  domain: {domain}  |  cached: {len(cached_questions)}/{len(sub_questions)}  |"
        f"  sub-questions: {len(sub_questions)}  |  hats: {active_hats}"
    )

    return final_answer

def _print_timings(timings: dict[str, float], total: float):
    """Print a breakdown of time spent in each pipeline step."""
    print(f"\n{'─'*50}")
    print(f"  TIMING BREAKDOWN")
    print(f"{'─'*50}")
    for step, secs in sorted(timings.items(), key=lambda x: -x[1]):
        pct = (secs / total * 100) if total > 0 else 0
        bar = '█' * int(pct / 5)
        print(f"  {step:<30} {secs:>7.1f}s  {pct:>5.1f}%  {bar}")
    print(f"  {'─'*46}")
    print(f"  {'TOTAL':<30} {total:>7.1f}s  100.0%")
    print(f"{'─'*50}")