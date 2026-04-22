import asyncio
from core.prompt_generator import generate_prompt_async, refine_prompt_async
from core.eval_generator import generate_eval_cases_async
from core.eval_runner import run_evals

MAX_ITERATIONS = 20
PASS_THRESHOLD = 0.95
NO_IMPROVEMENT_LIMIT = 7
MAX_FRESH_STARTS = 2


async def run_loop(user_description: str, custom_cases: list = None, existing_prompt: str = None, context_files: list = None):

    # Step 1 — Tell user we're starting
    yield {"type": "status", "message": "Analyzing your description and preparing..."}

    # Step 2 — Run eval generation and prompt generation IN PARALLEL
    # (saves ~15s vs running them sequentially)
    if existing_prompt and existing_prompt.strip():
        # Existing prompt provided — only need to generate eval cases
        yield {"type": "status", "message": "Generating 15 test cases..."}
        eval_cases = await asyncio.wait_for(
            generate_eval_cases_async(user_description, custom_cases, context_files),
            timeout=180.0
        )
        current_prompt = existing_prompt.strip()
    else:
        # Need both — run them at the same time, 3-minute timeout each
        yield {"type": "status", "message": "Generating test cases and system prompt in parallel..."}
        eval_cases, current_prompt = await asyncio.gather(
            asyncio.wait_for(generate_eval_cases_async(user_description, custom_cases, context_files), timeout=180.0),
            asyncio.wait_for(generate_prompt_async(user_description, context_files), timeout=300.0)
        )

    yield {"type": "status", "message": "Ready — running first evaluation round..."}

    best_prompt = current_prompt
    best_pass_rate = 0.0
    no_improvement_count = 0
    fresh_starts = 0
    persistent_failures = {}

    for iteration in range(1, MAX_ITERATIONS + 1):

        # Run all evals in parallel
        eval_results = await run_evals(current_prompt, eval_cases)
        pass_rate = eval_results["pass_rate"]

        # Track best prompt
        if pass_rate > best_pass_rate:
            best_pass_rate = pass_rate
            best_prompt = current_prompt
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        # Track persistent failures
        for result in eval_results["results"]:
            if not result["passed"]:
                key = result["input"]
                persistent_failures[key] = persistent_failures.get(key, 0) + 1

        # Stream iteration update to frontend
        yield {
            "type": "iteration",
            "iteration": iteration,
            "max_iterations": MAX_ITERATIONS,
            "pass_rate": pass_rate,
            "results": eval_results["results"]
        }

        # Success
        if pass_rate >= PASS_THRESHOLD:
            yield {
                "type": "done",
                "status": "success",
                "prompt": current_prompt,
                "pass_rate": pass_rate
            }
            return

        # Stuck — regenerate from scratch instead of giving up
        if no_improvement_count >= NO_IMPROVEMENT_LIMIT:
            if fresh_starts < MAX_FRESH_STARTS:
                fresh_starts += 1
                no_improvement_count = 0
                persistent_failures = {}
                yield {
                    "type": "status",
                    "message": f"Stuck at {int(best_pass_rate * 100)}% — regenerating prompt from scratch (attempt {fresh_starts}/{MAX_FRESH_STARTS})"
                }
                current_prompt = await asyncio.wait_for(
                    generate_prompt_async(user_description, context_files),
                    timeout=300.0
                )
                continue
            else:
                yield {
                    "type": "done",
                    "status": "stuck",
                    "prompt": best_prompt,
                    "pass_rate": best_pass_rate
                }
                return

        # Build failure report
        failed_results = [r for r in eval_results["results"] if not r["passed"]]

        by_category = {}
        for r in failed_results:
            cat = r.get("category", "normal")
            by_category.setdefault(cat, []).append(r)

        chronic_failures = [
            inp for inp, count in persistent_failures.items()
            if count >= 3
        ]

        failure_report = f"EVALUATION FAILURE REPORT — Iteration {iteration}\n"
        failure_report += f"Score: {eval_results['passed']}/{eval_results['total']} passed ({int(pass_rate * 100)}%)\n"
        failure_report += f"Trend: {'improving' if no_improvement_count == 0 else f'no improvement for {no_improvement_count} round(s)'}\n\n"

        if chronic_failures:
            failure_report += "CHRONIC FAILURES (failed 3+ times — highest priority to fix):\n"
            for inp in chronic_failures:
                failure_report += f"  - \"{inp}\"\n"
            failure_report += "\n"

        if by_category:
            failure_report += "FAILURES BY CATEGORY:\n"
            for cat, failures in by_category.items():
                failure_report += f"  - {cat.upper()}: {len(failures)} failed\n"
            failure_report += "\n"

        failure_report += "DETAILED FAILURES:\n\n"
        for result in failed_results:
            failure_report += f"[{result.get('category', 'normal').upper()}]\n"
            failure_report += f"Input: {result['input']}\n"
            failure_report += f"Criteria: {result['criteria']}\n"
            failure_report += f"AI responded with: \"{result.get('response', 'N/A')}\"\n"
            failure_report += f"Score: {result.get('score', 'N/A')}/10"

            scores = result.get("scores", {})
            if scores:
                failure_report += (
                    f" (instruction_following: {scores.get('instruction_following', '?')}, "
                    f"criteria_met: {scores.get('criteria_met', '?')}, "
                    f"quality: {scores.get('quality', '?')})"
                )

            failure_report += f"\nWhy it failed: {result['reason']}\n\n"

        # Refine with Flash — hard 3-minute timeout so a hung API call never freezes the loop
        try:
            current_prompt = await asyncio.wait_for(
                refine_prompt_async(current_prompt, failure_report),
                timeout=180.0
            )
        except asyncio.TimeoutError:
            yield {"type": "status", "message": f"Refinement timed out on round {iteration} — retrying with best prompt so far"}
            current_prompt = best_prompt

    # Reached max iterations
    yield {
        "type": "done",
        "status": "max_iterations",
        "prompt": best_prompt,
        "pass_rate": best_pass_rate
    }


async def run_test(user_description: str, existing_prompt: str, custom_cases: list = None, context_files: list = None):
    """Run evals exactly once on an existing prompt — no loop, no refinement.
    Shows what passes and what fails, then signals whether improvement is needed."""

    yield {"type": "status", "message": "Generating test cases..."}

    eval_cases = await asyncio.wait_for(
        generate_eval_cases_async(user_description, custom_cases, context_files),
        timeout=180.0
    )

    yield {"type": "status", "message": f"Running {len(eval_cases)} test cases against your prompt..."}

    eval_results = await run_evals(existing_prompt, eval_cases)
    pass_rate = eval_results["pass_rate"]

    # Stream the test results exactly like a normal iteration
    yield {
        "type": "iteration",
        "iteration": 1,
        "max_iterations": 1,
        "pass_rate": pass_rate,
        "results": eval_results["results"]
    }

    # Done — tell frontend whether improvement is needed
    needs_improvement = pass_rate < PASS_THRESHOLD
    yield {
        "type": "test_done",
        "pass_rate": pass_rate,
        "passed": eval_results["passed"],
        "total": eval_results["total"],
        "needs_improvement": needs_improvement,
        "prompt": existing_prompt
    }
