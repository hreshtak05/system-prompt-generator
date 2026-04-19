import asyncio
from core.prompt_generator import generate_prompt_async, refine_prompt_async
from core.eval_generator import generate_eval_cases_async
from core.eval_runner import run_evals

MAX_ITERATIONS = 50
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
        yield {"type": "status", "message": "Generating 30 test cases..."}
        eval_cases = await generate_eval_cases_async(user_description, custom_cases, context_files)
        current_prompt = existing_prompt.strip()
    else:
        # Need both — run them at the same time
        yield {"type": "status", "message": "Generating test cases and system prompt in parallel..."}
        eval_cases, current_prompt = await asyncio.gather(
            generate_eval_cases_async(user_description, custom_cases, context_files),
            generate_prompt_async(user_description, context_files)
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
                current_prompt = await generate_prompt_async(user_description, context_files)
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

        # Refine with Flash (faster, sufficient for surgical fixes)
        current_prompt = await refine_prompt_async(current_prompt, failure_report)

    # Reached max iterations
    yield {
        "type": "done",
        "status": "max_iterations",
        "prompt": best_prompt,
        "pass_rate": best_pass_rate
    }
