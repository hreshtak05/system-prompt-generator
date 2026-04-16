import asyncio
from core.prompt_generator import generate_prompt, refine_prompt
from core.eval_generator import generate_eval_cases
from core.eval_runner import run_evals

MAX_ITERATIONS = 50
PASS_THRESHOLD = 0.99
NO_IMPROVEMENT_LIMIT = 7


async def run_loop(user_description: str, custom_cases: list = None, existing_prompt: str = None, context_files: list = None):
    # Step 1: Generate eval test cases once — reused every iteration
    eval_cases = generate_eval_cases(user_description, custom_cases, context_files)

    # Step 2: Use existing prompt if provided, otherwise generate a new one
    if existing_prompt and existing_prompt.strip():
        current_prompt = existing_prompt.strip()
    else:
        current_prompt = generate_prompt(user_description, context_files)

    best_prompt = current_prompt
    best_pass_rate = 0.0
    no_improvement_count = 0

    # Track which tests have persistently failed across iterations
    persistent_failures = {}  # input -> fail count

    for iteration in range(1, MAX_ITERATIONS + 1):

        # Step 3a: Run all evals in parallel
        eval_results = await run_evals(current_prompt, eval_cases)
        pass_rate = eval_results["pass_rate"]

        # Track best prompt and improvement streak
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

        # Step 3b: Stream iteration update to frontend
        yield {
            "type": "iteration",
            "iteration": iteration,
            "max_iterations": MAX_ITERATIONS,
            "pass_rate": pass_rate,
            "results": eval_results["results"]
        }

        # Step 3c: Success — hit the threshold
        if pass_rate >= PASS_THRESHOLD:
            yield {
                "type": "done",
                "status": "success",
                "prompt": current_prompt,
                "pass_rate": pass_rate
            }
            return

        # Step 3d: Early stop — genuinely stuck
        if no_improvement_count >= NO_IMPROVEMENT_LIMIT:
            yield {
                "type": "done",
                "status": "stuck",
                "prompt": best_prompt,
                "pass_rate": best_pass_rate
            }
            return

        # Step 3e: Build rich failure report for the refiner
        failed_results = [r for r in eval_results["results"] if not r["passed"]]

        # Group failures by category
        by_category = {}
        for r in failed_results:
            cat = r.get("category", "normal")
            by_category.setdefault(cat, []).append(r)

        # Find tests that have failed repeatedly (3+ times)
        chronic_failures = [
            inp for inp, count in persistent_failures.items()
            if count >= 3
        ]

        failure_report = f"EVALUATION FAILURE REPORT — Iteration {iteration}\n"
        failure_report += f"Score: {eval_results['passed']}/{eval_results['total']} passed ({int(pass_rate * 100)}%)\n"
        failure_report += f"Trend: {'improving' if no_improvement_count == 0 else f'no improvement for {no_improvement_count} round(s)'}\n\n"

        # Highlight chronic failures
        if chronic_failures:
            failure_report += "CHRONIC FAILURES (failed 3+ times — highest priority to fix):\n"
            for inp in chronic_failures:
                failure_report += f"  - \"{inp}\"\n"
            failure_report += "\n"

        # Category breakdown
        if by_category:
            failure_report += "FAILURES BY CATEGORY:\n"
            for cat, failures in by_category.items():
                failure_report += f"  - {cat.upper()}: {len(failures)} failed\n"
            failure_report += "\n"

        # Detailed failures — include the actual AI response so refiner can see what went wrong
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

        current_prompt = refine_prompt(current_prompt, failure_report)

    # Step 4: Reached max iterations
    yield {
        "type": "done",
        "status": "max_iterations",
        "prompt": best_prompt,
        "pass_rate": best_pass_rate
    }
