import asyncio
from core.prompt_generator import generate_prompt, refine_prompt
from core.eval_generator import generate_eval_cases
from core.eval_runner import run_evals

MAX_ITERATIONS = 50
PASS_THRESHOLD = 0.90       # Keep going until 90%+ of tests pass
NO_IMPROVEMENT_LIMIT = 7   # Only stop early if stuck for 7 rounds in a row


async def run_loop(user_description: str, custom_cases: list = None, existing_prompt: str = None):
    # Step 1: Generate eval test cases once (reused every loop)
    eval_cases = generate_eval_cases(user_description, custom_cases)

    # Step 2: Use existing prompt if provided, otherwise generate a new one
    if existing_prompt and existing_prompt.strip():
        current_prompt = existing_prompt.strip()
    else:
        current_prompt = generate_prompt(user_description)

    best_prompt = current_prompt
    best_pass_rate = 0.0
    no_improvement_count = 0

    for iteration in range(1, MAX_ITERATIONS + 1):
        # Step 3a: Run evals on the current system prompt
        eval_results = await run_evals(current_prompt, eval_cases)
        pass_rate = eval_results["pass_rate"]

        # Track best and count stale rounds
        if pass_rate > best_pass_rate:
            best_pass_rate = pass_rate
            best_prompt = current_prompt
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        # Step 3b: Yield a status update
        yield {
            "type": "iteration",
            "iteration": iteration,
            "max_iterations": MAX_ITERATIONS,
            "pass_rate": pass_rate,
            "results": eval_results["results"]
        }

        # Step 3c: Check if we hit 100%
        if pass_rate >= PASS_THRESHOLD:
            yield {
                "type": "done",
                "status": "success",
                "prompt": current_prompt,
                "pass_rate": pass_rate
            }
            return

        # Step 3d: Early stop if stuck — no improvement for N rounds
        if no_improvement_count >= NO_IMPROVEMENT_LIMIT:
            yield {
                "type": "done",
                "status": "stuck",
                "prompt": best_prompt,
                "pass_rate": best_pass_rate
            }
            return

        # Step 3e: Build a rich failure report grouped by category
        failed_results = [r for r in eval_results["results"] if not r["passed"]]

        # Group failures by category
        by_category = {}
        for r in failed_results:
            cat = r.get("category", "normal")
            by_category.setdefault(cat, []).append(r)

        failure_report = f"EVALUATION FAILURE REPORT — Iteration {iteration}\n"
        failure_report += f"Score: {eval_results['passed']}/{eval_results['total']} passed ({int(pass_rate * 100)}%)\n\n"

        # Pattern summary
        if by_category:
            failure_report += "FAILURE PATTERNS BY CATEGORY:\n"
            for cat, failures in by_category.items():
                failure_report += f"  - {cat.upper()}: {len(failures)} test(s) failed\n"
            failure_report += "\n"

        # Detailed failures
        failure_report += "DETAILED FAILURES:\n\n"
        for result in failed_results:
            failure_report += f"[{result.get('category', 'normal').upper()}]\n"
            failure_report += f"Input: {result['input']}\n"
            failure_report += f"Criteria: {result['criteria']}\n"
            failure_report += f"Score: {result.get('score', 'N/A')}/10"

            scores = result.get("scores", {})
            if scores:
                failure_report += (
                    f" (instruction_following: {scores.get('instruction_following', '?')}, "
                    f"criteria_met: {scores.get('criteria_met', '?')}, "
                    f"quality: {scores.get('quality', '?')})"
                )

            failure_report += f"\nWhat went wrong: {result['reason']}\n\n"

        current_prompt = refine_prompt(current_prompt, failure_report)

    # Step 4: Hit max iterations
    yield {
        "type": "done",
        "status": "max_iterations",
        "prompt": best_prompt,
        "pass_rate": best_pass_rate
    }
