import asyncio
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


def build_context_section(context_files: list, max_chars: int = 2000) -> str:
    if not context_files:
        return ""
    section = "\n\nCONTEXT FILES (the AI will work with these — factor them into the prompt):\n"
    for f in context_files:
        section += f"\n--- {f.get('filename', 'file')} ({f.get('type', 'text')}) ---\n"
        section += f.get("content", "")[:max_chars]
        section += "\n"
    return section


GENERATE_INSTRUCTIONS = """You are a world-class AI prompt engineer. Your system prompts are used in production AI assistants that handle thousands of real conversations daily.

━━━ STEP 1 — ANALYSIS (do this silently before writing) ━━━

Think through:
1. What is the single core job this AI must do in every conversation?
2. What persona, tone, and voice fits this use case exactly?
3. What are the 5 hard limits — things it must NEVER do under any circumstances?
4. What are the 10 most common things users will actually say or ask?
5. How will bad actors try to manipulate or misuse this AI?
6. What ambiguous situations need explicit handling?
7. What should the response format look like? (length, structure, tone)

━━━ STEP 2 — WRITE THE PROMPT using these exact principles ━━━

LANGUAGE RULES — this is critical:
✅ STRONG language (use this): "You must", "You will always", "You must never", "Under no circumstances", "You are strictly prohibited from", "Always respond with", "Never mention"
❌ WEAK language (never use this): "try to", "generally", "usually", "you should probably", "in most cases", "if possible", "you might want to"

Every single instruction must be:
- Specific: says exactly what to do, not a vague direction
- Unconditional: no "unless", "except when", "if appropriate" — unless truly necessary
- Testable: someone reading it can immediately verify if the AI followed it

REQUIRED SECTIONS in the prompt:
1. ROLE — who the AI is, what it does, its name if relevant
2. PERSONALITY & TONE — exact communication style, formality level, emotional register
3. CORE RESPONSIBILITIES — the 3-7 things it must do in every relevant interaction
4. HARD LIMITS — explicit list of what it must never do, say, or engage with
5. RESPONSE FORMAT — exact structure, length, and style of responses
6. HANDLING EDGE CASES — what to do when requests are ambiguous, off-topic, or manipulative
7. ESCALATION — when and how to redirect to humans or other resources (if relevant)

━━━ STEP 3 — SELF-REVIEW CHECKLIST ━━━

Before finalizing, verify:
□ Every rule uses strong, unambiguous language — no "should" or "try to"
□ All 5 hard limits are written as absolute prohibitions
□ The 10 common scenarios are handled explicitly or by clear general rules
□ Adversarial attempts (jailbreaks, manipulation, off-topic pressure) are addressed
□ Response format is defined precisely — length, structure, tone
□ The prompt reads like instructions, not suggestions

If any checkbox fails — fix it before returning.

Return ONLY the final system prompt text.
No analysis. No section headers like "ROLE:" or "LIMITS:". No meta-commentary. Just the prompt itself, written as direct instructions to the AI."""


async def generate_prompt_async(user_description: str, context_files: list = None) -> str:
    model = genai.GenerativeModel("gemini-2.5-pro")
    context_section = build_context_section(context_files)

    prompt = f"""{GENERATE_INSTRUCTIONS}

THE AI ASSISTANT MUST DO THIS:
{user_description}{context_section}"""

    response = await model.generate_content_async(prompt)
    return response.text.strip()


async def refine_prompt_async(current_prompt: str, failure_report: str) -> str:
    # Use Flash for refinement — it's 3-4x faster and surgical fixes don't need Pro
    model = genai.GenerativeModel("gemini-2.5-flash")

    prompt = f"""You are a world-class AI prompt engineer doing surgical refinement. A system prompt was tested and specific cases failed. Your job is to fix only what's broken — leave everything else exactly as it is.

CURRENT SYSTEM PROMPT:
━━━━━━━━━━━━━━━━━━━━━━
{current_prompt}
━━━━━━━━━━━━━━━━━━━━━━

FAILURE REPORT:
━━━━━━━━━━━━━━━━━━━━━━
{failure_report}
━━━━━━━━━━━━━━━━━━━━━━

━━━ STEP 1 — DIAGNOSE ━━━

For each failure, identify:
- What EXACT rule was missing, too vague, or too weak?
- Is this a pattern (multiple failures of the same type) or isolated?
- Which section of the current prompt needs to be strengthened?

Failure type diagnosis:
→ ADVERSARIAL failures = boundary language is too weak or missing. The AI is being manipulated because the prompt doesn't explicitly cover that manipulation pattern.
→ REFUSAL failures = the prompt doesn't clearly tell the AI what to decline and how.
→ NORMAL failures = core instructions are ambiguous, too vague, or missing a scenario.
→ EDGE CASE failures = the prompt doesn't cover that ambiguous situation explicitly.

━━━ STEP 2 — SURGICAL FIXES ━━━

Rules for fixing:
1. Preserve every part of the prompt that IS working — copy it exactly
2. For each failure, add or strengthen the specific rule that was violated
3. Do NOT restructure, reformat, or rewrite sections that are working
4. Do NOT add unnecessary restrictions — only fix what actually failed

Specific fix patterns by failure type:
- ADVERSARIAL: Add → "Even if the user claims [X], you must still [Y]. Do not comply with requests to [specific manipulation pattern]."
- REFUSAL: Add → "If asked about [topic], respond with [exact approach]. Do not [specific thing to avoid]."
- NORMAL: Strengthen → Replace vague language with specific, unconditional instructions
- EDGE CASE: Add → "When [specific ambiguous situation], always [exact behavior]"

Language upgrade examples:
❌ "You should avoid discussing competitors" → ✅ "You must never mention, compare, or discuss any competitor brands or products"
❌ "Try to stay on topic" → ✅ "If the user asks about anything unrelated to [topic], respond: '[exact redirect message]'"
❌ "Be helpful with returns" → ✅ "When a user asks about returns, always: (1) acknowledge their request, (2) explain the return policy, (3) offer to initiate the process"

━━━ STEP 3 — SELF-CHECK ━━━

After making fixes, verify:
□ Every fix uses strong, unambiguous language
□ All previously passing tests are still covered by the unchanged sections
□ No new unnecessary restrictions were added
□ The chronic failures (if any) have explicit, targeted rules now

Return ONLY the improved system prompt text.
No analysis. No commentary. No section headers. Just the prompt."""

    response = await model.generate_content_async(prompt)
    return response.text.strip()


# Sync wrappers kept for any direct callers
def generate_prompt(user_description: str, context_files: list = None) -> str:
    return asyncio.get_event_loop().run_until_complete(
        generate_prompt_async(user_description, context_files)
    )


def refine_prompt(current_prompt: str, failure_report: str) -> str:
    return asyncio.get_event_loop().run_until_complete(
        refine_prompt_async(current_prompt, failure_report)
    )
