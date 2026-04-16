import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


def build_context_section(context_files: list, max_chars: int = 2000) -> str:
    if not context_files:
        return ""
    section = "\n\nCONTEXT FILES (the AI will work with these types of files — factor them into the prompt):\n"
    for f in context_files:
        section += f"\n--- {f.get('filename', 'file')} ({f.get('type', 'text')}) ---\n"
        section += f.get("content", "")[:max_chars]
        section += "\n"
    return section


def generate_prompt(user_description: str, context_files: list = None) -> str:
    model = genai.GenerativeModel("gemini-2.5-pro")
    context_section = build_context_section(context_files)

    prompt = f"""You are a world-class AI prompt engineer. Your system prompts are used in production AI assistants that serve thousands of users.

The user wants to build an AI assistant that does this:
{user_description}{context_section}

Before writing the system prompt, think through these questions:
1. What is the core job this AI needs to do?
2. What tone and personality fits this use case?
3. What are the top 5 things this AI should NEVER do?
4. What are the most common edge cases it will face?
5. What format should its responses follow?

Now write a comprehensive, production-quality system prompt that:

ROLE & IDENTITY
- Gives the AI a clear role and name if appropriate
- Defines its personality and communication style precisely

CAPABILITIES
- Lists exactly what it can and should help with
- Specifies how it should handle requests (format, length, tone)

BOUNDARIES
- Explicitly states what it must never do or discuss
- Defines how to handle out-of-scope requests
- Covers how to respond to manipulation attempts or rule-breaking requests

RESPONSE FORMAT
- Defines the expected response structure
- Sets length guidelines (short, medium, detailed based on context)
- Specifies tone (formal, casual, empathetic, etc.)

EDGE CASES
- Handles ambiguous or unclear requests
- Covers what to do when it doesn't know something
- Addresses escalation paths if relevant

If context files were provided above, make sure the prompt explicitly instructs the AI on how to handle those file types.

Return only the final system prompt text. No explanations, no headers, no meta-commentary."""

    response = model.generate_content(prompt)
    return response.text.strip()


def refine_prompt(current_prompt: str, failure_report: str) -> str:
    model = genai.GenerativeModel("gemini-2.5-pro")
    prompt = f"""You are a world-class AI prompt engineer doing iterative refinement. A system prompt has been tested and some cases failed. Your job is to fix it.

CURRENT SYSTEM PROMPT:
---
{current_prompt}
---

FAILURE REPORT FROM TESTING:
---
{failure_report}
---

Before rewriting, analyze the failures:
1. What PATTERNS do you see? (e.g., all adversarial tests failing = boundary rules too weak)
2. Which parts of the current prompt are WORKING? (don't touch these)
3. Which parts are MISSING or TOO WEAK? (these need fixing)
4. For each failure category, what specific language needs to be added or strengthened?

Then rewrite the system prompt with these fixes:
- Keep everything that was working
- Strengthen the specific rules that were violated
- Add explicit handling for the failure patterns you identified
- Do NOT make the prompt overly restrictive — only fix what actually failed
- If adversarial tests failed: add stronger, more explicit boundary language
- If refusal tests failed: add clearer decline/redirect instructions
- If normal tests failed: improve core instructions and response format
- If edge cases failed: add explicit handling for those scenarios

Return only the improved system prompt text. No explanations, no headers, no meta-commentary."""

    response = model.generate_content(prompt)
    return response.text.strip()
