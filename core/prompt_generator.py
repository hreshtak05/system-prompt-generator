import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


def generate_prompt(user_description: str) -> str:
    model = genai.GenerativeModel("gemini-2.5-pro")
    prompt = f"""You are an expert at writing system prompts for AI assistants.

The user wants an AI assistant that does this:
{user_description}

Write a detailed, clear system prompt that the AI should follow.
The system prompt should:
- Define the AI's role and personality
- Specify exactly what it should and should not do
- Define the format and tone of its responses
- Handle edge cases

Return only the system prompt text. No explanations. No extra text."""

    response = model.generate_content(prompt)
    return response.text.strip()


def refine_prompt(current_prompt: str, failure_report: str) -> str:
    model = genai.GenerativeModel("gemini-2.5-pro")
    prompt = f"""You are an expert at improving AI system prompts.

Here is the current system prompt:
---
{current_prompt}
---

Here is a report of what went wrong when this prompt was tested:
---
{failure_report}
---

Rewrite the system prompt to fix all the issues mentioned in the report.
Keep what was working. Only fix what was failing.

Return only the improved system prompt text. No explanations. No extra text."""

    response = model.generate_content(prompt)
    return response.text.strip()
