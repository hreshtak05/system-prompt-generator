# System Prompt Generator

An AI-powered tool that writes, tests, and automatically improves system prompts for AI assistants — until they reach a 95%+ pass rate across adversarial, edge case, refusal, and normal test scenarios.

**Live app:** https://system-prompt-generator-u68e.onrender.com

---

## What it does

You describe what you want your AI assistant to do. The tool:

1. Generates 15 realistic test cases (normal, edge, adversarial, refusal)
2. Runs your system prompt against all 15 cases in parallel
3. Scores each response across 4 dimensions (prompt coverage, instruction following, criteria met, quality)
4. Automatically refines the prompt based on failures
5. Repeats until 95%+ pass rate is achieved (up to 20 rounds)

---

## Modes

### ✨ Generate new prompt
Describe what your AI should do → get a production-ready system prompt.

### 🔧 Improve existing prompt
Paste your existing prompt → the tool tests it and improves it in a loop until it scores 95%+.

### 🧪 Test then improve
Paste your prompt → run one round of evals to see exactly what passes and fails → optionally start the full improvement loop.

### ▶ Run prompt
Paste your prompt + upload files (images, PDFs, text) → see the actual AI response. Use this to verify your prompt works on real content (e.g. run a homework-checking prompt on a real homework image).

---

## How the scoring works

Each test case is judged on 4 dimensions:

| Dimension | Weight | What it checks |
|-----------|--------|----------------|
| `prompt_coverage` | 35% | Does the system prompt **explicitly define rules** for this scenario? Vague prompts ("be helpful") score 1–3 and auto-fail. |
| `instruction_following` | 30% | Did the AI follow the explicit rules that exist in the prompt? |
| `criteria_met` | 25% | Did the response satisfy the specific test criteria? |
| `quality` | 10% | Was the response well-written and appropriately formatted? |

**Pass condition:** overall score ≥ 7 AND prompt_coverage ≥ 5

This means a smart AI giving a good response by accident will still **fail** if the system prompt doesn't explicitly define that behavior.

---

## File upload support

- **Images** (PNG, JPG, WebP) — sent directly as vision input
- **PDFs & documents** — extracted via Gemini Files API
- **Text files** (TXT, MD, CSV, JSON, code) — read as plain text

Upload context files to generate more specific, realistic test cases based on your actual content.

---

## Running locally

**Requirements:** Python 3.11+, a Gemini API key

```bash
# Clone the repo
git clone https://github.com/hreshtak05/system-prompt-generator.git
cd system-prompt-generator

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Add your Gemini API key
echo "GEMINI_API_KEY=your_key_here" > .env

# Run
uvicorn app:app --reload --port 8000
```

Open http://localhost:8000

---

## Project structure

```
├── app.py                  # FastAPI server — endpoints: /generate, /test, /run, /upload
├── core/
│   ├── loop_controller.py  # Main orchestration loop — runs evals, tracks best prompt, regenerates when stuck
│   ├── prompt_generator.py # Generates and refines system prompts (Gemini 2.5 Pro/Flash)
│   ├── eval_generator.py   # Generates 15 test cases from description + context files
│   └── eval_runner.py      # Runs evals in parallel, 4-dimension judge scoring
├── frontend/
│   └── index.html          # Full UI — single file, no framework
├── requirements.txt
├── runtime.txt             # Python 3.11 for Render
└── Procfile                # Render start command
```

---

## Key settings (loop_controller.py)

| Setting | Value | Meaning |
|---------|-------|---------|
| `MAX_ITERATIONS` | 20 | Max improvement rounds before returning best result |
| `PASS_THRESHOLD` | 0.95 | 95% pass rate required to stop |
| `NO_IMPROVEMENT_LIMIT` | 7 | Rounds with no improvement before regenerating from scratch |
| `MAX_FRESH_STARTS` | 2 | How many full prompt regenerations to try when stuck |

---

## Models used

| Task | Model | Why |
|------|-------|-----|
| Initial prompt generation | `gemini-2.5-pro` | Highest quality for the first draft |
| Prompt refinement | `gemini-2.5-flash` | Fast enough for iteration, good quality |
| Eval case generation | `gemini-2.5-flash` | Speed matters here |
| Test model (runs the prompt) | `gemini-2.5-flash` | Simulates a real AI assistant |
| Judge model (scores responses) | `gemini-2.5-flash` | Evaluates across 4 dimensions |

---

## Deployment (Render)

The app is deployed on Render. Every push to `main` triggers an automatic redeploy.

**Environment variables needed on Render:**
- `GEMINI_API_KEY` — your Google Gemini API key

**To get a Gemini API key:** https://aistudio.google.com/app/apikey

---

## Known limitations

- Free Gemini API tier has rate limits — running multiple sessions simultaneously may cause slowdowns
- 95% target means 1–2 test cases may still fail in the final result (judge scoring has slight randomness)
- Very domain-specific prompts (e.g. medical, legal) may need more than 20 rounds — increase `MAX_ITERATIONS` if needed
- Run mode supports images up to 20MB; very large PDFs may take 30–60 seconds to process
