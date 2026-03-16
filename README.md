# Chainlit + LangGraph + Gemini + Playwright Demo

This is a minimal demo project:
- Chainlit as Chat UI
- LangGraph to orchestrate: parse -> act -> summarize
- Gemini 1.5 Flash via LangChain
- Playwright fills a local sample form (name/email/message) and returns a screenshot

## Setup

1. Install `uv` if you haven't already.
2. Install dependencies and create a virtual environment:

```bash
uv sync
```

3. Install Playwright browsers:

```bash
uv run playwright install
```

4. Set your Gemini API key:

```bash
export GEMINI_API_KEY=your_key_here
```

## Run

```bash
uv run chainlit run app.py -w
```

Then open the Chainlit UI and send a message like:

"Please submit the form with name Alice, email alice@example.com, and message Hello from Chainlit."

## Files

- `app.py`: Chainlit app + LangGraph flow
- `sample_site/index.html`: Local form for Playwright
- `outputs/last_run.png`: Latest screenshot (generated at runtime)
