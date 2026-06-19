# BCU AI Hackathon 2026 — Team <TEAMNAME>

Generative-AI pipeline that answers 100 multiple-choice questions using
internet-assisted retrieval + an **≤8B-parameter** LLM.

## Model used
- **Primary:** `llama-3.1-8b-instant` via Groq API — **8B parameters** (≤8B ✓)
- **Local alternative:** `qwen2.5:7b` via Ollama — **7B parameters** (≤8B ✓)
- No model above 8B is used at any point.

## How it works
1. Load `questions_100.csv`.
2. Build focused search queries from each question + its options.
3. Retrieve evidence from **Wikipedia API** (primary) and **DuckDuckGo** (fallback).
4. Deduplicate, split into sentences, rank with **BM25** against question+options.
5. Build a strict evidence-grounded RAG prompt.
6. LLM returns a single letter; output is validated to `A`–`E`.
7. Export `TEAMNAME_submission.csv`.

## Install
```bash
python3 -m pip install -r starter_code/requirements.txt
```

## Run
```bash
# Recommended: Groq (set your key first)
export GROQ_API_KEY=your_key_here          # Windows: set GROQ_API_KEY=your_key_here
python3 starter_code/run.py --backend groq --output TEAMNAME_submission.csv

# Local, no API key (needs `ollama serve` + `ollama pull qwen2.5:7b`)
python3 starter_code/run.py --backend ollama --output TEAMNAME_submission.csv

# Zero-dependency safety baseline (no LLM, no key)
python3 starter_code/run.py --backend heuristic --output TEAMNAME_submission.csv

# Quick 5-question smoke test
python3 starter_code/run.py --backend groq --limit 5
```

## Answer policy
Final answers are **A–E only** (no `Unknown`), matching `docs/submission_checklist.md`.
A guess always beats `Unknown` on a 5-option question. To allow `Unknown`
(only if the organiser confirms it is scored), add `--allow-unknown`.

## Reproducibility
- Deterministic LLM calls (`temperature=0`).
- Full evidence + confidence logged to `debug_log.jsonl`.
