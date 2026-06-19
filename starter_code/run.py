"""
BCU AI Hackathon 2026 - Improved QA pipeline
=============================================
Drop-in replacement for starter_code/run.py.

Pipeline:
    load questions -> build queries -> retrieve evidence (Wikipedia + DuckDuckGo)
    -> dedupe + BM25 rank -> strict RAG prompt -> 8B LLM -> validate letter
    -> export TEAMNAME_submission.csv (+ optional debug log)

Backends (choose with --backend):
    heuristic : NO LLM. Pure BM25 option-vs-evidence scoring. Zero API keys,
                runs in minutes, always produces a complete A-E submission.
                Use this as your guaranteed fallback.
    groq      : Groq API, model 'llama-3.1-8b-instant' (8B, compliant). Needs
                env var GROQ_API_KEY. Fastest accurate option for a 4h hackathon.
    ollama    : Local Ollama server. Default model 'qwen2.5:7b' (7B, compliant).
                Needs `ollama serve` running. No API key, fully offline-capable.

Model rule: every backend here is <= 8B parameters. Do NOT switch to a >8B model.

Unknown rule (repo is ambiguous):
    README "Answer format" allows A/B/C/D/E/Unknown, but docs/submission_checklist.md
    and run.py both say final answers must be A-E only. SAFEST strategy = always
    emit a letter (never blank, never Unknown), because with 5 options a guess
    beats a guaranteed-zero "Unknown". Default is --no-allow-unknown.
    Pass --allow-unknown ONLY if the organiser confirms Unknown is scored and there
    is negative marking.
"""

import argparse
import json
import os
import re
import time
import urllib.parse
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# ---- optional deps, all degrade gracefully -------------------------------
try:
    import requests
except ImportError:
    requests = None

try:
    from ddgs import DDGS            # new package name
    DDGS_AVAILABLE = True
except ImportError:
    try:
        from duckduckgo_search import DDGS   # old package name fallback
        DDGS_AVAILABLE = True
    except ImportError:
        DDGS_AVAILABLE = False

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False


# -----------------------------
# Configuration
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_QUESTIONS_FILE = BASE_DIR / "questions_100.csv"
if not DEFAULT_QUESTIONS_FILE.exists():
    DEFAULT_QUESTIONS_FILE = BASE_DIR.parent / "questions_100.csv"
DEFAULT_OUTPUT_FILE = "TEAMNAME_submission.csv"

OPTION_LETTERS = ["A", "B", "C", "D", "E"]
WIKI_API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "BCU-Hackathon-2026/1.0 (educational; contact team)"
SEARCH_SLEEP_SECONDS = 0.4
GROQ_MODEL = "llama-3.1-8b-instant"   # 8B - compliant
OLLAMA_MODEL = "qwen2.5:7b"           # 7B - compliant
OLLAMA_URL = "http://localhost:11434/api/chat"

STOPWORDS = set("""a an the of in on at to for and or is are was were be been being which what who
whom whose when where why how that this these those as by with from into during according
following did does do done has have had not no than then into about over under between""".split())


# -----------------------------
# Data loading
# -----------------------------
def load_questions(path: str) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Could not find {path}. Pass --questions with the correct path.")
    questions = pd.read_csv(file_path)
    required = {"question_no", "question"}
    missing = required - set(questions.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return questions


def get_options(row: pd.Series) -> Dict[str, str]:
    out = {}
    for letter in OPTION_LETTERS:
        val = row.get(letter, "")
        if pd.notna(val) and str(val).strip():
            out[letter] = str(val).strip()
    return out


# -----------------------------
# Query construction (improved)
# -----------------------------
def keywords(text: str) -> List[str]:
    toks = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-']+", str(text).lower())
    return [t for t in toks if t not in STOPWORDS and len(t) > 2]


def build_search_queries(row: pd.Series) -> List[str]:
    """Return a short, ranked list of search queries (most specific first)."""
    question = str(row.get("question", "")).strip()
    # strip meta-phrases that hurt search
    q_clean = re.sub(r"(?i)\b(according to wikipedia|as mentioned in the provided.*?|"
                     r"based on the (provided )?(text|passage|article))\b", "", question).strip(" ,.?")
    # capitalised proper-noun spans are the strongest retrieval anchors
    proper = re.findall(r"\b([A-Z][A-Za-z0-9.\-]+(?:\s+[A-Z][A-Za-z0-9.\-]+){0,4})\b", question)
    proper = [p for p in proper if p.lower() not in {"what", "which", "who", "when", "where", "why", "how"}]
    queries = []
    if q_clean:
        queries.append(q_clean)
    if proper:
        queries.append(" ".join(dict.fromkeys(proper))[:120])  # dedupe, keep order
    # nothing useful -> fall back to raw question
    if not queries:
        queries.append(question)
    # dedupe, keep order, cap at 2 (keeps it fast)
    seen, out = set(), []
    for q in queries:
        if q and q.lower() not in seen:
            seen.add(q.lower())
            out.append(q)
    return out[:2]


# -----------------------------
# Evidence retrieval
# -----------------------------
def search_wikipedia(query: str, max_results: int = 3) -> List[Dict[str, str]]:
    if requests is None:
        return []
    out = []
    try:
        sr = requests.get(WIKI_API, params={
            "action": "query", "list": "search", "srsearch": query,
            "srlimit": max_results, "format": "json"},
            headers={"User-Agent": USER_AGENT}, timeout=12).json()
        titles = [hit["title"] for hit in sr.get("query", {}).get("search", [])]
        if not titles:
            return []
        ex = requests.get(WIKI_API, params={
            "action": "query", "prop": "extracts", "exintro": 1, "explaintext": 1,
            "redirects": 1, "titles": "|".join(titles), "format": "json"},
            headers={"User-Agent": USER_AGENT}, timeout=12).json()
        for page in ex.get("query", {}).get("pages", {}).values():
            extract = (page.get("extract") or "").strip()
            if extract:
                out.append({"title": page.get("title", ""), "snippet": extract[:1200],
                            "url": "https://en.wikipedia.org/wiki/" +
                                   urllib.parse.quote(page.get("title", "").replace(" ", "_"))})
    except Exception as e:
        print(f"[warn] wikipedia failed: {e}")
    return out


def search_duckduckgo(query: str, max_results: int = 4) -> List[Dict[str, str]]:
    if not DDGS_AVAILABLE:
        return []
    out = []
    try:
        with DDGS() as ddgs:
            for item in ddgs.text(query, max_results=max_results) or []:
                out.append({"title": item.get("title", ""),
                            "snippet": item.get("body", ""),
                            "url": item.get("href", "")})
    except Exception as e:
        print(f"[warn] ddg failed: {e}")
    time.sleep(SEARCH_SLEEP_SECONDS)
    return out


def retrieve_evidence(row: pd.Series, use_ddg: bool = True) -> List[Dict[str, str]]:
    evidence: List[Dict[str, str]] = []
    for q in build_search_queries(row):
        evidence += search_wikipedia(q, max_results=3)
        if use_ddg:
            evidence += search_duckduckgo(q, max_results=4)
    # dedupe by url then by snippet text
    seen, deduped = set(), []
    for e in evidence:
        key = (e.get("url") or "") + "|" + (e.get("snippet") or "")[:80]
        if key not in seen and (e.get("snippet") or "").strip():
            seen.add(key)
            deduped.append(e)
    return deduped


# -----------------------------
# Evidence ranking (BM25 over sentences, keyword fallback)
# -----------------------------
def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", str(text))
    return [p.strip() for p in parts if len(p.strip()) > 25]


def rank_evidence(row: pd.Series, evidence: List[Dict[str, str]], top_k: int = 6) -> List[Dict[str, str]]:
    """Sentence-level ranking against question + all options."""
    if not evidence:
        return []
    sentences = []
    for e in evidence:
        for s in split_sentences(e.get("snippet", "")):
            sentences.append({"text": s, "title": e.get("title", ""), "url": e.get("url", "")})
    if not sentences:
        return evidence[:top_k]

    options = get_options(row)
    query_terms = keywords(row.get("question", "")) + sum((keywords(v) for v in options.values()), [])

    if BM25_AVAILABLE and query_terms:
        tokenised = [keywords(s["text"]) for s in sentences]
        bm25 = BM25Okapi(tokenised)
        scores = bm25.get_scores(query_terms)
    else:  # keyword-overlap fallback
        qset = set(query_terms)
        scores = [len(qset & set(keywords(s["text"]))) for s in sentences]

    ranked = sorted(zip(scores, sentences), key=lambda x: x[0], reverse=True)
    out = []
    for score, s in ranked[:top_k]:
        s = dict(s)
        s["score"] = round(float(score), 3)
        out.append(s)
    return out


def heuristic_answer(row: pd.Series, ranked: List[Dict[str, str]]) -> (str, float):
    """No-LLM baseline: score each option by term overlap with top evidence."""
    options = get_options(row)
    if not options:
        return "A", 0.0
    evidence_terms: List[str] = []
    for s in ranked[:6]:
        evidence_terms += keywords(s.get("text", ""))
    ev = set(evidence_terms)
    if not ev:
        return next(iter(options)), 0.0
    scores = {l: len(set(keywords(t)) & ev) for l, t in options.items()}
    best = max(scores, key=scores.get)
    total = sum(scores.values()) or 1
    confidence = scores[best] / total
    return best, round(confidence, 3)


# -----------------------------
# Prompt building (strict)
# -----------------------------
def build_prompt(row: pd.Series, ranked: List[Dict[str, str]], allow_unknown: bool) -> str:
    options = get_options(row)
    options_block = "\n".join(f"{l}. {t}" for l, t in options.items())
    ev_block = "\n".join(f"[{i+1}] {s.get('title','')}: {s.get('text','')}"
                         for i, s in enumerate(ranked[:6])) or "(no evidence retrieved)"
    valid = "A, B, C, D, or E" + (", or Unknown" if allow_unknown else "")
    unknown_rule = ("If the evidence is missing or conflicting, you may answer Unknown.\n"
                    if allow_unknown else
                    "Even if unsure, you MUST pick the single most likely option. Do not refuse.\n")
    return (
        "You are answering a multiple-choice question. Use the evidence when relevant, "
        "otherwise use your own knowledge.\n"
        f"Reply with ONLY one character ({valid}). No words, no punctuation, no explanation.\n"
        f"{unknown_rule}\n"
        f"Question:\n{row.get('question','')}\n\n"
        f"Options:\n{options_block}\n\n"
        f"Evidence:\n{ev_block}\n\n"
        "Answer:"
    )


# -----------------------------
# LLM backends (all <= 8B)
# -----------------------------
def call_groq(prompt: str) -> str:
    key = os.environ.get("GROQ_API_KEY")
    if not key or requests is None:
        return ""
    try:
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"model": GROQ_MODEL, "temperature": 0, "max_tokens": 4,
                  "messages": [{"role": "user", "content": prompt}]},
            timeout=30)
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"[warn] groq failed: {e}")
        return ""


def call_ollama(prompt: str) -> str:
    if requests is None:
        return ""
    try:
        r = requests.post(OLLAMA_URL, json={
            "model": OLLAMA_MODEL, "stream": False,
            "options": {"temperature": 0, "num_predict": 4},
            "messages": [{"role": "user", "content": prompt}]}, timeout=120)
        return r.json().get("message", {}).get("content", "")
    except Exception as e:
        print(f"[warn] ollama failed: {e}")
        return ""


# -----------------------------
# Answer validation
# -----------------------------
def extract_letter(text: str, allow_unknown: bool) -> str:
    if not text:
        return "Unknown" if allow_unknown else ""
    if allow_unknown and re.search(r"\bunknown\b", text, re.I):
        return "Unknown"
    m = re.search(r"\b([ABCDE])\b", text.upper())
    if m:
        return m.group(1)
    m = re.search(r"[ABCDE]", text.upper())
    return m.group(0) if m else ("Unknown" if allow_unknown else "")


# -----------------------------
# Per-question pipeline
# -----------------------------
def answer_question(row: pd.Series, backend: str, allow_unknown: bool,
                    use_ddg: bool, votes: int, logf=None) -> Dict[str, str]:
    qno = row.get("question_no")
    evidence = retrieve_evidence(row, use_ddg=use_ddg)
    ranked = rank_evidence(row, evidence)

    if backend == "heuristic":
        answer, conf = heuristic_answer(row, ranked)
    else:
        prompt = build_prompt(row, ranked, allow_unknown)
        caller = call_groq if backend == "groq" else call_ollama
        picks = [extract_letter(caller(prompt), allow_unknown) for _ in range(max(1, votes))]
        picks = [p for p in picks if p]
        if picks:
            answer = max(set(picks), key=picks.count)             # majority vote
            conf = picks.count(answer) / len(picks)
        else:
            # LLM gave nothing usable -> fall back to heuristic so we never blank
            answer, conf = heuristic_answer(row, ranked)

    # final safety net: only allow blank/Unknown if explicitly enabled
    if answer not in OPTION_LETTERS:
        if allow_unknown:
            answer = "Unknown"
        else:
            answer, _ = heuristic_answer(row, ranked)
            if answer not in OPTION_LETTERS:
                answer = "A"

    if logf is not None:
        logf.write(json.dumps({
            "question_no": int(qno) if pd.notna(qno) else None,
            "answer": answer, "confidence": conf, "backend": backend,
            "n_evidence": len(evidence),
            "top_evidence": [{"score": s.get("score"), "title": s.get("title"),
                              "text": s.get("text", "")[:160]} for s in ranked[:3]],
        }, ensure_ascii=False) + "\n")
        logf.flush()
    return {"question_no": qno, "answer": answer}


def run_pipeline(args) -> pd.DataFrame:
    questions = load_questions(args.questions)
    if args.limit is not None:
        questions = questions.head(args.limit)

    logf = open(args.log, "w", encoding="utf-8") if args.log else None
    preds = []
    for idx, row in questions.iterrows():
        qno = row.get("question_no", idx + 1)
        print(f"Answering question {qno} ...")
        preds.append(answer_question(row, args.backend, args.allow_unknown,
                                     not args.no_ddg, args.votes, logf))
    if logf:
        logf.close()

    submission = pd.DataFrame(preds, columns=["question_no", "answer"])
    submission.to_csv(args.output, index=False)
    print(f"\nSaved {len(submission)} answers -> {args.output}")
    dist = submission["answer"].value_counts().to_dict()
    print(f"Answer distribution: {dist}")
    if not args.allow_unknown:
        bad = submission[~submission["answer"].isin(OPTION_LETTERS)]
        if len(bad):
            print(f"[ERROR] {len(bad)} non A-E answers slipped through!")
    return submission


def parse_args():
    p = argparse.ArgumentParser(description="BCU AI Hackathon 2026 - improved pipeline")
    p.add_argument("--questions", default=str(DEFAULT_QUESTIONS_FILE))
    p.add_argument("--output", default=DEFAULT_OUTPUT_FILE)
    p.add_argument("--backend", choices=["heuristic", "groq", "ollama"], default="heuristic")
    p.add_argument("--votes", type=int, default=1, help="self-consistency votes for LLM backends")
    p.add_argument("--limit", type=int, default=None, help="answer only first N (for testing)")
    p.add_argument("--no-ddg", action="store_true", help="Wikipedia only (faster, no DDG rate limits)")
    p.add_argument("--log", default="debug_log.jsonl", help="JSONL debug log path ('' to disable)")
    p.add_argument("--allow-unknown", dest="allow_unknown", action="store_true",
                   help="permit 'Unknown' answers (default OFF - emit A-E only)")
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    if a.log == "":
        a.log = None
    run_pipeline(a)
