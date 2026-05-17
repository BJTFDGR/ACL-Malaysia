"""
Round 2 Benchmarking — GOLD dataset only.

Tests run:
  1b  Macro-Function Classification (Unassisted)
        Prompt the LLM to predict the Macro-Function from the utterance alone.
  1c  Macro-Function Classification (Attribute-Assisted)
        Provide human ground-truth attributes; prompt LLM to predict Macro-Function.
  2b  Function-Constrained Particle Generation
        Provide target Macro-Function; ask LLM to predict the masked particle.

Macro-Function labels are derived from the 16-cluster k-modes results:
  Clusters 0, 6, 9   → Assumed-Agreement Rhetorical Stance
  Cluster  1         → Neutral Declarative
  Clusters 4, 10, 13 → Information-Seeking Verification
  Clusters 2, 14, 15 → Affective Confirmation-Seeking Question
  Cluster  3         → Emphatic / Discourse-Marking
  Clusters 5, 12     → Null Form Retaining Particle-Like Pragmatic Meaning
  Clusters 11, 7, 8  → Negative Rhetorical Challenge / Evaluation

Execution flow:
  1. Smoke-test on 1 row (all models, all three tests).
  2. If smoke passes, full run on all GOLD rows with 50 parallel workers.
  3. Error-retry pass for any remaining ERROR rows.

Environment overrides (optional):
  PARALLEL_WORKERS          (default 50)
  STOP_AFTER_SMOKE          set to "1" to exit after smoke test
  STOP_AFTER_1B             set to "1" to exit after Test 1b saves (useful for iterating on 1c/2b)
  SKIP_1B                   set to "1" to skip Test 1b entirely and start from Test 1c
  SKIP_RETRY                set to "1" to skip the final error-retry pass
  OPENAI_API_KEY / ANTHROPIC_API_KEY / GEMINI_API_KEY /
  DEEPSEEK_API_KEY / SEA_LION_API_KEY / LLAMA_BASE_URL
"""

# ─────────────────────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────────────────────

import json
import os
import random
import re
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import anthropic
from google import genai
from google.genai import types as genai_types
import pandas as pd
import requests
from openai import OpenAI
from tqdm import tqdm

try:
    from IPython.display import display
except Exception:
    def display(x):
        print(x)

pd.set_option("display.max_colwidth", 120)


# ─────────────────────────────────────────────────────────────────────────────
# Runtime state
# ─────────────────────────────────────────────────────────────────────────────

IO_LOGS: list = []
RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

FINAL_METRICS = Path("../Final Metrics")
IO_LOG_JSON        = FINAL_METRICS / f"round2_io_logs_{RUN_ID}.json"
IO_LOG_JSON_LATEST = FINAL_METRICS / "round2_io_logs.json"

PARALLEL_WORKERS = int(os.getenv("PARALLEL_WORKERS", "50"))

# Per-model concurrency caps — respects rate limits
MODEL_MAX_WORKERS = {
    "claude_strong":  5,
    "claude_weak":    5,
    "sealion_strong": 1,
    "sealion_weak":   1,
}

MODEL_FATAL_ERRORS: dict = {}


# ─────────────────────────────────────────────────────────────────────────────
# Label sets
# ─────────────────────────────────────────────────────────────────────────────

MACRO_FUNCTION_LABELS = [
    "Assumed-Agreement Rhetorical Stance",
    "Neutral Declarative",
    "Information-Seeking Verification",
    "Affective Confirmation-Seeking Question",
    "Emphatic / Discourse-Marking",
    "Null Form Retaining Particle-Like Pragmatic Meaning",
    "Negative Rhetorical Challenge / Evaluation",
]

MACRO_FUNCTION_DEFINITIONS = {
    "Assumed-Agreement Rhetorical Stance":
        "Speaker presents proposition as already obvious/shared knowledge; "
        "listener is expected to align rather than genuinely answer.",
    "Neutral Declarative":
        "Plain informational statements with minimal discourse pressure or stance marking.",
    "Information-Seeking Verification":
        "Genuine request for verification or clarification; speaker leaves room for disagreement.",
    "Affective Confirmation-Seeking Question":
        "Speaker seeks confirmation while simultaneously expressing affect "
        "(surprise, irritation, humour, excitement, disbelief, etc.)",
    "Emphatic / Discourse-Marking":
        "Particle functions less as a literal confirmation marker and more as a "
        "discourse-management or emphasis device.",
    "Null Form Retaining Particle-Like Pragmatic Meaning":
        "Pragmatic meaning associated with particles remains inferable even after "
        "overt particle removal.",
    "Negative Rhetorical Challenge / Evaluation":
        "Speaker uses rhetorical questioning to criticise, challenge, mock, or "
        "negatively evaluate a proposition rather than genuinely seek information.",
}

PARTICLE_LABELS = ["ke", "kan", "neutral"]

# Cluster → Macro-Function mapping (16-cluster k-modes)
CLUSTER_TO_MACRO = {
    0:  "Assumed-Agreement Rhetorical Stance",
    6:  "Assumed-Agreement Rhetorical Stance",
    9:  "Assumed-Agreement Rhetorical Stance",
    1:  "Neutral Declarative",
    4:  "Information-Seeking Verification",
    10: "Information-Seeking Verification",
    13: "Information-Seeking Verification",
    2:  "Affective Confirmation-Seeking Question",
    14: "Affective Confirmation-Seeking Question",
    15: "Affective Confirmation-Seeking Question",
    3:  "Emphatic / Discourse-Marking",
    5:  "Null Form Retaining Particle-Like Pragmatic Meaning",
    12: "Null Form Retaining Particle-Like Pragmatic Meaning",
    11: "Negative Rhetorical Challenge / Evaluation",
    7:  "Negative Rhetorical Challenge / Evaluation",
    8:  "Negative Rhetorical Challenge / Evaluation",
}

ATTRIBUTE_DESCRIPTIONS = {
    "Epistemic_Stance": {
        "Certain":       "The speaker treats the statement as already true or established — no hedging, no doubt, full confidence.",
        "Uncertain":     "The speaker sounds unsure, is making a guess, estimating, or checking whether something is the case.",
        "Neutral/Unclear": "No detectable certainty signal; the sentence does not lean toward confident assertion or uncertainty.",
        "Neutral / NA":  "No detectable certainty signal; epistemic stance is not relevant here.",
    },
    "Particle_Position": {
        "Front":      "The particle appears at the very start of the sentence, before any other content words.",
        "Middle/End": "The particle appears anywhere other than the front — mid-sentence or at the end.",
        "N/A":        "No discourse particle position is applicable.",
    },
    "Listener_Agreement": {
        "Assumed Agreement":    "The speaker treats the information as shared or obvious — the underlying tone is 'you already know this'.",
        "Confirmation Seeking": "The speaker is actively checking whether the listener agrees or can confirm something.",
        "Neutral/Unclear":      "No clear orientation toward listener agreement; a plain statement or ambiguous stance.",
    },
    "Emotion": {
        "Positive":        "The speaker expresses happiness, excitement, enthusiasm, satisfaction, humour, affection, or relief.",
        "Negative":        "The speaker expresses frustration, annoyance, disappointment, sadness, anger, or bitterness.",
        "Neutral/Unclear": "No detectable emotional charge, or the emotion is genuinely ambiguous.",
    },
    "Question_Type": {
        "Declarative/Statement":   "The sentence makes an assertion or conveys information — not structured as a question.",
        "Rhetorical Interrogative": "Phrased as a question but does not expect a genuine answer; used to make a point or emphasise something.",
        "Yes/No Interrogative":    "A genuine question inviting the listener to confirm or deny; the speaker does not already know the answer.",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Prompts (from prompt_text_phase2.md)
# ─────────────────────────────────────────────────────────────────────────────

# ── Test 1b: Macro-Function Classification (Unassisted) ──────────────────────

MACRO_LABEL_LIST_TEXT = "\n\n".join(
    f"{label}: {MACRO_FUNCTION_DEFINITIONS[label]}"
    for label in MACRO_FUNCTION_LABELS
)

TEST_1B_SYSTEM = (
    "You are a linguist specialising in colloquial Malay discourse pragmatics. "
    "You must output exactly one label from the provided list and nothing else."
)

TEST_1B_PROMPT_TEMPLATE = (
    "You are a linguist specialising in colloquial Malay discourse pragmatics. "
    "Your task is to read the Malay sentence below and identify the primary communicative role "
    "the utterance plays in interaction, beyond its literal propositional content.\n\n"
    "Referring to the following seven labels and their definitions to make your decision:\n\n"
    + MACRO_LABEL_LIST_TEXT
    + '\n\nSpeaker: "{TEXT}"\n\n'
    "Considering what the speaker is communicatively doing with this utterance, their stance, "
    "their orientation toward the listener, and the function the sentence serves in interaction, "
    "which of the seven labels best captures its primary discourse role?\n"
    "The most likely, single label is:"
)

# ── Test 1c: Macro-Function Classification (Attribute-Assisted) ──────────────

TEST_1C_SYSTEM = (
    "You are a linguist specialising in colloquial Malay discourse pragmatics. "
    "You must output exactly one label from the provided list and nothing else."
)

TEST_1C_PROMPT_TEMPLATE = (
    "You are a linguist specialising in colloquial Malay discourse pragmatics. "
    "Your task is to read the Malay sentence below and identify the primary communicative role "
    "the utterance plays in interaction, beyond its literal propositional content.\n\n"
    "You are provided with the following human-annotated attribute labels for this sentence "
    "as additional context:\n{attr_block}\n\n"
    "Use these attributes to inform your decision, but base your final label on the overall "
    "communicative function of the utterance.\n\n"
    "Referring to the following seven labels and their definitions to make your decision:\n\n"
    + MACRO_LABEL_LIST_TEXT
    + '\n\nSpeaker: "{TEXT}"\n\n'
    "Considering what the speaker is communicatively doing with this utterance, their stance, "
    "their orientation toward the listener, and the function the sentence serves in interaction, "
    "which of the seven labels best captures its primary discourse role?\n"
    "The most likely, single label is:"
)

# ── Test 2b: Function-Constrained Particle Generation ────────────────────────

TEST_2B_SYSTEM = (
    "You are a linguist specialising in colloquial Malay discourse particles. "
    'You must output exactly one word — either "ke" or "kan" or "neutral" — and nothing else.'
)

TEST_2B_PROMPT_TEMPLATE = (
    "You are given a Malay sentence in which one discourse particle has been replaced with [___].\n"
    "Your task is to predict which particle — \"ke\", \"kan\", or \"neutral\" — belongs in the "
    "[___] slot, such that the sentence is consistent with the primary communicative role the "
    "utterance plays in interaction, beyond its literal propositional content:\n"
    "{marco_function_attr_block}\n\n\n"
    "Particle meanings:\n"
    "  ke      : signals genuine uncertainty or invites the listener to confirm something "
    "the speaker is unsure about.\n"
    "  kan     : signals assumed shared knowledge and seeks light confirmation of something "
    'the speaker already believes ("right?").\n'
    "  neutral : indicates no particle.\n\n"
    'Speaker:\n  "{TEXT}"\n\n'
    "Using the sentence context and the macro-function above, which single particle — "
    '"ke" or "kan" or "neutral" — best fills [___]?\n\n'
    "Return exactly one word from this set and nothing else: ke, kan, neutral"
)

TEST_2B_STRICT_TAIL = ""  # tail already embedded in template above


# ─────────────────────────────────────────────────────────────────────────────
# Prompt builders
# ─────────────────────────────────────────────────────────────────────────────

def _build_attr_block(row: pd.Series) -> str:
    """Build human-annotated attribute block for Test 1c."""
    lines = []

    es_val = str(row["Epistemic_Stance"]).strip()
    es_desc = ATTRIBUTE_DESCRIPTIONS["Epistemic_Stance"].get(es_val, "")
    # append val: description if description exists, else just val (which may be empty or "Neutral/Unclear")
    lines.append(f"{es_val}: {es_desc}" if es_desc else es_val)

    pp_val = str(row["Particle_Position"]).strip()
    pp_desc = ATTRIBUTE_DESCRIPTIONS["Particle_Position"].get(pp_val, "")
    lines.append(f"{pp_val}: {pp_desc}" if pp_desc else pp_val)

    la_val = str(row["Listener_Agreement"]).strip()
    la_desc = ATTRIBUTE_DESCRIPTIONS["Listener_Agreement"].get(la_val, "")
    lines.append(f"{la_val}: {la_desc}" if la_desc else la_val)

    em_val = str(row["Emotion"]).strip()
    em_desc = ATTRIBUTE_DESCRIPTIONS["Emotion"].get(em_val, "")
    lines.append(f"{em_val}: {em_desc}" if em_desc else em_val)

    qt_val = str(row["Question_Type"]).strip()
    qt_desc = ATTRIBUTE_DESCRIPTIONS["Question_Type"].get(qt_val, "")
    lines.append(f"{qt_val}: {qt_desc}" if qt_desc else qt_val)

    return "\n".join(lines)


def _build_macro_function_block(macro_label: str) -> str:
    """Build macro-function context block for Test 2b."""
    defn = MACRO_FUNCTION_DEFINITIONS.get(macro_label, "")
    if defn:
        return f"  {macro_label}: {defn}"
    return f"  {macro_label}"


def build_1b_prompt(text: str) -> str:
    return TEST_1B_PROMPT_TEMPLATE.replace("{TEXT}", text)


def build_1c_prompt(text: str, row: pd.Series) -> str:
    attr_block = _build_attr_block(row)
    return TEST_1C_PROMPT_TEMPLATE.replace("{attr_block}", attr_block).replace("{TEXT}", text)


def build_2b_prompt(text_masked: str, macro_label: str) -> str:
    block = _build_macro_function_block(macro_label)
    return TEST_2B_PROMPT_TEMPLATE.replace("{marco_function_attr_block}", block).replace("{TEXT}", text_masked)


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def _save_io_logs():
    IO_LOG_JSON.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(IO_LOGS, ensure_ascii=False, indent=2)
    IO_LOG_JSON.write_text(payload, encoding="utf-8")
    IO_LOG_JSON_LATEST.write_text(payload, encoding="utf-8")


def _append_io_log(entry, log_sink=None):
    sink = IO_LOGS if log_sink is None else log_sink
    sink.append(entry)


def _clean_raw(text: str) -> str:
    """Normalise LLM raw output: remove think tags and SEA-LION tokenizer unicode artifacts."""
    # SEA-LION tokenizer encodes newlines as \u010a and spaces as \u0120
    text = text.replace('\u010a', '\n').replace('\u0120', ' ')
    text = re.sub(r'[\u0100-\u017f]', ' ', text)   # other BPE whitespace tokens
    # Strip <think>...</think> blocks
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # Strip orphan closing </think> and everything before it
    text = re.sub(r'^.*?</think>', '', text, flags=re.DOTALL)
    return text.strip()


def _extract_macro_label(raw: str) -> str:
    """Extract one of the 7 Macro-Function labels from raw LLM output."""
    text = _clean_raw(str(raw or ""))
    text_lower = text.lower()

    # 1. Exact full match (case-insensitive)
    for label in MACRO_FUNCTION_LABELS:
        if label.lower() == text_lower:
            return label

    # 2. Substring match — longest label first to avoid false partial hits
    for label in sorted(MACRO_FUNCTION_LABELS, key=len, reverse=True):
        if label.lower() in text_lower:
            return label

    # 3. Key-phrase heuristics (ordered most-specific first)
    HEURISTICS = [
        (["null form", "particle-like", "retaining"],
         "Null Form Retaining Particle-Like Pragmatic Meaning"),
        (["negative rhetorical", "rhetorical challenge", "challenge",
          "criticis", "mock", "negatively evaluat"],
         "Negative Rhetorical Challenge / Evaluation"),
        (["affective confirmation", "affective", "confirmation-seeking", "affect"],
         "Affective Confirmation-Seeking Question"),
        (["assumed-agreement", "assumed agreement", "rhetorical stance"],
         "Assumed-Agreement Rhetorical Stance"),
        (["information-seeking", "information seeking",
          "verification", "genuine request"],
         "Information-Seeking Verification"),
        (["emphatic", "discourse-marking", "discourse marking",
          "discourse management", "emphasis device"],
         "Emphatic / Discourse-Marking"),
        (["neutral declarative", "plain informational", "plain statement"],
         "Neutral Declarative"),
        # last-resort single-word fallbacks
        (["negative"],    "Negative Rhetorical Challenge / Evaluation"),
        (["assumed"],     "Assumed-Agreement Rhetorical Stance"),
        (["neutral"],     "Neutral Declarative"),
        (["emphatic"],    "Emphatic / Discourse-Marking"),
        (["declarative"], "Neutral Declarative"),
    ]
    for keywords, label in HEURISTICS:
        if any(kw in text_lower for kw in keywords):
            return label

    return text   # return cleaned text for error analysis


def _extract_particle(raw: str) -> str:
    """Extract ke / kan / neutral from raw LLM output."""
    text = _clean_raw(str(raw or "")).lower()
    if text in {"ke", "kan", "neutral"}:
        return text
    for p in ["neutral", "kan", "ke"]:
        if re.search(rf"\b{p}\b", text):
            return p
    return str(raw or "").strip()


def _is_fatal_error(error):
    text = str(error).lower()
    return any(m in text for m in [
        "model_not_found", "does not exist", "unsupported parameter",
        "unsupported value", "invalid_request_error", "api key not valid",
        "permission", "authentication", "insufficient_quota",
    ])


def _is_rate_limit_error(error):
    text = str(error).lower()
    return "429" in text or "rate_limit" in text or "ratelimit" in text or "too many requests" in text


def _probe_ollama(base_url, timeout=2):
    try:
        resp = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=timeout)
        return resp.ok
    except Exception:
        return False


def _ensure_ollama_runtime(base_url):
    if _probe_ollama(base_url):
        return True, "Ollama endpoint is reachable."
    parsed = urlparse(base_url)
    host = (parsed.hostname or "").lower()
    is_local = host in {"localhost", "127.0.0.1", "::1"}
    ollama_cmd = shutil.which("ollama")
    if not ollama_cmd:
        return False, "Ollama not installed."
    if not is_local:
        return False, f"Ollama not reachable at {base_url}"
    try:
        subprocess.Popen(
            [ollama_cmd, "serve"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except Exception as exc:
        return False, f"Failed to start ollama serve: {exc}"
    for _ in range(12):
        time.sleep(1)
        if _probe_ollama(base_url):
            return True, "Ollama auto-started."
    return False, "Ollama still unreachable after auto-start attempt."


def _extract_keys_from_notebook(nb_path: Path):
    keys = {"openai": None, "anthropic": None, "gemini": [], "deepseek": None, "sealion": None}
    try:
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        text = "\n".join(
            "\n".join(cell.get("source", []))
            for cell in nb.get("cells", [])
            if cell.get("cell_type") == "markdown"
        )
        patterns = {
            "openai":    r"GPT\s*\(general\s*\)\s*:\s*(sk-[A-Za-z0-9_\-]+)",
            "anthropic": r"Claude\s*:\s*(sk-ant-[A-Za-z0-9_\-]+)",
            "deepseek":  r"DeepSeek\s*:\s*(sk-[A-Za-z0-9_\-]+)",
            "sealion":   r"SEA-LION\s*:\s*(sk-[A-Za-z0-9_\-]+)",
        }
        for k, pat in patterns.items():
            m = re.search(pat, text)
            if m:
                keys[k] = m.group(1).strip()
        keys["gemini"] = re.findall(r"AIza[0-9A-Za-z_\-]+", text)
    except Exception:
        pass
    return keys


# ─────────────────────────────────────────────────────────────────────────────
# API keys and model registry
# ─────────────────────────────────────────────────────────────────────────────

notebook_path = Path("04_round1_1a_attribute_accuracy.ipynb")
fallback = _extract_keys_from_notebook(notebook_path)

OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY")    or fallback["openai"]
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY") or fallback["anthropic"]
GEMINI_API_KEYS   = [k for k in [os.getenv("GEMINI_API_KEY")] + fallback["gemini"] if k]
DEEPSEEK_API_KEY  = os.getenv("DEEPSEEK_API_KEY")  or fallback["deepseek"]
SEA_LION_API_KEY  = os.getenv("SEA_LION_API_KEY")  or fallback["sealion"]

LLAMA_BASE_URL         = os.getenv("LLAMA_BASE_URL",         "http://localhost:11434")
LLAMA_STRONG_MODEL     = os.getenv("LLAMA_STRONG_MODEL",     "llama3.2:7b")
LLAMA_WEAK_MODEL       = os.getenv("LLAMA_WEAK_MODEL",       "llama3.2:1b")
SEA_LION_BASE_URL      = os.getenv("SEA_LION_BASE_URL",      "https://api.sea-lion.ai/v1")
SEA_LION_STRONG_MODEL  = os.getenv("SEA_LION_STRONG_MODEL",  "aisingapore/Llama-SEA-LION-v3.5-70B-R")
SEA_LION_WEAK_MODEL    = os.getenv("SEA_LION_WEAK_MODEL",    "aisingapore/Gemma-SEA-LION-v4-27B-IT")
GPT_STRONG_MODEL       = os.getenv("GPT_STRONG_MODEL",       "gpt-5")
GPT_WEAK_MODEL         = os.getenv("GPT_WEAK_MODEL",         "gpt-5.4-mini")
DEEPSEEK_STRONG_MODEL  = os.getenv("DEEPSEEK_STRONG_MODEL",  "deepseek-v4-pro")
DEEPSEEK_WEAK_MODEL    = os.getenv("DEEPSEEK_WEAK_MODEL",    "deepseek-v4-flash")
CLAUDE_STRONG_MODEL    = os.getenv("CLAUDE_STRONG_MODEL",    "claude-sonnet-4-6")
CLAUDE_WEAK_MODEL      = os.getenv("CLAUDE_WEAK_MODEL",      "claude-haiku-4-5")
GEMINI_STRONG_MODEL    = os.getenv("GEMINI_STRONG_MODEL",    "gemini-3.1-pro-preview")
GEMINI_WEAK_MODEL      = os.getenv("GEMINI_WEAK_MODEL",      "gemini-3.1-flash-lite")
GEMINI_STRONG_FALLBACK_MODELS = [
    m.strip()
    for m in os.getenv("GEMINI_STRONG_FALLBACK_MODELS", "gemini-3.1-flash-lite").split(",")
    if m.strip()
]

MODEL_RUNS = []

if OPENAI_API_KEY:
    MODEL_RUNS.append({
        "name": "gpt_strong", "provider": "openai",
        "client": OpenAI(api_key=OPENAI_API_KEY), "api_key": OPENAI_API_KEY,
        "model": GPT_STRONG_MODEL, "max_tokens": 8192,
        "reasoning_effort": "minimal",
        "sleep": 0.5,
    })
    MODEL_RUNS.append({
        "name": "gpt_weak", "provider": "openai_responses",
        "client": OpenAI(api_key=OPENAI_API_KEY), "api_key": OPENAI_API_KEY,
        "model": GPT_WEAK_MODEL, "max_tokens": 128,
        "sleep": 0.5,
    })

if ANTHROPIC_API_KEY:
    MODEL_RUNS.append({
        "name": "claude_strong", "provider": "anthropic",
        "client": anthropic.Anthropic(api_key=ANTHROPIC_API_KEY),
        "api_key": ANTHROPIC_API_KEY, "model": CLAUDE_STRONG_MODEL, "sleep": 0.5,
    })
    MODEL_RUNS.append({
        "name": "claude_weak", "provider": "anthropic",
        "client": anthropic.Anthropic(api_key=ANTHROPIC_API_KEY),
        "api_key": ANTHROPIC_API_KEY, "model": CLAUDE_WEAK_MODEL, "sleep": 0.5,
    })

if GEMINI_API_KEYS:
    MODEL_RUNS.append({
        "name": "gemini_strong", "provider": "gemini",
        "client": None, "api_keys": GEMINI_API_KEYS,
        "model": GEMINI_STRONG_MODEL,
        "fallback_models": GEMINI_STRONG_FALLBACK_MODELS, "sleep": 0.5,
    })
    MODEL_RUNS.append({
        "name": "gemini_weak", "provider": "gemini",
        "client": None, "api_keys": GEMINI_API_KEYS,
        "model": GEMINI_WEAK_MODEL, "sleep": 0.5,
    })

if DEEPSEEK_API_KEY:
    MODEL_RUNS.append({
        "name": "deepseek_strong", "provider": "openai",
        "client": OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com"),
        "api_key": DEEPSEEK_API_KEY, "base_url": "https://api.deepseek.com",
        "model": DEEPSEEK_STRONG_MODEL, "sleep": 0.3,
    })
    MODEL_RUNS.append({
        "name": "deepseek_weak", "provider": "openai",
        "client": OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com"),
        "api_key": DEEPSEEK_API_KEY, "base_url": "https://api.deepseek.com",
        "model": DEEPSEEK_WEAK_MODEL, "sleep": 0.3,
    })

llama_available, llama_status = _ensure_ollama_runtime(LLAMA_BASE_URL)
print("Llama/Ollama status:", llama_status)
if llama_available:
    MODEL_RUNS.append({
        "name": "llama_strong", "provider": "ollama",
        "client": None, "model": LLAMA_STRONG_MODEL,
        "sleep": 0.2, "base_url": LLAMA_BASE_URL,
    })
    MODEL_RUNS.append({
        "name": "llama_weak", "provider": "ollama",
        "client": None, "model": LLAMA_WEAK_MODEL,
        "sleep": 0.2, "base_url": LLAMA_BASE_URL,
    })
else:
    print("Llama skipped:", llama_status)

if SEA_LION_API_KEY:
    MODEL_RUNS.append({
        "name": "sealion_strong", "provider": "openai",
        "client": OpenAI(api_key=SEA_LION_API_KEY, base_url=SEA_LION_BASE_URL),
        "api_key": SEA_LION_API_KEY, "base_url": SEA_LION_BASE_URL,
        "model": SEA_LION_STRONG_MODEL, "max_tokens": 512, "sleep": 0.3,
    })
    MODEL_RUNS.append({
        "name": "sealion_weak", "provider": "openai",
        "client": OpenAI(api_key=SEA_LION_API_KEY, base_url=SEA_LION_BASE_URL),
        "api_key": SEA_LION_API_KEY, "base_url": SEA_LION_BASE_URL,
        "model": SEA_LION_WEAK_MODEL, "sleep": 0.3,
    })

if not MODEL_RUNS:
    raise ValueError("No model keys/runtimes configured. Set at least one API key.")

print("Models to run:", [m["name"] for m in MODEL_RUNS])
MODEL_VERSION_MAP = {m["name"].upper(): m["model"] for m in MODEL_RUNS}


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

DATA_PATH    = Path("../Datasets/GOLD_187.csv")
CLUSTER_PATH = Path("clustered_kmodes_discourse_context_16.csv")

df_gold = pd.read_csv(DATA_PATH)

# Derive Macro_Function from 16-cluster CSV
_cluster_df = pd.read_csv(CLUSTER_PATH)[["Text", "cluster"]].drop_duplicates("Text")
df_gold = df_gold.merge(_cluster_df, on="Text", how="left")
df_gold["Macro_Function"] = df_gold["cluster"].map(CLUSTER_TO_MACRO)

unmapped = df_gold["Macro_Function"].isna().sum()
if unmapped:
    print(f"WARNING: {unmapped} rows could not be mapped to a Macro_Function — they will be excluded from 1b/1c/2b.")

df_gold = df_gold[df_gold["Macro_Function"].notna()].reset_index(drop=True)

# Build masked text column for Test 2b
def _mask_particle(text: str, particle: str) -> str:
    if particle == "neutral":
        return text  # already contains [___] or has no particle
    masked, n = re.subn(rf"(?i)\b{re.escape(particle)}\b", "[___]", text, count=1)
    if n == 0:
        masked = re.sub(re.escape(particle), "[___]", text, count=1, flags=re.IGNORECASE)
    return masked

def _get_gt_particle(row) -> str:
    if row["Particle"] in ("kan", "ke", "neutral"):
        return row["Particle"]
    m = re.search(r"removed\s+(\w+)", str(row.get("Sentence_Type", "")), re.IGNORECASE)
    return m.group(1).lower() if m else ""

df_gold["Text_Masked"]  = df_gold.apply(lambda r: _mask_particle(r["Text"], r["Particle"]), axis=1)
df_gold["GT_Particle"]  = df_gold.apply(_get_gt_particle, axis=1)

df_gold = df_gold[df_gold["GT_Particle"].isin(["kan", "ke", "neutral"])].reset_index(drop=True)

print(f"\nGOLD dataset: {len(df_gold)} rows")
print(f"Macro_Function distribution:\n{df_gold['Macro_Function'].value_counts().to_string()}")
print(f"GT Particle distribution:\n{df_gold['GT_Particle'].value_counts().to_string()}")
display(df_gold[["Text", "Macro_Function", "GT_Particle"]].head(3))


# ─────────────────────────────────────────────────────────────────────────────
# LLM call infrastructure  (mirrors round1 scripts)
# ─────────────────────────────────────────────────────────────────────────────

def _call_gemini_legacy(api_key, model_name, prompt_text, constraint_text):
    import google.generativeai as genai_legacy
    genai_legacy.configure(api_key=api_key)
    model = genai_legacy.GenerativeModel(
        model_name=model_name,
        system_instruction=constraint_text,
    )
    response = model.generate_content(
        prompt_text,
        generation_config={"temperature": 0, "max_output_tokens": 64},
        request_options={"timeout": 120},
    )
    return (getattr(response, "text", "") or "").strip()


def _call_gemini_with_fallback(run_cfg, prompt_text, constraint_text):
    last_error = None
    model_candidates = [run_cfg["model"]] + [
        m for m in run_cfg.get("fallback_models", []) if m != run_cfg["model"]
    ]
    for model_name in model_candidates:
        for api_key in run_cfg.get("api_keys", []):
            try:
                text = _call_gemini_legacy(api_key, model_name, prompt_text, constraint_text)
                if text:
                    return text
            except Exception as e:
                last_error = e
            try:
                client = genai.Client(
                    api_key=api_key,
                    http_options=genai_types.HttpOptions(timeout=120),
                )
                response = client.models.generate_content(
                    model=model_name,
                    config=genai_types.GenerateContentConfig(
                        system_instruction=constraint_text,
                        max_output_tokens=64,
                        temperature=0,
                    ),
                    contents=prompt_text,
                )
                raw_text = (response.text or "").strip()
                if raw_text:
                    return raw_text
                last_error = RuntimeError(f"Empty Gemini response for model {model_name}")
            except Exception as e:
                last_error = e
                if _is_fatal_error(e):
                    break
    raise last_error


def _build_client(run_cfg):
    client = run_cfg.get("client")
    if client is not None:
        return client
    provider = run_cfg.get("provider")
    api_key  = run_cfg.get("api_key")
    if provider == "openai_responses":
        return OpenAI(api_key=api_key or OPENAI_API_KEY)
    if provider == "openai":
        return OpenAI(api_key=api_key, base_url=run_cfg.get("base_url"))
    if provider == "anthropic":
        return anthropic.Anthropic(api_key=api_key or ANTHROPIC_API_KEY)
    return None


def _worker_run_cfg(run_cfg):
    """Strip non-serialisable client object for worker threads."""
    cfg = dict(run_cfg)
    cfg.pop("client", None)
    return cfg


def _extract_openai_responses_text(response):
    output_text = (getattr(response, "output_text", "") or "").strip()
    if output_text:
        return output_text
    chunks = []
    for item in (getattr(response, "output", None) or []):
        if getattr(item, "type", None) == "message":
            for content in (getattr(item, "content", None) or []):
                if getattr(content, "type", None) in {"output_text", "text"}:
                    t = (getattr(content, "text", "") or "").strip()
                    if t:
                        chunks.append(t)
    return "\n".join(chunks).strip()


def call_llm(run_cfg, prompt_text, constraint_text, extractor_fn,
             retries=1, delay=1.0, output_try=1, log_meta=None, log_sink=None):
    for attempt in range(retries):
        try:
            provider = run_cfg["provider"]
            model    = run_cfg["model"]
            client   = _build_client(run_cfg)

            if provider == "openai_responses":
                req = {
                    "model": model,
                    "input": [
                        {"role": "system", "content": constraint_text},
                        {"role": "user",   "content": prompt_text},
                    ],
                    "max_output_tokens": max(64, run_cfg.get("max_tokens", 64)),
                }
                if run_cfg.get("reasoning_effort"):
                    req["reasoning"] = {"effort": run_cfg["reasoning_effort"]}
                try:
                    response = client.responses.create(**req)
                except Exception as e:
                    msg = str(e).lower()
                    if "reasoning.effort" in msg and "unsupported" in msg:
                        req.pop("reasoning", None)
                        response = client.responses.create(**req)
                    else:
                        raise
                raw = _extract_openai_responses_text(response)
                if not raw:
                    fb = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": constraint_text},
                            {"role": "user",   "content": prompt_text},
                        ],
                        max_completion_tokens=max(64, run_cfg.get("max_tokens", 128)),
                    )
                    raw = (fb.choices[0].message.content or "").strip()

            elif provider == "openai":
                req = dict(
                    model=model,
                    messages=[
                        {"role": "system", "content": constraint_text},
                        {"role": "user",   "content": prompt_text},
                    ],
                    max_completion_tokens=max(64, run_cfg.get("max_tokens", 64)),
                )
                if run_cfg.get("reasoning_effort"):
                    req["reasoning_effort"] = run_cfg["reasoning_effort"]
                response = client.chat.completions.create(**req)
                raw = (response.choices[0].message.content or "").strip()

            elif provider == "anthropic":
                response = client.messages.create(
                    model=model,
                    system=constraint_text,
                    max_tokens=256,
                    messages=[{"role": "user", "content": prompt_text}],
                )
                raw = response.content[0].text.strip()

            elif provider == "gemini":
                raw = _call_gemini_with_fallback(run_cfg, prompt_text, constraint_text)

            elif provider == "ollama":
                url = f"{run_cfg['base_url'].rstrip('/')}/api/generate"
                resp = requests.post(
                    url,
                    json={
                        "model": model,
                        "prompt": f"{constraint_text}\n\n{prompt_text}",
                        "stream": False,
                        "options": {"temperature": 0},
                    },
                    timeout=60,
                )
                resp.raise_for_status()
                raw = resp.json().get("response", "").strip()

            else:
                raise ValueError(f"Unsupported provider: {provider}")

            label  = extractor_fn(raw)
            status = "ok" if str(label).strip() else "empty"
            _append_io_log({
                "timestamp":    _utc_now_iso(),
                "model_alias":  run_cfg["name"],
                "provider":     provider,
                "model":        model,
                "output_try":   output_try,
                "transport_try": attempt + 1,
                "input":        prompt_text,
                "constraint":   constraint_text,
                "raw_output":   str(raw),
                "parsed_output": str(label),
                "status":       status,
                "meta":         log_meta or {},
            }, log_sink=log_sink)
            if not str(label).strip():
                raise RuntimeError("Empty model response")
            return label

        except Exception as e:
            _append_io_log({
                "timestamp":    _utc_now_iso(),
                "model_alias":  run_cfg["name"],
                "provider":     run_cfg.get("provider"),
                "model":        run_cfg.get("model"),
                "output_try":   output_try,
                "transport_try": attempt + 1,
                "input":        prompt_text,
                "constraint":   constraint_text,
                "raw_output":   "",
                "parsed_output": "",
                "status":       "error",
                "error":        str(e),
                "meta":         log_meta or {},
            }, log_sink=log_sink)
            if _is_fatal_error(e):
                MODEL_FATAL_ERRORS[run_cfg["name"]] = f"ERROR_FATAL: {e}"
                return f"ERROR_FATAL: {e}"
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
            else:
                return f"ERROR: {e}"


def predict_with_output_retry(run_cfg, prompt_text, constraint_text, extractor_fn,
                               max_output_retries=6, delay=0.6,
                               log_meta=None, log_sink=None):
    last_pred = "ERROR: Empty model response"
    for output_try in range(1, max_output_retries + 1):
        pred = call_llm(
            run_cfg, prompt_text, constraint_text, extractor_fn,
            retries=1, delay=0.4, output_try=output_try,
            log_meta=log_meta, log_sink=log_sink,
        )
        if pred and not str(pred).startswith("ERROR"):
            return pred
        last_pred = pred or "ERROR: Empty model response"
        if output_try < max_output_retries:
            if isinstance(last_pred, str) and ("429" in last_pred or "rate_limit" in last_pred.lower()):
                wait = 65 + random.uniform(0, 15)
                print(f"  [rate-limit backoff {wait:.0f}s — {run_cfg['name']}, try {output_try}]", flush=True)
                time.sleep(wait)
            else:
                time.sleep(delay * output_try)
    return last_pred


def _predict_task(task):
    local_logs = []
    pred = predict_with_output_retry(
        task["run_cfg"],
        task["prompt_text"],
        task["constraint_text"],
        task["extractor_fn"],
        max_output_retries=task.get("max_output_retries", 6),
        delay=task.get("delay", 0.6),
        log_meta=task.get("log_meta"),
        log_sink=local_logs,
    )
    return {
        "row_idx":    task["row_idx"],
        "model_name": task["model_name"],
        "prediction": pred,
        "logs":       local_logs,
    }


def _run_parallel(tasks, desc="parallel"):
    if not tasks:
        return []
    _sems = {}
    for t in tasks:
        name = t["model_name"]
        if name not in _sems:
            cap = MODEL_MAX_WORKERS.get(name, PARALLEL_WORKERS)
            _sems[name] = threading.Semaphore(cap)

    def _guarded(task):
        with _sems[task["model_name"]]:
            return _predict_task(task)

    worker_count = min(PARALLEL_WORKERS, len(tasks))
    print(f"Launching {len(tasks)} tasks across {worker_count} threads [{desc}]", flush=True)
    if worker_count <= 1:
        return [
            _predict_task(t)
            for t in tqdm(tasks, total=len(tasks), desc=desc, file=sys.stdout, dynamic_ncols=True)
        ]
    results_list = []
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(_guarded, t) for t in tasks]
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc=desc, file=sys.stdout, dynamic_ncols=True):
            results_list.append(fut.result())
    return results_list


def _build_tasks(eval_df, test_name, prompt_fn, system_text, extractor_fn, col_suffix):
    """Build task list for one test across all active models and all rows."""
    tasks = []
    for run_cfg in MODEL_RUNS:
        if MODEL_FATAL_ERRORS.get(run_cfg["name"]):
            continue
        for row_idx, row in enumerate(eval_df.itertuples(index=False)):
            row_series = eval_df.iloc[row_idx]
            tasks.append({
                "row_idx":           row_idx,
                "model_name":        run_cfg["name"],
                "run_cfg":           _worker_run_cfg(run_cfg),
                "prompt_text":       prompt_fn(row_series),
                "constraint_text":   system_text,
                "extractor_fn":      extractor_fn,
                "max_output_retries": 6,
                "delay":             0.6,
                "log_meta":          {"phase": test_name, "row_idx": row_idx},
                "col_suffix":        col_suffix,
            })
    return tasks


def _apply_parallel_results(eval_df, parallel_results, col_suffix):
    """Write predictions back into a copy of eval_df and return it."""
    result_df = eval_df.copy()
    active_names = list({t["model_name"] for t in parallel_results if "model_name" in t})
    preds_by_model = {name: [None] * len(eval_df) for name in active_names}
    new_logs = []
    for item in parallel_results:
        preds_by_model[item["model_name"]][item["row_idx"]] = item["prediction"]
        new_logs.extend(item["logs"])
    for name in active_names:
        result_df[f"{name}_{col_suffix}"] = preds_by_model[name]
    IO_LOGS.extend(new_logs)
    _save_io_logs()
    return result_df


# ─────────────────────────────────────────────────────────────────────────────
# Per-test run functions
# ─────────────────────────────────────────────────────────────────────────────

def run_test(eval_df, test_name, prompt_fn, system_text, extractor_fn, col_suffix, desc):
    """Generic runner: build tasks, run parallel, return results DataFrame."""
    tasks = _build_tasks(eval_df, test_name, prompt_fn, system_text, extractor_fn, col_suffix)
    # Fill fatal-error models with cached value
    result_df = eval_df.copy()
    for run_cfg in MODEL_RUNS:
        fatal = MODEL_FATAL_ERRORS.get(run_cfg["name"])
        if fatal:
            result_df[f"{run_cfg['name']}_{col_suffix}"] = [fatal] * len(eval_df)
    parallel_results = _run_parallel(tasks, desc=desc)
    result_df = _apply_parallel_results(result_df, parallel_results, col_suffix)
    return result_df


# ─────────────────────────────────────────────────────────────────────────────
# Accuracy + save helpers
# ─────────────────────────────────────────────────────────────────────────────

def accuracy_report(result_df: pd.DataFrame, gt_col: str, col_suffix: str,
                    test_label: str, label_set: list) -> pd.DataFrame:
    gt = result_df[gt_col].str.strip() if result_df[gt_col].dtype == object else result_df[gt_col]
    rows = []
    for m in MODEL_RUNS:
        col = f"{m['name']}_{col_suffix}"
        if col not in result_df.columns:
            continue
        pred   = result_df[col].astype(str).str.strip()
        acc    = (gt == pred).mean()
        errors = pred.str.startswith("ERROR").sum()
        rows.append({
            "Model":    MODEL_VERSION_MAP.get(m["name"].upper(), m["name"].upper()),
            "Alias":    m["name"],
            "Accuracy": round(float(acc), 4),
            "Errors":   int(errors),
        })
    report = pd.DataFrame(rows).sort_values(["Accuracy", "Alias"], ascending=[False, True])
    print(f"\n{'='*60}")
    print(f"  {test_label}")
    print(f"{'='*60}")
    display(report)
    return report


def save_results(result_df: pd.DataFrame, tag: str):
    out = FINAL_METRICS / f"round2_{tag}_predictions.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(out, index=False)
    print(f"Saved → {out.resolve()}")


def save_accuracy_markdown(report_df: pd.DataFrame, tag: str, test_label: str):
    lines = [
        f"# Round 2 — {test_label}",
        "",
        f"Run ID: {RUN_ID}",
        "",
        "| Model | Alias | Accuracy | Errors |",
        "|---|---|---:|---:|",
    ]
    for row in report_df.itertuples(index=False):
        lines.append(f"| {row.Model} | {row.Alias} | {row.Accuracy:.4f} | {int(row.Errors)} |")
    text = "\n".join(lines) + "\n"
    out      = FINAL_METRICS / f"round2_{tag}_accuracy_{RUN_ID}.md"
    out_latest = FINAL_METRICS / f"round2_{tag}_accuracy.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    out_latest.write_text(text, encoding="utf-8")
    print(f"Saved → {out_latest.resolve()}")


def save_confusion_matrices(result_df: pd.DataFrame, gt_col: str, col_suffix: str,
                             tag: str, label_set: list):
    """Save per-model confusion matrices as CSVs (rows=Actual, cols=Predicted)."""
    gt = result_df[gt_col].astype(str).str.strip()
    FINAL_METRICS.mkdir(parents=True, exist_ok=True)
    saved = []
    for m in MODEL_RUNS:
        col = f"{m['name']}_{col_suffix}"
        if col not in result_df.columns:
            continue
        pred = result_df[col].astype(str).str.strip()
        # Only include rows where gt is a known label (exclude any unmapped)
        mask = gt.isin(label_set)
        cm = pd.crosstab(
            gt[mask], pred[mask],
            rownames=["Actual"], colnames=["Predicted"],
        )
        cm = cm.reindex(index=label_set, columns=label_set, fill_value=0)
        out = FINAL_METRICS / f"round2_{tag}_cm_{m['name']}.csv"
        cm.to_csv(out)
        saved.append(out.name)
    if saved:
        print(f"Confusion matrices saved → {FINAL_METRICS}/ [{', '.join(saved)}]")


# ─────────────────────────────────────────────────────────────────────────────
# Error retry pass
# ─────────────────────────────────────────────────────────────────────────────

def retry_errors(result_df: pd.DataFrame, test_name: str,
                 prompt_fn, system_text, extractor_fn, col_suffix: str,
                 gt_col: str, label_set: list, test_label: str) -> pd.DataFrame:
    """Re-run rows that still have ERROR values for any model."""
    retry_tasks = []
    for m in MODEL_RUNS:
        col = f"{m['name']}_{col_suffix}"
        if col not in result_df.columns:
            continue
        if MODEL_FATAL_ERRORS.get(m["name"]):
            continue
        error_mask = result_df[col].astype(str).str.startswith("ERROR:")
        error_indices = result_df.index[error_mask].tolist()
        if not error_indices:
            continue
        print(f"  {m['name']}: {len(error_indices)} ERROR rows to retry", flush=True)
        for df_idx in error_indices:
            row_idx = result_df.index.get_loc(df_idx)
            row_series = result_df.iloc[row_idx]
            retry_tasks.append({
                "row_idx":           row_idx,
                "model_name":        m["name"],
                "run_cfg":           _worker_run_cfg(m),
                "prompt_text":       prompt_fn(row_series),
                "constraint_text":   system_text,
                "extractor_fn":      extractor_fn,
                "max_output_retries": 8,
                "delay":             1.0,
                "log_meta":          {"phase": f"{test_name}_retry", "row_idx": row_idx},
                "col_suffix":        col_suffix,
            })

    if not retry_tasks:
        print(f"  [{test_label}] No ERROR rows to retry.", flush=True)
        return result_df

    print(f"\n{'─'*50}")
    print(f"  Retrying {len(retry_tasks)} error tasks for [{test_label}]")
    print(f"{'─'*50}")
    retry_results = _run_parallel(retry_tasks, desc=f"retry_{col_suffix}")

    for item in retry_results:
        name = item["model_name"]
        col  = f"{name}_{col_suffix}"
        row_idx = item["row_idx"]
        if col in result_df.columns:
            result_df.at[result_df.index[row_idx], col] = item["prediction"]
        IO_LOGS.extend(item["logs"])
    _save_io_logs()

    print(f"  Retry complete. Updated accuracy:")
    accuracy_report(result_df, gt_col, col_suffix, test_label, label_set)
    return result_df


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────

def smoke_test():
    """Run all three tests on a single row to validate connectivity."""
    print("\n" + "=" * 70)
    print("SMOKE TEST — 1 row, all models, all three tests")
    print("=" * 70)

    sample = df_gold.iloc[[0]].copy()
    row    = sample.iloc[0]

    print(f"\n  Text          : {row['Text'][:100]}")
    print(f"  Macro_Function: {row['Macro_Function']}")
    print(f"  GT_Particle   : {row['GT_Particle']}")

    # Test 1b
    print("\n─── Test 1b prompt ─────────────────────────────────────")
    print(build_1b_prompt(row["Text"])[:500] + " ...")

    for test_name, prompt_fn, system_text, extractor_fn, col_suffix in [
        ("1b", lambda r: build_1b_prompt(r["Text"]),
         TEST_1B_SYSTEM, _extract_macro_label, "macro_1b"),
        ("1c", lambda r: build_1c_prompt(r["Text"], r),
         TEST_1C_SYSTEM, _extract_macro_label, "macro_1c"),
        ("2b", lambda r: build_2b_prompt(r["Text_Masked"], r["Macro_Function"]),
         TEST_2B_SYSTEM, _extract_particle, "particle_2b"),
    ]:
        print(f"\n─── Smoke: Test {test_name} ─────────────────────────────────")
        tasks = []
        for run_cfg in MODEL_RUNS:
            if MODEL_FATAL_ERRORS.get(run_cfg["name"]):
                continue
            tasks.append({
                "row_idx":           0,
                "model_name":        run_cfg["name"],
                "run_cfg":           _worker_run_cfg(run_cfg),
                "prompt_text":       prompt_fn(sample.iloc[0]),
                "constraint_text":   system_text,
                "extractor_fn":      extractor_fn,
                "max_output_retries": 3,
                "delay":             0.4,
                "log_meta":          {"phase": f"smoke_{test_name}"},
            })
        smoke_results = _run_parallel(tasks, desc=f"smoke_{test_name}")
        smoke_map = {item["model_name"]: item["prediction"] for item in smoke_results}
        IO_LOGS.extend(log for item in smoke_results for log in item["logs"])
        for m in MODEL_RUNS:
            pred = smoke_map.get(m["name"], "(skipped — fatal error)")
            if isinstance(pred, str) and pred.startswith("ERROR_FATAL"):
                MODEL_FATAL_ERRORS[m["name"]] = pred
            print(f"  {m['name']:22s}: {str(pred)[:80]}", flush=True)

    _save_io_logs()
    print(f"\nSmoke test complete.")


# ─────────────────────────────────────────────────────────────────────────────
# Main execution
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__" or True:

    # ── Step 1: Smoke test ────────────────────────────────────────────────────
    smoke_test()

    if os.getenv("STOP_AFTER_SMOKE", "0") == "1":
        _save_io_logs()
        print("STOP_AFTER_SMOKE=1 — exiting after smoke test.")
        raise SystemExit(0)

    eval_df = df_gold.copy()

    # ── Step 2: Test 1b — Macro-Function, Unassisted ─────────────────────────
    if os.getenv("SKIP_1B", "0") != "1":
        print("\n" + "=" * 70)
        print("STEP 2 — Test 1b: Macro-Function Classification (Unassisted)")
        print("=" * 70)

        results_1b = run_test(
            eval_df,
            test_name  = "test_1b",
            prompt_fn  = lambda r: build_1b_prompt(r["Text"]),
            system_text = TEST_1B_SYSTEM,
            extractor_fn = _extract_macro_label,
            col_suffix = "macro_1b",
            desc       = "test_1b",
        )
        report_1b = accuracy_report(
            results_1b, "Macro_Function", "macro_1b",
            "Test 1b — Macro-Function Unassisted", MACRO_FUNCTION_LABELS,
        )
        save_results(results_1b, "1b")
        save_accuracy_markdown(report_1b, "1b", "Test 1b — Macro-Function Unassisted")
        save_confusion_matrices(results_1b, "Macro_Function", "macro_1b", "1b", MACRO_FUNCTION_LABELS)

        if os.getenv("STOP_AFTER_1B", "0") == "1":
            _save_io_logs()
            print("STOP_AFTER_1B=1 — exiting after Test 1b.")
            raise SystemExit(0)
    else:
        print("SKIP_1B=1 — skipping Test 1b.")

    # ── Step 3: Test 1c — Macro-Function, Attribute-Assisted ─────────────────
    print("\n" + "=" * 70)
    print("STEP 3 — Test 1c: Macro-Function Classification (Attribute-Assisted)")
    print("=" * 70)

    results_1c = run_test(
        eval_df,
        test_name  = "test_1c",
        prompt_fn  = lambda r: build_1c_prompt(r["Text"], r),
        system_text = TEST_1C_SYSTEM,
        extractor_fn = _extract_macro_label,
        col_suffix = "macro_1c",
        desc       = "test_1c",
    )
    report_1c = accuracy_report(
        results_1c, "Macro_Function", "macro_1c",
        "Test 1c — Macro-Function Attribute-Assisted", MACRO_FUNCTION_LABELS,
    )
    save_results(results_1c, "1c")
    save_accuracy_markdown(report_1c, "1c", "Test 1c — Macro-Function Attribute-Assisted")
    save_confusion_matrices(results_1c, "Macro_Function", "macro_1c", "1c", MACRO_FUNCTION_LABELS)

    # ── Step 4: Test 2b — Function-Constrained Particle Generation ───────────
    print("\n" + "=" * 70)
    print("STEP 4 — Test 2b: Function-Constrained Particle Generation")
    print("=" * 70)

    results_2b = run_test(
        eval_df,
        test_name  = "test_2b",
        prompt_fn  = lambda r: build_2b_prompt(r["Text_Masked"], r["Macro_Function"]),
        system_text = TEST_2B_SYSTEM,
        extractor_fn = _extract_particle,
        col_suffix = "particle_2b",
        desc       = "test_2b",
    )
    report_2b = accuracy_report(
        results_2b, "GT_Particle", "particle_2b",
        "Test 2b — Function-Constrained Particle Generation", PARTICLE_LABELS,
    )
    save_results(results_2b, "2b")
    save_accuracy_markdown(report_2b, "2b", "Test 2b — Function-Constrained Particle Generation")
    save_confusion_matrices(results_2b, "GT_Particle", "particle_2b", "2b", PARTICLE_LABELS)

    # ── Step 5: Error retry pass ──────────────────────────────────────────────
    if os.getenv("SKIP_RETRY", "0") != "1":
        print("\n" + "=" * 70)
        print("STEP 5 — Error retry pass")
        print("=" * 70)

        results_1b = retry_errors(
            results_1b, "test_1b",
            lambda r: build_1b_prompt(r["Text"]),
            TEST_1B_SYSTEM, _extract_macro_label, "macro_1b",
            "Macro_Function", MACRO_FUNCTION_LABELS,
            "Test 1b — Macro-Function Unassisted",
        )
        save_results(results_1b, "1b")
        save_confusion_matrices(results_1b, "Macro_Function", "macro_1b", "1b", MACRO_FUNCTION_LABELS)

        results_1c = retry_errors(
            results_1c, "test_1c",
            lambda r: build_1c_prompt(r["Text"], r),
            TEST_1C_SYSTEM, _extract_macro_label, "macro_1c",
            "Macro_Function", MACRO_FUNCTION_LABELS,
            "Test 1c — Macro-Function Attribute-Assisted",
        )
        save_results(results_1c, "1c")
        save_confusion_matrices(results_1c, "Macro_Function", "macro_1c", "1c", MACRO_FUNCTION_LABELS)

        results_2b = retry_errors(
            results_2b, "test_2b",
            lambda r: build_2b_prompt(r["Text_Masked"], r["Macro_Function"]),
            TEST_2B_SYSTEM, _extract_particle, "particle_2b",
            "GT_Particle", PARTICLE_LABELS,
            "Test 2b — Function-Constrained Particle Generation",
        )
        save_results(results_2b, "2b")
        save_confusion_matrices(results_2b, "GT_Particle", "particle_2b", "2b", PARTICLE_LABELS)

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ROUND 2 COMPLETE")
    print("=" * 70)
    print(f"Run ID  : {RUN_ID}")
    print(f"IO logs : {IO_LOG_JSON_LATEST.resolve()}")
    print(f"Outputs :")
    for tag in ["1b", "1c", "2b"]:
        p = FINAL_METRICS / f"round2_{tag}_predictions.csv"
        if p.exists():
            print(f"  {p.resolve()}")
