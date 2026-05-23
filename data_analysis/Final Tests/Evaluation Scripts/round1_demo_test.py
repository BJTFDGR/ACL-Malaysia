# %% [markdown]
# # Round 1 Benchmarking — Test 1a: Attribute Accuracy
# 
# Predict all **5 discourse-context micro-tags** for each sentence across multiple LLMs.
# Each attribute is tested in its own cell using a structured zero-shot prompt.
# 
# | Setting | Value |
# |---|---|
# | Dataset | GOLD_187.csv |
# | Task | Attribute Accuracy (Test 1a) |
# | Models | GPT, Claude, Gemini, DeepSeek, SEA-LION (+ optional local Llama) |
# | Attributes | Epistemic_Stance, Particle_Position, Listener_Agreement, Emotion, Question_Type |

# %% [markdown]
# Test the most and least powerful models on this task, 
# 
# GPT (general ): [REDACTED]
# 
# Claude: [REDACTED]
# 
# Gemini: [REDACTED]
# (如果上面的gemini api到了限度，可以用这个) [REDACTED]
# 
# DeepSeek: sk-170c271f55974de080f05c36b0440646
# Llama: 1b, 3b and 7b insturction-tuned models 
# SEA-LION: sk-zMRmceahO6cvFeksiwtW0A
# 

# %%
import json
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
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

IO_LOGS = []
RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
IO_LOG_JSON = Path(f"../Final Metrics/round1_1a_io_logs_{RUN_ID}.json")
IO_LOG_JSON_LATEST = Path("../Final Metrics/round1_1a_io_logs.json")
PARALLEL_WORKERS = int(os.getenv("PARALLEL_WORKERS", "50"))
MODEL_FATAL_ERRORS = {}
MS_ONLY = os.getenv("MS_ONLY", "0") == "1"

ATTRIBUTE_LABEL_MAP = {
    "Epistemic_Stance": ["Certain", "Uncertain", "Neutral/Unclear", "Neutral / NA"],
    "Particle_Position": ["Front", "Middle/End", "N/A"],
    "Listener_Agreement": ["Assumed Agreement", "Confirmation Seeking", "Neutral/Unclear"],
    "Emotion": ["Positive", "Negative", "Neutral/Unclear"],
    "Question_Type": ["Declarative/Statement", "Rhetorical Interrogative", "Yes/No Interrogative"],
}

ATTRIBUTE_ALIASES = {
    "Epistemic_Stance": {
        "pasti": "Certain",
        "tidak pasti": "Uncertain",
        "neutral/na": "Neutral/Unclear",
        "neutral / na": "Neutral/Unclear",
        "neutral": "Neutral/Unclear",
    },
    "Particle_Position": {
        "hadapan": "Front",
        "depan": "Front",
        "tengah/akhir": "Middle/End",
        "tengah atau akhir": "Middle/End",
        "tiada": "N/A",
        "tidak ada": "N/A",
    },
    "Listener_Agreement": {
        "anggapan persetujuan": "Assumed Agreement",
        "persetujuan diandaikan": "Assumed Agreement",
        "mencari pengesahan": "Confirmation Seeking",
        "neutral/tidak jelas": "Neutral/Unclear",
        "neutral": "Neutral/Unclear",
    },
    "Emotion": {
        "positif": "Positive",
        "negatif": "Negative",
        "neutral/tidak jelas": "Neutral/Unclear",
        "neutral": "Neutral/Unclear",
    },
    "Question_Type": {
        "deklaratif/pernyataan": "Declarative/Statement",
        "soalan retorik": "Rhetorical Interrogative",
        "tanya jawab retorik": "Rhetorical Interrogative",
        "soalan ya/tidak": "Yes/No Interrogative",
        "tanya jawab ya/tidak": "Yes/No Interrogative",
    },
}


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


def _normalize_for_scoring(val, label_set, attribute=None):
    if pd.isna(val):
        return val
    raw = str(val).strip()
    # Strip BPE tokenizer artifacts (Ġ=space token U+0120, Ċ=newline token U+010A)
    raw = raw.replace('\u0120', ' ').replace('\u010a', '\n')
    # Extract label from verbose chain-of-thought / <think> output
    if "</think>" in raw:
        raw = raw[raw.rfind("</think>") + len("</think>"):].strip()
    # Strip markdown bold/italic formatting
    import re as _re
    raw = _re.sub(r'\*+', '', raw).strip()
    lowered = raw.lower()

    if lowered in {"na", "n/a", "n.a.", "none", "tiada", "tidak ada"} and "N/A" in label_set:
        return "N/A"

    for lbl in sorted(label_set, key=len, reverse=True):
        if lbl.lower() == lowered:
            return lbl

    if attribute in ATTRIBUTE_ALIASES:
        for alias, canonical in ATTRIBUTE_ALIASES[attribute].items():
            if alias == lowered:
                return canonical

    # Substring match for verbose outputs that embed the label in explanatory text
    for lbl in sorted(label_set, key=len, reverse=True):
        if lbl.lower() in lowered:
            return lbl
    if attribute in ATTRIBUTE_ALIASES:
        for alias, canonical in sorted(ATTRIBUTE_ALIASES[attribute].items(), key=lambda x: -len(x[0])):
            if alias in lowered:
                return canonical

    return raw


def _attribute_accuracy_rows(attribute, label_set, col_suffix=""):
    gt = results[attribute].apply(lambda x: _normalize_for_scoring(x, label_set, attribute))
    rows = []
    for run_cfg in MODEL_RUNS:
        col = f"{run_cfg['name']}_{attribute}{col_suffix}"
        if col not in results.columns:
            continue
        pred = results[col].apply(lambda x: _normalize_for_scoring(x, label_set, attribute))
        rows.append({
            "Model": MODEL_VERSION_MAP.get(run_cfg["name"].upper(), run_cfg["name"].upper()),
            "Accuracy": round(float((gt == pred).mean()), 4),
            "Errors": int(pred.astype(str).str.startswith("ERROR").sum()),
        })
    return rows


def _print_attribute_report(stage_label, attribute, label_set, col_suffix=""):
    rows = _attribute_accuracy_rows(attribute, label_set, col_suffix=col_suffix)
    print(f"[{stage_label}] Attribute report for {attribute}", flush=True)
    if not rows:
        print(f"[{stage_label}] No completed prediction columns found yet.", flush=True)
        return
    report_df = pd.DataFrame(rows).sort_values(["Accuracy", "Model"], ascending=[False, True])
    display(report_df)
    best_row = report_df.iloc[0]
    print(
        f"[{stage_label}] Best accuracy so far: {best_row['Model']} = {best_row['Accuracy']:.4f}",
        flush=True,
    )


def _accuracy_markdown_text():
    sections = []
    attributes = [
        "Epistemic_Stance",
        "Particle_Position",
        "Listener_Agreement",
        "Emotion",
        "Question_Type",
    ]
    variant_suffixes = [("EN", ""), ("MS", "_ms")]
    alias_to_model = {run_cfg["name"]: run_cfg["model"] for run_cfg in MODEL_RUNS}
    model_rows = list(dict.fromkeys(alias_to_model.values()))

    for variant_name, suffix in variant_suffixes:
        rows = []
        for model_name in model_rows:
            alias = next((name for name, version in alias_to_model.items() if version == model_name), None)
            row = [model_name]
            for attr in attributes:
                col = f"{alias}_{attr}{suffix}" if alias else ""
                if col in results.columns:
                    gt = results[attr].apply(lambda x: _normalize_for_scoring(x, ATTRIBUTE_LABEL_MAP[attr], attr))
                    pred = results[col].apply(lambda x: _normalize_for_scoring(x, ATTRIBUTE_LABEL_MAP[attr], attr))
                    row.append(f"{(gt == pred).mean():.4f}")
                else:
                    row.append("")
            rows.append(row)

        header = "| Model | " + " | ".join(attributes) + " |"
        separator = "|---|" + "---|" * len(attributes)
        table_lines = [header, separator]
        for row in rows:
            table_lines.append("| " + " | ".join(row) + " |")

        sections.append(f"## {variant_name}\n\n" + "\n".join(table_lines))

    return "\n\n".join(sections).rstrip() + "\n"


def _write_accuracy_markdown():
    md_path = Path("../Final Metrics/round1_1a_accuracy_summary.md")
    md_path.write_text(_accuracy_markdown_text(), encoding="utf-8")


def _probe_ollama(base_url, timeout=2):
    try:
        tags_url = f"{base_url.rstrip('/')}/api/tags"
        resp = requests.get(tags_url, timeout=timeout)
        return resp.ok
    except Exception:
        return False


def _ensure_ollama_runtime(base_url):
    """Try to ensure Ollama is reachable; auto-start local server when possible."""
    if _probe_ollama(base_url):
        return True, "Ollama endpoint is reachable."

    parsed = urlparse(base_url)
    host = (parsed.hostname or "").lower()
    is_local_target = host in {"localhost", "127.0.0.1", "::1"}

    ollama_cmd = shutil.which("ollama")
    if not ollama_cmd:
        install_cmd = "curl -fsSL https://ollama.com/install.sh | sh"
        return (
            False,
            "Ollama is not installed. Install it with: "
            f"{install_cmd} ; then run: ollama serve ; and pull models: "
            f"ollama pull {LLAMA_STRONG_MODEL} && ollama pull {LLAMA_WEAK_MODEL}",
        )

    if not is_local_target:
        return False, f"Ollama endpoint not reachable at non-local target: {base_url}"

    try:
        subprocess.Popen(
            [ollama_cmd, "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except Exception as exc:
        return False, f"Failed to start 'ollama serve': {exc}"

    for _ in range(12):
        time.sleep(1)
        if _probe_ollama(base_url):
            return True, "Ollama server auto-started successfully."

    return False, f"Tried to auto-start Ollama but endpoint is still unreachable at {base_url}"


def _extract_keys_from_notebook(nb_path: Path):
    """Parse markdown notes cell for API keys."""
    keys = {
        "openai": None,
        "anthropic": None,
        "gemini": [],
        "deepseek": None,
        "sealion": None,
    }
    try:
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        text = "\n".join(
            "\n".join(cell.get("source", []))
            for cell in nb.get("cells", [])
            if cell.get("cell_type") == "markdown"
        )
        patterns = {
            "openai": r"GPT\s*\(general\s*\)\s*:\s*(sk-[A-Za-z0-9_\-]+)",
            "anthropic": r"Claude\s*:\s*(sk-ant-[A-Za-z0-9_\-]+)",
            "gemini": r"Gemini\s*:\s*([A-Za-z0-9_\-]+)",
            "deepseek": r"DeepSeek\s*:\s*(sk-[A-Za-z0-9_\-]+)",
            "sealion": r"SEA-LION\s*:\s*(sk-[A-Za-z0-9_\-]+)",
        }
        for k, pat in patterns.items():
            m = re.search(pat, text)
            if m and k != "gemini":
                keys[k] = m.group(1).strip()
        keys["gemini"] = re.findall(r"AIza[0-9A-Za-z_\-]+", text)
    except Exception:
        pass
    return keys


# ── Keys and clients ──────────────────────────────────────────────────────────
notebook_path = Path("04_round1_1a_attribute_accuracy.ipynb")
fallback = _extract_keys_from_notebook(notebook_path)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or fallback["openai"]
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY") or fallback["anthropic"]
GEMINI_API_KEYS = [key for key in [os.getenv("GEMINI_API_KEY")] + fallback["gemini"] if key]
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY") or fallback["deepseek"]
SEA_LION_API_KEY = os.getenv("SEA_LION_API_KEY") or fallback["sealion"]

# Optional runtime for local llama via Ollama.
LLAMA_BASE_URL = os.getenv("LLAMA_BASE_URL", "http://localhost:11434")
LLAMA_STRONG_MODEL = os.getenv("LLAMA_STRONG_MODEL", "llama3.2:7b")
LLAMA_WEAK_MODEL = os.getenv("LLAMA_WEAK_MODEL", "llama3.2:1b")

# Optional SEA-LION endpoint if OpenAI-compatible.
SEA_LION_BASE_URL = os.getenv("SEA_LION_BASE_URL", "https://api.sea-lion.ai/v1")
SEA_LION_STRONG_MODEL = os.getenv("SEA_LION_STRONG_MODEL", "aisingapore/Llama-SEA-LION-v3.5-70B-R")
SEA_LION_WEAK_MODEL = os.getenv("SEA_LION_WEAK_MODEL", "aisingapore/Gemma-SEA-LION-v4-27B-IT")
GPT_STRONG_MODEL = os.getenv("GPT_STRONG_MODEL", "gpt-5")
GPT_WEAK_MODEL = os.getenv("GPT_WEAK_MODEL", "gpt-5.4-mini")
GPT_STRONG_REASONING_EFFORT = os.getenv("GPT_STRONG_REASONING_EFFORT", "none")
GPT_WEAK_REASONING_EFFORT = os.getenv("GPT_WEAK_REASONING_EFFORT", "none")
DEEPSEEK_STRONG_MODEL = os.getenv("DEEPSEEK_STRONG_MODEL", "deepseek-v4-pro")
DEEPSEEK_WEAK_MODEL = os.getenv("DEEPSEEK_WEAK_MODEL", "deepseek-v4-flash")
CLAUDE_STRONG_MODEL = os.getenv("CLAUDE_STRONG_MODEL", "claude-sonnet-4-6")
CLAUDE_WEAK_MODEL = os.getenv("CLAUDE_WEAK_MODEL", "claude-haiku-4-5")
GEMINI_STRONG_MODEL = os.getenv("GEMINI_STRONG_MODEL", "gemini-3.1-pro-preview")
GEMINI_WEAK_MODEL = os.getenv("GEMINI_WEAK_MODEL", "gemini-3.1-flash-lite")
GEMINI_STRONG_FALLBACK_MODELS = [
    m.strip()
    for m in os.getenv("GEMINI_STRONG_FALLBACK_MODELS", "gemini-3.1-flash-lite").split(",")
    if m.strip()
]

MODEL_RUNS = []

if OPENAI_API_KEY:
    MODEL_RUNS.append({
        "name": "gpt_strong",
        "provider": "openai_responses",
        "client": OpenAI(api_key=OPENAI_API_KEY),
        "api_key": OPENAI_API_KEY,
        "model": GPT_STRONG_MODEL,
        "max_tokens": 64,
        "reasoning_effort": GPT_STRONG_REASONING_EFFORT,
        "sleep": 0.5,
    }),
    MODEL_RUNS.append({
        "name": "gpt_weak",
        "provider": "openai_responses",
        "client": OpenAI(api_key=OPENAI_API_KEY),
        "api_key": OPENAI_API_KEY,
        "model": GPT_WEAK_MODEL,
        "max_tokens": 64,
        "reasoning_effort": GPT_WEAK_REASONING_EFFORT,
        "sleep": 0.5,
    })

if ANTHROPIC_API_KEY:
    MODEL_RUNS.append({
        "name": "claude_strong",
        "provider": "anthropic",
        "client": anthropic.Anthropic(api_key=ANTHROPIC_API_KEY),
        "api_key": ANTHROPIC_API_KEY,
        "model": CLAUDE_STRONG_MODEL,
        "sleep": 0.5,
    })
    MODEL_RUNS.append({
        "name": "claude_weak",
        "provider": "anthropic",
        "client": anthropic.Anthropic(api_key=ANTHROPIC_API_KEY),
        "api_key": ANTHROPIC_API_KEY,
        "model": CLAUDE_WEAK_MODEL,
        "sleep": 0.5,
    })

if GEMINI_API_KEYS:
    MODEL_RUNS.append({
        "name": "gemini_strong",
        "provider": "gemini",
        "client": None,
        "api_keys": GEMINI_API_KEYS,
        "model": GEMINI_STRONG_MODEL,
        "fallback_models": GEMINI_STRONG_FALLBACK_MODELS,
        "sleep": 0.5,
    })
    MODEL_RUNS.append({
        "name": "gemini_weak",
        "provider": "gemini",
        "client": None,
        "api_keys": GEMINI_API_KEYS,
        "model": GEMINI_WEAK_MODEL,
        "sleep": 0.5,
    })

if DEEPSEEK_API_KEY:
    MODEL_RUNS.append({
        "name": "deepseek_strong",
        "provider": "openai",
        "client": OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com"),
        "api_key": DEEPSEEK_API_KEY,
        "base_url": "https://api.deepseek.com",
        "model": DEEPSEEK_STRONG_MODEL,
        "sleep": 0.3,
    })
    MODEL_RUNS.append({
        "name": "deepseek_weak",
        "provider": "openai",
        "client": OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com"),
        "api_key": DEEPSEEK_API_KEY,
        "base_url": "https://api.deepseek.com",
        "model": DEEPSEEK_WEAK_MODEL,
        "sleep": 0.3,
    })

llama_available, llama_status = _ensure_ollama_runtime(LLAMA_BASE_URL)
print("Llama/Ollama status:", llama_status)

if llama_available:
    MODEL_RUNS.append({
        "name": "llama_strong",
        "provider": "ollama",
        "client": None,
        "model": LLAMA_STRONG_MODEL,
        "sleep": 0.2,
        "base_url": LLAMA_BASE_URL,
    })
    MODEL_RUNS.append({
        "name": "llama_weak",
        "provider": "ollama",
        "client": None,
        "model": LLAMA_WEAK_MODEL,
        "sleep": 0.2,
        "base_url": LLAMA_BASE_URL,
    })
else:
    print("Llama skipped:", llama_status)

if SEA_LION_API_KEY:
    MODEL_RUNS.append({
        "name": "sealion_strong",
        "provider": "openai",
        "client": OpenAI(api_key=SEA_LION_API_KEY, base_url=SEA_LION_BASE_URL),
        "api_key": SEA_LION_API_KEY,
        "base_url": SEA_LION_BASE_URL,
        "model": SEA_LION_STRONG_MODEL,
        "max_tokens": 512,
        "sleep": 0.3,
    })
    MODEL_RUNS.append({
        "name": "sealion_weak",
        "provider": "openai",
        "client": OpenAI(api_key=SEA_LION_API_KEY, base_url=SEA_LION_BASE_URL),
        "api_key": SEA_LION_API_KEY,
        "base_url": SEA_LION_BASE_URL,
        "model": SEA_LION_WEAK_MODEL,
        "sleep": 0.3,
    })

if not MODEL_RUNS:
    raise ValueError("No model keys/runtimes configured.")

print("Models to run:", [m["name"] for m in MODEL_RUNS])
MODEL_VERSION_MAP = {m["name"].upper(): m["model"] for m in MODEL_RUNS}

# ── Load data ─────────────────────────────────────────────────────────────────
DATA_PATH = Path("../Datasets/GOLD_187.csv")
df = pd.read_csv(DATA_PATH)

# Pilot mode: default run on 5 samples; override with env SAMPLE_N.
SAMPLE_N = int(os.getenv("SAMPLE_N", "5"))
eval_df = df.head(SAMPLE_N).copy() if SAMPLE_N else df.copy()
results = eval_df.copy()

if MS_ONLY:
    _existing_csv = Path("../Final Metrics/round1_1a_predictions.csv")
    if _existing_csv.exists():
        _existing = pd.read_csv(_existing_csv)
        for col in _existing.columns:
            if col not in results.columns and len(_existing) == len(results):
                results[col] = _existing[col].values
        print(f"MS_ONLY=1: merged {len(_existing.columns)} columns from {_existing_csv}")
    else:
        print(f"MS_ONLY=1: no existing CSV at {_existing_csv}, starting fresh")

print(f"Loaded {len(df)} total rows; evaluating {len(eval_df)} rows")
display(eval_df.head(3))

# %%
import requests


def _extract_label(raw, label_set):
    text = str(raw or "").strip()
    for label in sorted(label_set, key=len, reverse=True):
        if label.lower() in text.lower():
            return label
    return text


def _is_fatal_error(error):
    text = str(error).lower()
    fatal_markers = [
        "model_not_found",
        "does not exist",
        "unsupported parameter",
        "unsupported value",
        "invalid_request_error",
        "api key not valid",
        "permission",
        "authentication",
        "insufficient_quota",
        "quota",
    ]
    return any(marker in text for marker in fatal_markers)


def _is_transient_error(error):
    text = str(error).lower()
    transient_markers = [
        "503",
        "service unavailable",
        "unavailable",
        "high demand",
        "resource_exhausted",
        "quota",
        "429",
        "deadline",
        "timeout",
        "internal",
    ]
    return any(marker in text for marker in transient_markers)


def _call_gemini_legacy(api_key, model_name, prompt_text, constraint_text):
    import google.generativeai as genai_legacy

    genai_legacy.configure(api_key=api_key)
    model = genai_legacy.GenerativeModel(
        model_name=model_name,
        system_instruction=constraint_text,
    )
    response = model.generate_content(
        prompt_text,
        generation_config={
            "temperature": 0,
            "max_output_tokens": 16,
        },
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
                legacy_text = _call_gemini_legacy(api_key, model_name, prompt_text, constraint_text)
                if legacy_text:
                    return legacy_text
            except Exception as legacy_error:
                last_error = legacy_error
            try:
                client = genai.Client(
                    api_key=api_key,
                    http_options=genai_types.HttpOptions(timeout=120),
                )
                response = client.models.generate_content(
                    model=model_name,
                    config=genai_types.GenerateContentConfig(
                        system_instruction=constraint_text,
                        max_output_tokens=16,
                        temperature=0,
                    ),
                    contents=prompt_text,
                )
                raw_text = (response.text or "").strip()
                if raw_text:
                    return raw_text
                last_error = RuntimeError(f"Empty Gemini response for model {model_name}")
                continue
            except Exception as error:
                last_error = error
                error_text = str(error).lower()
                if "ssl" in error_text or "handshake" in error_text or "timed out" in error_text:
                    try:
                        legacy_text = _call_gemini_legacy(api_key, model_name, prompt_text, constraint_text)
                        if legacy_text:
                            return legacy_text
                    except Exception as legacy_error:
                        last_error = legacy_error
                if _is_fatal_error(error):
                    break
                if _is_transient_error(error):
                    continue
                continue
    raise last_error


def _build_client(run_cfg):
    client = run_cfg.get("client")
    if client is not None:
        return client

    provider = run_cfg.get("provider")
    api_key = run_cfg.get("api_key")

    if provider == "openai_responses":
        return OpenAI(api_key=api_key or OPENAI_API_KEY)
    if provider == "openai":
        return OpenAI(api_key=api_key, base_url=run_cfg.get("base_url"))
    if provider == "anthropic":
        return anthropic.Anthropic(api_key=api_key or ANTHROPIC_API_KEY)
    return None


def _worker_run_cfg(run_cfg):
    worker_cfg = dict(run_cfg)
    worker_cfg.pop("client", None)
    return worker_cfg


def _extract_openai_responses_text(response):
    raw = (getattr(response, "output_text", "") or "").strip()
    if raw:
        return raw

    try:
        payload = response.model_dump()
    except Exception:
        return ""

    for item in payload.get("output", []) or []:
        if item.get("type") != "message":
            continue
        for content_item in item.get("content", []) or []:
            if content_item.get("type") == "output_text":
                txt = (content_item.get("text") or "").strip()
                if txt:
                    return txt
    return ""


def call_llm(
    run_cfg,
    prompt_text,
    label_set,
    constraint_text,
    retries=1,
    delay=1.0,
    output_try=1,
    log_meta=None,
    log_sink=None,
):
    """Call one configured model and return the best-matching label from label_set."""
    constraint = constraint_text

    for attempt in range(retries):
        try:
            provider = run_cfg["provider"]
            model = run_cfg["model"]
            client = _build_client(run_cfg)

            if provider == "openai_responses":
                req = {
                    "model": model,
                    "input": [
                        {"role": "system", "content": constraint},
                        {"role": "user", "content": prompt_text},
                    ],
                    "max_output_tokens": run_cfg.get("max_tokens", 64),
                }
                reasoning_effort = run_cfg.get("reasoning_effort")
                if reasoning_effort:
                    req["reasoning"] = {"effort": reasoning_effort}
                response = client.responses.create(**req)
                raw = _extract_openai_responses_text(response)

            elif provider == "openai":
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": constraint},
                        {"role": "user", "content": prompt_text},
                    ],
                    max_completion_tokens=run_cfg.get("max_tokens", 32),
                )
                raw = (response.choices[0].message.content or "").strip()

            elif provider == "anthropic":
                response = client.messages.create(
                    model=model,
                    system=constraint,
                    max_tokens=12,
                    messages=[{"role": "user", "content": prompt_text}],
                )
                raw = response.content[0].text.strip()

            elif provider == "gemini":
                raw = _call_gemini_with_fallback(run_cfg, prompt_text, constraint)

            elif provider == "ollama":
                url = f"{run_cfg['base_url'].rstrip('/')}/api/generate"
                response = requests.post(
                    url,
                    json={
                        "model": model,
                        "prompt": f"{constraint}\n\n{prompt_text}",
                        "stream": False,
                        "options": {"temperature": 0},
                    },
                    timeout=60,
                )
                response.raise_for_status()
                raw = response.json().get("response", "").strip()

            else:
                raise ValueError(f"Unsupported provider: {provider}")

            label = _extract_label(raw, label_set)
            status = "ok" if str(label).strip() else "empty"
            _append_io_log({
                "timestamp": _utc_now_iso(),
                "model_alias": run_cfg["name"],
                "provider": provider,
                "model": model,
                "output_try": output_try,
                "transport_try": attempt + 1,
                "input": prompt_text,
                "constraint": constraint,
                "raw_output": str(raw),
                "parsed_output": str(label),
                "status": status,
                "meta": log_meta or {},
            }, log_sink=log_sink)
            if not str(label).strip():
                raise RuntimeError("Empty model response")
            return label

        except Exception as e:
            _append_io_log({
                "timestamp": _utc_now_iso(),
                "model_alias": run_cfg["name"],
                "provider": run_cfg.get("provider"),
                "model": run_cfg.get("model"),
                "output_try": output_try,
                "transport_try": attempt + 1,
                "input": prompt_text,
                "constraint": constraint,
                "raw_output": "",
                "parsed_output": "",
                "status": "error",
                "error": str(e),
                "meta": log_meta or {},
            }, log_sink=log_sink)
            if _is_fatal_error(e):
                return f"ERROR_FATAL: {e}"
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
            else:
                return f"ERROR: {e}"


def predict_with_output_retry(
    run_cfg,
    prompt_text,
    label_set,
    constraint_text,
    max_output_retries=3,
    delay=0.6,
    log_meta=None,
    log_sink=None,
):
    """Retry up to max_output_retries when model output is empty or error."""
    last_pred = "ERROR: Empty model response"
    for output_try in range(1, max_output_retries + 1):
        pred = call_llm(
            run_cfg,
            prompt_text,
            label_set,
            constraint_text,
            retries=1,
            delay=0.4,
            output_try=output_try,
            log_meta=log_meta,
            log_sink=log_sink,
        )
        if pred and not pred.startswith("ERROR"):
            return pred
        last_pred = pred or "ERROR: Empty model response"
        if output_try < max_output_retries:
            time.sleep(delay * output_try)
    return last_pred


def _predict_task(task):
    local_logs = []
    pred = predict_with_output_retry(
        task["run_cfg"],
        task["prompt_text"],
        task["label_set"],
        task["constraint_text"],
        max_output_retries=task.get("max_output_retries", 3),
        delay=task.get("delay", 0.6),
        log_meta=task.get("log_meta"),
        log_sink=local_logs,
    )
    return {
        "row_idx": task["row_idx"],
        "model_name": task["model_name"],
        "prediction": pred,
        "logs": local_logs,
    }


def _run_parallel_predictions(tasks, desc="parallel"):
    if not tasks:
        return []

    worker_count = min(PARALLEL_WORKERS, len(tasks))
    if worker_count <= 1:
        return [
            _predict_task(task)
            for task in tqdm(tasks, total=len(tasks), desc=desc, file=sys.stdout, dynamic_ncols=True)
        ]

    results = []
    print(f"Launching {len(tasks)} tasks across {worker_count} subprocesses", flush=True)
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(_predict_task, task) for task in tasks]
        for future in tqdm(as_completed(futures), total=len(futures), desc=desc, file=sys.stdout, dynamic_ncols=True):
            results.append(future.result())
    return results


def _language_controls(label_set, lang="en"):
    if lang == "ms":
        constraint = "Anda mesti mengeluarkan tepat satu label daripada set label yang diberi dan tiada teks lain."
        strict_tail = "\n\nPulangkan tepat satu label daripada set ini dan tiada teks lain: " + ", ".join(label_set)
    else:
        constraint = "You must output exactly one label from the provided set and nothing else."
        strict_tail = "\n\nReturn exactly one label from this set and nothing else: " + ", ".join(label_set)
    return constraint, strict_tail


def run_attribute(attribute, label_set, prompt_template, col_suffix="", lang="en"):
    """Run all configured models for one attribute and append prediction columns to `results`."""
    print(f"\n{'─'*60}", flush=True)
    print(f"  Attribute : {attribute}", flush=True)
    print(f"  Labels    : {label_set}", flush=True)
    if col_suffix:
        print(f"  Variant   : {col_suffix}", flush=True)
    print(f"{'─'*60}", flush=True)

    constraint, strict_tail = _language_controls(label_set, lang=lang)

    tasks = []
    active_model_names = []
    for run_cfg in MODEL_RUNS:
        worker_cfg = _worker_run_cfg(run_cfg)
        fatal_note = MODEL_FATAL_ERRORS.get(run_cfg["name"])
        if fatal_note:
            results[f"{run_cfg['name']}_{attribute}{col_suffix}"] = [fatal_note] * len(eval_df)
            print(f"  {run_cfg['name'].upper()} skipped → cached fatal error from smoke test", flush=True)
            continue
        active_model_names.append(run_cfg["name"])
        for row_idx, row in enumerate(eval_df.itertuples(index=False)):
            tasks.append({
                "row_idx": row_idx,
                "model_name": run_cfg["name"],
                "run_cfg": worker_cfg,
                "prompt_text": prompt_template.format(TEXT=row.Text) + strict_tail,
                "label_set": label_set,
                "constraint_text": constraint,
                "max_output_retries": 3,
                "delay": 0.6,
                "log_meta": {
                    "phase": "run_attribute",
                    "attribute": attribute,
                    "lang": lang,
                },
            })

    parallel_results = _run_parallel_predictions(
        tasks,
        desc=f"{attribute}{col_suffix or ''}",
    )
    preds_by_model = {model_name: [None] * len(eval_df) for model_name in active_model_names}
    new_logs = []
    for item in parallel_results:
        preds_by_model[item["model_name"]][item["row_idx"]] = item["prediction"]
        new_logs.extend(item["logs"])

    for run_cfg in MODEL_RUNS:
        model_name = run_cfg["name"]
        if model_name not in preds_by_model:
            continue
        col = f"{model_name}_{attribute}{col_suffix}"
        results[col] = preds_by_model[model_name]
        print(f"  {model_name.upper()} done → sample: {preds_by_model[model_name][:3]}", flush=True)

    IO_LOGS.extend(new_logs)
    _save_io_logs()


def save_progress_artifacts(stage_label, attribute=None, label_set=None, col_suffix=""):
    output_csv = Path("../Final Metrics/round1_1a_predictions.csv")
    output_md = Path("../Final Metrics/round1_1a_accuracy_summary.md")
    stage_slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", stage_label)
    output_md_stage = Path(f"../Final Metrics/round1_1a_accuracy_summary_{stage_slug}.md")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_csv, index=False)
    md_text = _accuracy_markdown_text()
    output_md.write_text(md_text, encoding="utf-8")
    output_md_stage.write_text(md_text, encoding="utf-8")
    _save_io_logs()
    print(f"[{stage_label}] Saved predictions CSV → {output_csv.resolve()}", flush=True)
    print(f"[{stage_label}] Saved markdown accuracy table (latest) → {output_md.resolve()}", flush=True)
    print(f"[{stage_label}] Saved markdown accuracy table (snapshot) → {output_md_stage.resolve()}", flush=True)
    print(f"[{stage_label}] Saved IO logs JSON → {IO_LOG_JSON}", flush=True)
    if attribute and label_set:
        _print_attribute_report(stage_label, attribute, label_set, col_suffix=col_suffix)


def smoke_test_models(sample_text, label_set, prompt_template, lang="en"):
    """Run one-sample inference per model and report compatibility without dropping models."""

    print("\n" + "=" * 70)
    print("Smoke test (1 sample) to validate model inference")
    print("=" * 70)

    prompt = prompt_template.format(TEXT=sample_text)
    constraint, _ = _language_controls(label_set, lang=lang)
    failures = []

    tasks = []
    for run_cfg in MODEL_RUNS:
        tasks.append({
            "row_idx": 0,
            "model_name": run_cfg["name"],
            "run_cfg": _worker_run_cfg(run_cfg),
            "prompt_text": prompt,
            "label_set": label_set,
            "constraint_text": constraint,
            "max_output_retries": 3,
            "delay": 0.2,
            "log_meta": {
                "phase": "smoke_test",
                "lang": lang,
            },
        })

    smoke_results = _run_parallel_predictions(tasks)
    smoke_map = {item["model_name"]: item for item in smoke_results}

    for run_cfg in MODEL_RUNS:
        pred = smoke_map[run_cfg["name"]]["prediction"]
        if (not pred) or pred.startswith("ERROR"):
            failures.append((run_cfg["name"], pred))
        if isinstance(pred, str) and ("insufficient_quota" in pred.lower() or pred.startswith("ERROR_FATAL")):
            MODEL_FATAL_ERRORS[run_cfg["name"]] = pred
        print(f"  {run_cfg['name']}: {pred}")

    IO_LOGS.extend(log for item in smoke_results for log in item["logs"])
    _save_io_logs()

    print("\nAll models retained:", [m["name"] for m in MODEL_RUNS])
    if failures:
        print("Smoke-test failures (models still kept):")
        for name, reason in failures:
            print(f"  - {name}: {reason}")

# %% [markdown]
# ## Attribute 1 — Epistemic Stance
# How certain does the speaker sound about the information they are conveying?

# %%
EPISTEMIC_LABELS = ["Certain", "Uncertain", "Neutral/Unclear", "Neutral / NA"]

EPISTEMIC_PROMPT = """\
You are a linguist specialising in colloquial Malay. Your task is to read the Malay sentence below and decide how certain the speaker sounds about the information they are conveying.
Referring to the following three labels and their definitions to make your decision:
Certain: The speaker treats the statement as already true or established. There is no hedging, no doubt, and no checking. The speaker is asserting the information with full confidence.
Uncertain: The speaker sounds unsure, is making a guess, is estimating, or is checking whether something is the case. Words like "agaknya" (I think/probably), "kot" (maybe), or question particles that probe for confirmation are typical signals.
Neutral/NA: The sentence does not carry any detectable certainty signal in either direction. This applies to neutral descriptions, commands, or sentences where certainty is simply not relevant.
Speaker: "{TEXT}"
Given the three labels "Certain, Uncertain, Neutral/NA", the most likely, single label of the speaker's utterance is:"""

# Validate inference first on one sample, then proceed to full SAMPLE_N run.
if not MS_ONLY:
    smoke_test_models(eval_df.iloc[0]["Text"], EPISTEMIC_LABELS, EPISTEMIC_PROMPT, lang="en")
    if os.getenv("STOP_AFTER_SMOKE", "0") == "1":
        _save_io_logs()
        print(f"Saved IO logs to {IO_LOG_JSON}")
        print("STOP_AFTER_SMOKE=1 set; exiting after smoke test.")
        raise SystemExit(0)

    run_attribute("Epistemic_Stance", EPISTEMIC_LABELS, EPISTEMIC_PROMPT)
    save_progress_artifacts("EN.Epistemic_Stance", "Epistemic_Stance", EPISTEMIC_LABELS)

# %% [markdown]
# ## Attribute 2 — Particle Position
# Where does the particle appear in the sentence?

# %%
PARTICLE_POSITION_LABELS = ["Front", "Middle/End", "N/A"]

PARTICLE_POSITION_PROMPT = """\
You are a linguist specialising in colloquial Malay. Your task is to read the Malay sentence below and decide where the discourse particle appears in it.
Referring to the following three labels and their definitions to make your decision:
Front: The particle appears at the very start of the sentence, before any other content words.
Middle/End: The particle appears anywhere other than the front — mid-sentence, before the final word, or at the end.
N/A: No discourse particle is present in the sentence (e.g. the particle slot is shown as "[___]" or the sentence simply contains no particle).
Speaker: "{TEXT}"
Given the three labels "Front, Middle/End, N/A", the most likely, single label of the speaker's utterance is:"""

if not MS_ONLY:
    run_attribute("Particle_Position", PARTICLE_POSITION_LABELS, PARTICLE_POSITION_PROMPT)
    save_progress_artifacts("EN.Particle_Position", "Particle_Position", PARTICLE_POSITION_LABELS)

# %% [markdown]
# ## Attribute 3 — Listener Agreement
# How is the speaker orienting toward the listener in terms of shared knowledge or agreement?

# %%
LISTENER_LABELS = ["Assumed Agreement", "Confirmation Seeking", "Neutral/Unclear"]

LISTENER_PROMPT = """\
You are a linguist specialising in colloquial Malay. Your task is to read the Malay sentence below and decide how the speaker is orienting toward the listener in terms of shared knowledge or agreement.
Referring to the following three labels and their definitions to make your decision:
Assumed Agreement: The speaker treats the information as already shared or obvious to the listener. The sentence is presented as common ground — the underlying tone is "you already know this" or "of course this is true". No explicit confirmation is being requested.
Confirmation Seeking: The speaker is actively checking whether the listener agrees, knows, or can confirm the information. The sentence invites or requests the listener's validation before the speaker can proceed with confidence.
Neutral/Unclear: The sentence does not show any clear orientation toward listener agreement. This applies to plain statements, commands, or cases where the interpersonal stance toward agreement is ambiguous or absent.
Speaker: "{TEXT}"
Given the three labels "Assumed Agreement, Confirmation Seeking, Neutral/Unclear", the most likely, single label of the speaker's utterance is:"""

if not MS_ONLY:
    run_attribute("Listener_Agreement", LISTENER_LABELS, LISTENER_PROMPT)
    save_progress_artifacts("EN.Listener_Agreement", "Listener_Agreement", LISTENER_LABELS)

# %% [markdown]
# ## Attribute 4 — Emotion
# What is the emotional tone of the speaker's utterance?

# %%
EMOTION_LABELS = ["Positive", "Negative", "Neutral/Unclear"]

EMOTION_PROMPT = """\
You are a linguist specialising in colloquial Malay. Your task is to read the Malay sentence below and decide the emotional tone of the speaker.
Referring to the following three labels and their definitions to make your decision:
Positive: The speaker expresses happiness, excitement, enthusiasm, satisfaction, humour, affection, relief, or any other clearly positive feeling. This includes light-hearted teasing or playful sarcasm that is warm in tone.
Negative: The speaker expresses frustration, annoyance, disappointment, sadness, anger, bitterness, or any other clearly negative feeling. This includes hostile or bitter sarcasm.
Neutral/Unclear: The sentence carries no detectable emotional charge in either direction, or the emotion is genuinely ambiguous and cannot be reliably classified as positive or negative.
Speaker: "{TEXT}"
Given the three labels "Positive, Negative, Neutral/Unclear", the most likely, single label of the speaker's utterance is:"""

if not MS_ONLY:
    run_attribute("Emotion", EMOTION_LABELS, EMOTION_PROMPT)
    save_progress_artifacts("EN.Emotion", "Emotion", EMOTION_LABELS)

# %% [markdown]
# ## Attribute 5 — Question Type
# What is the primary sentence function of the utterance?

# %%
QUESTION_TYPE_LABELS = ["Declarative/Statement", "Rhetorical Interrogative", "Yes/No Interrogative"]

QUESTION_TYPE_PROMPT = """\
You are a linguist specialising in colloquial Malay. Your task is to read the Malay sentence below and decide its primary sentence function.
Referring to the following three labels and their definitions to make your decision:
Declarative/Statement: The sentence makes an assertion or conveys information. It describes a situation, states a fact, or expresses a view. It is not structured as a question, even if it ends with a particle.
Rhetorical Interrogative: The sentence is phrased as a question but does not expect a genuine answer from the listener. It is used to make a point, express emotion, or emphasise something — the speaker already implies the answer through the question itself.
Yes/No Interrogative: The sentence is a genuine question that invites the listener to confirm or deny something. The speaker does not already know the answer and is seeking a real yes-or-no response.
Speaker: "{TEXT}"
Given the three labels "Declarative/Statement, Rhetorical Interrogative, Yes/No Interrogative", the most likely, single label of the speaker's utterance is:"""

if not MS_ONLY:
    run_attribute("Question_Type", QUESTION_TYPE_LABELS, QUESTION_TYPE_PROMPT)
    save_progress_artifacts("EN.Question_Type", "Question_Type", QUESTION_TYPE_LABELS)

# %% [markdown]
# ## Save Results

# %%
if not MS_ONLY:
    OUTPUT_CSV = Path("../Final Metrics/round1_1a_predictions.csv")
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {len(results)} rows → {OUTPUT_CSV}")

    model_prefixes = [f"{m['name']}_" for m in MODEL_RUNS]
    keep_cols = ["Text"] + [c for c in results.columns if any(c.startswith(p) for p in model_prefixes)]
    display(results[keep_cols].head(5))

# %% [markdown]
# ## Accuracy Report
# 
# Per-attribute accuracy for each model, plus an overall macro-average.

# %%
EVAL_VARIANTS = {
    "EN": "",
    "MS": "_ms",
}

def normalize(val, label_set, attribute=None):
    return _normalize_for_scoring(val, label_set, attribute)

# ── Per-attribute accuracy table by prompt variant ───────────────────────────
rows = []
model_names = [m["name"] for m in MODEL_RUNS]

for variant_name, suffix in EVAL_VARIANTS.items():
    for attr, labels in ATTRIBUTE_LABEL_MAP.items():
        gt = results[attr].apply(lambda x: normalize(x, labels, attr))
        for model in model_names:
            col = f"{model}_{attr}{suffix}"
            if col not in results.columns:
                continue
            pred = results[col].apply(lambda x: normalize(x, labels, attr))
            acc = (gt == pred).mean()
            rows.append({
                "Variant": variant_name,
                "Attribute": attr,
                "Model": MODEL_VERSION_MAP.get(model.upper(), model.upper()),
                "Accuracy": round(acc, 4),
            })

acc_df = pd.DataFrame(rows)
if acc_df.empty:
    print("No prediction columns found for evaluation.")
else:
    print("=" * 55)
    print("  Test 1a — Attribute Accuracy")
    print("=" * 55)
    for variant_name in EVAL_VARIANTS:
        subset = acc_df[acc_df["Variant"] == variant_name]
        if subset.empty:
            continue
        pivot = subset.pivot(index="Attribute", columns="Model", values="Accuracy")
        macro = pivot.mean().rename("MACRO AVG")
        pivot = pd.concat([pivot, macro.to_frame().T])
        pivot.columns.name = None
        pivot.index.name = "Attribute"
        print(f"\n[{variant_name}] Prompt Suite")
        display((pivot * 100).round(1).astype(str) + "%")

# %%
from sklearn.metrics import classification_report

# ── Per-class precision / recall / F1 for each attribute × model × variant ───
model_names = [m["name"] for m in MODEL_RUNS]

for variant_name, suffix in EVAL_VARIANTS.items():
    has_any_variant_cols = any(
        f"{model}_{attr}{suffix}" in results.columns
        for model in model_names
        for attr in ATTRIBUTE_LABEL_MAP
    )
    if not has_any_variant_cols:
        continue

    print(f"\n{'='*60}")
    print(f"  Variant: {variant_name}")
    print(f"{'='*60}")

    for attr, labels in ATTRIBUTE_LABEL_MAP.items():
        gt_raw = results[attr].apply(lambda x: normalize(x, labels, attr)).fillna("N/A").astype(str)
        print(f"\n{'━'*60}")
        print(f"  {attr}")
        print(f"{'━'*60}")
        for model in model_names:
            col = f"{model}_{attr}{suffix}"
            if col not in results.columns:
                continue
            pred_raw = results[col].apply(lambda x: normalize(x, labels, attr)).fillna("N/A").astype(str)
            print(f"\n  ── {model.upper()} ──")
            print(classification_report(gt_raw, pred_raw, zero_division=0))

# %%
# Malay prompt set (same five attributes), evaluated into *_ms columns.

EPISTEMIC_LABELS_MS = ["Pasti", "Tidak Pasti", "Neutral/NA"]
PARTICLE_POSITION_LABELS_MS = ["Hadapan", "Tengah/Akhir", "Tidak Ada"]
LISTENER_LABELS_MS = ["Anggapan Persetujuan", "Mencari Pengesahan", "Neutral/Tidak Jelas"]
EMOTION_LABELS_MS = ["Positif", "Negatif", "Neutral/Tidak Jelas"]
QUESTION_TYPE_LABELS_MS = ["Deklaratif/Pernyataan", "Tanya Jawab Retorik", "Tanya Jawab Ya/Tidak"]

EPISTEMIC_PROMPT_MS = """\
Anda seorang ahli bahasa yang pakar dalam bahasa Melayu kolokial. Tugas anda adalah untuk membaca ayat Bahasa Melayu di bawah dan menentukan sejauh mana kepastian penutur tentang maklumat yang mereka sampaikan.
Rujuk tiga label berikut dan definisinya untuk membuat keputusan anda:
Pasti: Penutur menganggap pernyataan itu sebagai benar atau sah. Tiada lindung nilai, tiada keraguan, dan tiada semakan. Penutur menegaskan maklumat tersebut dengan penuh keyakinan.
Tidak Pasti: Penutur kedengaran tidak pasti, sedang meneka, sedang menganggarkan, atau sedang menyemak sama ada sesuatu itu benar. Perkataan seperti "agaknya" (saya fikir/mungkin), "kot" (mungkin), atau partikel soalan yang menyiasat untuk pengesahan ialah isyarat tipikal.
Neutral/NA: Ayat ini tidak membawa sebarang isyarat kepastian yang boleh dikesan dalam mana-mana arah. Ini terpakai kepada penerangan, arahan, atau ayat neutral apabila kepastian tidak relevan.
Penutur: "{TEXT}"
Memandangkan tiga label "Pasti, Tidak Pasti, Neutral/NA", label tunggal yang paling mungkin untuk ujaran penutur ialah:"""

PARTICLE_POSITION_PROMPT_MS = """\
Anda seorang ahli bahasa yang pakar dalam bahasa Melayu kolokial. Tugas anda adalah untuk membaca ayat Bahasa Melayu di bawah dan menentukan di mana partikel wacana muncul di dalamnya.
Rujuk tiga label berikut dan definisinya untuk membuat keputusan anda:
Hadapan: Partikel itu muncul pada permulaan ayat, sebelum sebarang perkataan kandungan yang lain.
Tengah/Akhir: Partikel itu muncul di mana-mana sahaja selain bahagian hadapan, termasuk pertengahan ayat, sebelum perkataan terakhir, atau di hujung ayat.
Tidak Ada: Tiada partikel wacana dalam ayat (contohnya slot partikel ditunjukkan sebagai "[___]" atau ayat memang tidak mengandungi partikel).
Penutur: "{TEXT}"
Memandangkan tiga label "Hadapan, Tengah/Akhir, Tidak Ada", label tunggal yang paling mungkin untuk ujaran penutur ialah:"""

LISTENER_PROMPT_MS = """\
Anda seorang ahli bahasa yang pakar dalam bahasa Melayu kolokial. Tugas anda adalah untuk membaca ayat Bahasa Melayu di bawah dan memutuskan bagaimana penutur memberi orientasi kepada pendengar dari segi pengetahuan bersama atau persetujuan.
Rujuk tiga label berikut dan definisinya untuk membuat keputusan anda:
Anggapan Persetujuan: Penutur menganggap maklumat tersebut telah dikongsi atau sudah jelas kepada pendengar. Ayat dibentangkan sebagai titik persamaan, tanpa permintaan pengesahan yang eksplisit.
Mencari Pengesahan: Penutur sedang menyemak sama ada pendengar bersetuju, tahu, atau boleh mengesahkan maklumat tersebut. Ayat menjemput atau meminta pengesahan pendengar.
Neutral/Tidak Jelas: Ayat tidak menunjukkan orientasi yang jelas terhadap persetujuan pendengar. Ini terpakai kepada pernyataan biasa, arahan, atau kes yang samar-samar.
Penutur: "{TEXT}"
Memandangkan tiga label "Anggapan Persetujuan, Mencari Pengesahan, Neutral/Tidak Jelas", label tunggal yang paling mungkin untuk ujaran penutur ialah:"""

EMOTION_PROMPT_MS = """\
Anda seorang ahli bahasa yang pakar dalam bahasa Melayu kolokial. Tugas anda adalah untuk membaca ayat Bahasa Melayu di bawah dan menentukan nada emosi penutur.
Rujuk tiga label berikut dan definisinya untuk membuat keputusan anda:
Positif: Penutur meluahkan kegembiraan, keterujaan, semangat, kepuasan, humor, kasih sayang, kelegaan, atau perasaan positif yang lain.
Negatif: Penutur meluahkan kekecewaan, kegusaran, kesedihan, kemarahan, kepahitan, atau perasaan negatif yang lain.
Neutral/Tidak Jelas: Ayat tidak membawa cas emosi yang boleh dikesan dalam kedua-dua arah, atau emosi benar-benar samar-samar.
Penutur: "{TEXT}"
Memandangkan tiga label "Positif, Negatif, Neutral/Tidak Jelas", label tunggal yang paling mungkin untuk ujaran penutur ialah:"""

QUESTION_TYPE_PROMPT_MS = """\
Anda seorang ahli bahasa yang pakar dalam bahasa Melayu kolokial. Tugas anda adalah untuk membaca ayat Bahasa Melayu di bawah dan menentukan fungsi ayat utamanya.
Rujuk tiga label berikut dan definisinya untuk membuat keputusan anda:
Deklaratif/Pernyataan: Ayat membuat penegasan atau menyampaikan maklumat. Ia bukan soalan tulen walaupun mungkin berakhir dengan partikel.
Tanya Jawab Retorik: Ayat berbentuk soalan tetapi tidak mengharapkan jawapan tulen daripada pendengar; ia digunakan untuk menegaskan poin atau meluahkan emosi.
Tanya Jawab Ya/Tidak: Ayat ialah soalan tulen yang meminta pengesahan atau penafian dan mengharapkan jawapan ya/tidak.
Penutur: "{TEXT}"
Memandangkan tiga label "Deklaratif/Pernyataan, Tanya Jawab Retorik, Tanya Jawab Ya/Tidak", label tunggal yang paling mungkin untuk ujaran penutur ialah:"""

MALAY_PROMPT_RUNS = [
    ("Epistemic_Stance", EPISTEMIC_LABELS_MS, EPISTEMIC_PROMPT_MS),
    ("Particle_Position", PARTICLE_POSITION_LABELS_MS, PARTICLE_POSITION_PROMPT_MS),
    ("Listener_Agreement", LISTENER_LABELS_MS, LISTENER_PROMPT_MS),
    ("Emotion", EMOTION_LABELS_MS, EMOTION_PROMPT_MS),
    ("Question_Type", QUESTION_TYPE_LABELS_MS, QUESTION_TYPE_PROMPT_MS),
]

for attr, labels, prompt in MALAY_PROMPT_RUNS:
    run_attribute(attr, labels, prompt, col_suffix="_ms", lang="ms")
    save_progress_artifacts(f"MS.{attr}", attr, ATTRIBUTE_LABEL_MAP[attr], col_suffix="_ms")


print("\n" + "=" * 55)
print("  Final Report After EN + MS Runs")
print("=" * 55)

accuracy_rows = []
f1_rows = []
model_names = [m["name"] for m in MODEL_RUNS]
for variant_name, suffix in EVAL_VARIANTS.items():
    for attr, labels in ATTRIBUTE_LABEL_MAP.items():
        gt = results[attr].apply(lambda x: normalize(x, labels, attr))
        for model in model_names:
            col = f"{model}_{attr}{suffix}"
            if col not in results.columns:
                continue
            pred = results[col].apply(lambda x: normalize(x, labels, attr))
            acc = (gt == pred).mean()
            accuracy_rows.append({
                "Variant": variant_name,
                "Attribute": attr,
                "Model": MODEL_VERSION_MAP.get(model.upper(), model.upper()),
                "Accuracy": round(acc, 4),
            })

            report = classification_report(gt.fillna("N/A").astype(str), pred.fillna("N/A").astype(str), zero_division=0, output_dict=True)
            for label_name, metrics in report.items():
                if isinstance(metrics, dict):
                    f1_rows.append({
                        "Variant": variant_name,
                        "Attribute": attr,
                        "Model": MODEL_VERSION_MAP.get(model.upper(), model.upper()),
                        "Label": label_name,
                        "Precision": round(float(metrics.get("precision", 0.0)), 4),
                        "Recall": round(float(metrics.get("recall", 0.0)), 4),
                        "F1": round(float(metrics.get("f1-score", 0.0)), 4),
                        "Support": int(metrics.get("support", 0)),
                    })

acc_df = pd.DataFrame(accuracy_rows)
f1_df = pd.DataFrame(f1_rows)
if acc_df.empty:
    print("No prediction columns found for final evaluation.")
else:
    for variant_name in EVAL_VARIANTS:
        subset = acc_df[acc_df["Variant"] == variant_name]
        if subset.empty:
            continue
        pivot = subset.pivot(index="Attribute", columns="Model", values="Accuracy")
        macro = pivot.mean().rename("MACRO AVG")
        pivot = pd.concat([pivot, macro.to_frame().T])
        pivot.columns.name = None
        pivot.index.name = "Attribute"
        print(f"\n[{variant_name}] Prompt Suite")
        display((pivot * 100).round(1).astype(str) + "%")

if not f1_df.empty:
    F1_CSV = Path("../Final Metrics/round1_1a_f1_summary.csv")
    f1_df.to_csv(F1_CSV, index=False)
    print(f"Saved final F1 summary CSV → {F1_CSV}")

# Persist final EN+MS predictions and summary artifacts.
OUTPUT_CSV = Path("../Final Metrics/round1_1a_predictions.csv")
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
results.to_csv(OUTPUT_CSV, index=False)
print(f"Saved final predictions CSV → {OUTPUT_CSV}")

if not acc_df.empty:
    ACCURACY_CSV = Path("../Final Metrics/round1_1a_accuracy_summary.csv")
    acc_df.to_csv(ACCURACY_CSV, index=False)
    print(f"Saved final accuracy summary CSV → {ACCURACY_CSV}")
    GENERAL_ACCURACY_CSV = Path("../Final Metrics/round1_1a_general_accuracy.csv")
    acc_df.to_csv(GENERAL_ACCURACY_CSV, index=False)
    print(f"Saved general accuracy CSV → {GENERAL_ACCURACY_CSV}")

_save_io_logs()
print(f"Saved IO logs JSON → {IO_LOG_JSON}")


