# %%
# # Round 1 Benchmarking — Test 2c: Unconstrained Baseline Generation
#
# Given ONLY the sentence (with particle masked as [___]), predict whether
# the missing particle is "kan" or "ke".
#
# The prompt describes the discourse attributes of each particle so the LLM
# can reason linguistically — but does NOT provide the sentence's actual
# attribute values (that would be Test 2a).
#
# | Setting | Value |
# |---|---|
# | Dataset  | GOLD_187.csv (Synthesised rows with [___]) |
# | Task     | Predict masked particle: kan or ke |
# | Models   | Same suite as Test 1a |
# | Labels   | kan, ke |

# ─────────────────────────────────────────────────────────────────────────────
# PROMPT REVIEW
# The prompt below describes the discourse attributes of each particle.
# Review it here before running.  To change it, edit PARTICLE_PREDICT_PROMPT.
# ─────────────────────────────────────────────────────────────────────────────
PARTICLE_PREDICT_PROMPT = """\
You are a linguist specialising in colloquial Malay discourse particles. \
A discourse particle has been removed from the Malay sentence below and replaced with [___]. \
Your task is to predict which particle, either "ke" or "kan" or "neutral", belongs in that slot.

Particle meanings:
  ke  : signals genuine uncertainty or invites the listener to confirm something the speaker is unsure about.
  kan : signals assumed shared knowledge and seeks light confirmation of something the speaker already believes ("right?").
  neutral : indicates no particle.


Speaker: "{TEXT}"


Given the three candidate particles "kan" and "ke" and "neutral", the single most likely particle to fill [___] is:\
"""


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

# ─── Runtime state ───────────────────────────────────────────────────────────
IO_LOGS: list = []
RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
IO_LOG_JSON = Path(f"../Final Metrics/round1_2c_io_logs_{RUN_ID}.json")
IO_LOG_JSON_LATEST = Path("../Final Metrics/round1_2c_io_logs.json")
PARALLEL_WORKERS = int(os.getenv("PARALLEL_WORKERS", "50"))
# Per-model concurrent worker cap — respects API rate limits
MODEL_MAX_WORKERS = {
    "claude_strong": 5,
    "claude_weak":   5,   # Anthropic org limit: ~50 req/min
    "sealion_strong": 1,
    "sealion_weak":   1,  # SEA-LION hard cap: 10 req/min
}
MODEL_FATAL_ERRORS: dict = {}

PARTICLE_LABELS = ["kan", "ke", "neutral"]

# ─── Helpers shared with round1_demo_test.py ─────────────────────────────────

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


def _extract_label(raw, label_set):
    text = str(raw or "").strip().lower()
    # Prefer exact full-word match first
    for label in sorted(label_set, key=len, reverse=True):
        if re.search(r'\b' + re.escape(label.lower()) + r'\b', text):
            return label
    # Fallback: substring
    for label in sorted(label_set, key=len, reverse=True):
        if label.lower() in text:
            return label
    return str(raw or "").strip()


def _is_fatal_error(error):
    text = str(error).lower()
    markers = [
        "model_not_found", "does not exist", "unsupported parameter",
        "unsupported value", "invalid_request_error", "api key not valid",
        "permission", "authentication", "insufficient_quota",
    ]
    return any(m in text for m in markers)


def _is_rate_limit_error(error):
    text = str(error).lower()
    return "429" in text or "rate_limit" in text or "ratelimit" in text or "too many requests" in text


def _is_transient_error(error):
    text = str(error).lower()
    markers = [
        "503", "service unavailable", "unavailable", "high demand",
        "resource_exhausted", "quota", "429", "deadline", "timeout", "internal",
    ]
    return any(m in text for m in markers)


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
        subprocess.Popen([ollama_cmd, "serve"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         start_new_session=True)
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


# ─── Keys and clients ────────────────────────────────────────────────────────
notebook_path = Path("04_round1_1a_attribute_accuracy.ipynb")
fallback = _extract_keys_from_notebook(notebook_path)

OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY")    or fallback["openai"]
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY") or fallback["anthropic"]
GEMINI_API_KEYS   = [k for k in [os.getenv("GEMINI_API_KEY")] + fallback["gemini"] if k]
DEEPSEEK_API_KEY  = os.getenv("DEEPSEEK_API_KEY")  or fallback["deepseek"]
SEA_LION_API_KEY  = os.getenv("SEA_LION_API_KEY")  or fallback["sealion"]

LLAMA_BASE_URL       = os.getenv("LLAMA_BASE_URL",       "http://localhost:11434")
LLAMA_STRONG_MODEL   = os.getenv("LLAMA_STRONG_MODEL",   "llama3.2:7b")
LLAMA_WEAK_MODEL     = os.getenv("LLAMA_WEAK_MODEL",     "llama3.2:1b")
SEA_LION_BASE_URL    = os.getenv("SEA_LION_BASE_URL",    "https://api.sea-lion.ai/v1")
SEA_LION_STRONG_MODEL = os.getenv("SEA_LION_STRONG_MODEL", "aisingapore/Llama-SEA-LION-v3.5-70B-R")
SEA_LION_WEAK_MODEL  = os.getenv("SEA_LION_WEAK_MODEL",  "aisingapore/Gemma-SEA-LION-v4-27B-IT")
GPT_STRONG_MODEL     = os.getenv("GPT_STRONG_MODEL",     "gpt-5")
GPT_WEAK_MODEL       = os.getenv("GPT_WEAK_MODEL",       "gpt-5.4-mini")
DEEPSEEK_STRONG_MODEL = os.getenv("DEEPSEEK_STRONG_MODEL", "deepseek-v4-pro")
DEEPSEEK_WEAK_MODEL  = os.getenv("DEEPSEEK_WEAK_MODEL",  "deepseek-v4-flash")
CLAUDE_STRONG_MODEL  = os.getenv("CLAUDE_STRONG_MODEL",  "claude-sonnet-4-6")
CLAUDE_WEAK_MODEL    = os.getenv("CLAUDE_WEAK_MODEL",    "claude-haiku-4-5")
GEMINI_STRONG_MODEL  = os.getenv("GEMINI_STRONG_MODEL",  "gemini-3.1-pro-preview")
GEMINI_WEAK_MODEL    = os.getenv("GEMINI_WEAK_MODEL",    "gemini-3.1-flash-lite")
GEMINI_STRONG_FALLBACK_MODELS = [
    m.strip()
    for m in os.getenv("GEMINI_STRONG_FALLBACK_MODELS", "gemini-3.1-flash-lite").split(",")
    if m.strip()
]

MODEL_RUNS = []

if OPENAI_API_KEY:
    MODEL_RUNS.append({
        "name": "gpt_strong", "provider": "openai_responses",
        "client": OpenAI(api_key=OPENAI_API_KEY),
        "api_key": OPENAI_API_KEY, "model": GPT_STRONG_MODEL,
        "max_tokens": 256, "reasoning_effort": "minimal", "sleep": 0.5,
    })
    MODEL_RUNS.append({
        "name": "gpt_weak", "provider": "openai_responses",
        "client": OpenAI(api_key=OPENAI_API_KEY),
        "api_key": OPENAI_API_KEY, "model": GPT_WEAK_MODEL,
        "max_tokens": 128, "reasoning_effort": "low", "sleep": 0.5,
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
    raise ValueError("No model keys/runtimes configured.")

print("Models to run:", [m["name"] for m in MODEL_RUNS])
MODEL_VERSION_MAP = {m["name"].upper(): m["model"] for m in MODEL_RUNS}

# ─── Load data ───────────────────────────────────────────────────────────────
DATA_PATH = Path("../Datasets/GOLD_187.csv")
df_all = pd.read_csv(DATA_PATH)

# Masking logic (per spec):
#   Particle='kan' → regex (?i)\bkan\b → [___]
#   Particle='ke'  → regex (?i)\bke\b  → [___]
#   Particle='neutral' → text already has [___], use as-is
def _apply_mask(text: str, particle: str) -> str:
    if particle == "neutral":
        return text
    masked, n = re.subn(rf"(?i)\b{re.escape(particle)}\b", "[___]", text, count=1)
    if n == 0:
        masked = re.sub(re.escape(particle), "[___]", text, count=1, flags=re.IGNORECASE)
    return masked

def _extract_gt_particle(row) -> str:
    if row["Particle"] in ("kan", "ke"):
        return row["Particle"]
    if row["Particle"] == "neutral":
        return "neutral"
    m = re.search(r"removed\s+(\w+)", str(row["Sentence_Type"]), re.IGNORECASE)
    return m.group(1).lower() if m else ""

masked_df = df_all.copy().reset_index(drop=True)
masked_df["Text_Masked"] = masked_df.apply(
    lambda row: _apply_mask(row["Text"], row["Particle"]), axis=1
)
masked_df["GT_Particle"] = masked_df.apply(_extract_gt_particle, axis=1)
masked_df = masked_df[masked_df["GT_Particle"].isin(["kan", "ke", "neutral"])].reset_index(drop=True)

print(f"Total masked rows usable for Test 2c: {len(masked_df)}")
print(f"GT distribution:\n{masked_df['GT_Particle'].value_counts().to_string()}")
display(masked_df[["Text_Masked", "Sentence_Type", "GT_Particle"]].head(3))

# ─── Inference helpers ────────────────────────────────────────────────────────

def _call_gemini_legacy(api_key, model_name, prompt_text, constraint_text):
    import google.generativeai as genai_legacy
    genai_legacy.configure(api_key=api_key)
    model = genai_legacy.GenerativeModel(
        model_name=model_name,
        system_instruction=constraint_text,
    )
    response = model.generate_content(
        prompt_text,
        generation_config={"temperature": 0, "max_output_tokens": 16},
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
                        max_output_tokens=16,
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


CONSTRAINT_TEXT = (
    "You must output exactly one word — either \"kan\" or \"ke\" or \"neutral\" — and nothing else."
)
STRICT_TAIL = "\n\nReturn exactly one word: kan  or  ke  or  neutral"


def _extract_openai_responses_text(response):
    """Extract text robustly from OpenAI Responses API objects."""
    output_text = (getattr(response, "output_text", "") or "").strip()
    if output_text:
        return output_text

    chunks = []
    for item in (getattr(response, "output", None) or []):
        item_type = getattr(item, "type", None)
        if item_type == "message":
            for content in (getattr(item, "content", None) or []):
                content_type = getattr(content, "type", None)
                if content_type in {"output_text", "text"}:
                    text = (getattr(content, "text", "") or "").strip()
                    if text:
                        chunks.append(text)
    return "\n".join(chunks).strip()


def call_llm(run_cfg, prompt_text, label_set, constraint_text,
             retries=1, delay=1.0, output_try=1, log_meta=None, log_sink=None):
    constraint = constraint_text
    for attempt in range(retries):
        try:
            provider = run_cfg["provider"]
            model = run_cfg["model"]
            client = _build_client(run_cfg)

            if provider == "openai_responses":
                response_payload = {
                    "model": model,
                    "input": [
                        {"role": "system", "content": constraint},
                        {"role": "user", "content": prompt_text},
                    ],
                    "max_output_tokens": max(64, run_cfg.get("max_tokens", 64)),
                }
                if run_cfg.get("reasoning_effort"):
                    response_payload["reasoning"] = {"effort": run_cfg["reasoning_effort"]}

                try:
                    response = client.responses.create(**response_payload)
                except Exception as e:
                    msg = str(e).lower()
                    if "reasoning.effort" in msg and "unsupported" in msg:
                        response_payload.pop("reasoning", None)
                        response = client.responses.create(**response_payload)
                    else:
                        raise
                raw = _extract_openai_responses_text(response)

                # Some OpenAI responses can be structurally valid but text-empty.
                # Fall back once to chat completions to avoid false empty outputs.
                if not raw:
                    fallback = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": constraint},
                            {"role": "user", "content": prompt_text},
                        ],
                        max_completion_tokens=max(16, min(run_cfg.get("max_tokens", 128), 512)),
                    )
                    raw = (fallback.choices[0].message.content or "").strip()

            elif provider == "openai":
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": constraint},
                        {"role": "user", "content": prompt_text},
                    ],
                    max_completion_tokens=max(16, run_cfg.get("max_tokens", 16)),
                )
                raw = (response.choices[0].message.content or "").strip()

            elif provider == "anthropic":
                response = client.messages.create(
                    model=model,
                    system=constraint,
                    max_tokens=16,
                    messages=[{"role": "user", "content": prompt_text}],
                )
                raw = response.content[0].text.strip()

            elif provider == "gemini":
                raw = _call_gemini_with_fallback(run_cfg, prompt_text, constraint)

            elif provider == "ollama":
                url = f"{run_cfg['base_url'].rstrip('/')}/api/generate"
                resp = requests.post(
                    url,
                    json={
                        "model": model,
                        "prompt": f"{constraint}\n\n{prompt_text}",
                        "stream": False,
                        "options": {"temperature": 0},
                    },
                    timeout=60,
                )
                resp.raise_for_status()
                raw = resp.json().get("response", "").strip()

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


def predict_with_output_retry(run_cfg, prompt_text, label_set, constraint_text,
                               max_output_retries=6, delay=0.6,
                               log_meta=None, log_sink=None):
    last_pred = "ERROR: Empty model response"
    for output_try in range(1, max_output_retries + 1):
        pred = call_llm(
            run_cfg, prompt_text, label_set, constraint_text,
            retries=1, delay=0.4, output_try=output_try,
            log_meta=log_meta, log_sink=log_sink,
        )
        if pred and not pred.startswith("ERROR"):
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
    # Per-model semaphores cap concurrency to respect API rate limits
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
    if worker_count <= 1:
        return [
            _predict_task(task)
            for task in tqdm(tasks, total=len(tasks), desc=desc,
                             file=sys.stdout, dynamic_ncols=True)
        ]
    results_list = []
    print(f"Launching {len(tasks)} tasks across {worker_count} threads", flush=True)
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(_guarded, task) for task in tasks]
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc=desc, file=sys.stdout, dynamic_ncols=True):
            results_list.append(future.result())
    return results_list


# ─── Run prediction on a slice of masked_df ──────────────────────────────────

def run_2c(eval_slice: pd.DataFrame, desc="2c"):
    """Run all models on eval_slice, return DataFrame with prediction columns appended."""
    local_results = eval_slice.copy()
    tasks = []
    active_names = []

    for run_cfg in MODEL_RUNS:
        fatal_note = MODEL_FATAL_ERRORS.get(run_cfg["name"])
        if fatal_note:
            local_results[f"{run_cfg['name']}_Particle"] = [fatal_note] * len(eval_slice)
            print(f"  {run_cfg['name'].upper()} skipped (cached fatal error)", flush=True)
            continue
        active_names.append(run_cfg["name"])
        for row_idx, row in enumerate(eval_slice.itertuples(index=False)):
            prompt = PARTICLE_PREDICT_PROMPT.format(TEXT=row.Text_Masked) + STRICT_TAIL
            tasks.append({
                "row_idx": row_idx,
                "model_name": run_cfg["name"],
                "run_cfg": _worker_run_cfg(run_cfg),
                "prompt_text": prompt,
                "label_set": PARTICLE_LABELS,
                "constraint_text": CONSTRAINT_TEXT,
                "max_output_retries": 6,
                "delay": 0.6,
                "log_meta": {"phase": "test_2c", "desc": desc},
            })

    parallel_results = _run_parallel_predictions(tasks, desc=desc)

    preds_by_model = {name: [None] * len(eval_slice) for name in active_names}
    new_logs = []
    for item in parallel_results:
        preds_by_model[item["model_name"]][item["row_idx"]] = item["prediction"]
        new_logs.extend(item["logs"])

    for run_cfg in MODEL_RUNS:
        name = run_cfg["name"]
        if name not in preds_by_model:
            continue
        col = f"{name}_Particle"
        local_results[col] = preds_by_model[name]
        preds = preds_by_model[name]
        print(f"  {name.upper()} → sample preds: {preds[:5]}", flush=True)

    IO_LOGS.extend(new_logs)
    _save_io_logs()
    return local_results


def accuracy_report(result_df: pd.DataFrame):
    """Print per-model accuracy against GT_Particle."""
    gt = result_df["GT_Particle"]
    model_names = [m["name"] for m in MODEL_RUNS]
    rows = []
    for name in model_names:
        col = f"{name}_Particle"
        if col not in result_df.columns:
            continue
        pred = result_df[col].str.lower().str.strip()
        acc = (gt == pred).mean()
        errors = pred.astype(str).str.startswith("error").sum()
        rows.append({
            "Model": MODEL_VERSION_MAP.get(name.upper(), name.upper()),
            "Alias": name,
            "Accuracy": round(float(acc), 4),
            "Errors": int(errors),
        })
    report = pd.DataFrame(rows).sort_values(["Accuracy", "Alias"], ascending=[False, True])
    print("\n" + "=" * 55)
    print("  Test 2c — Unconstrained Baseline Particle Prediction")
    print("=" * 55)
    display(report)
    return report


def save_results(result_df: pd.DataFrame, tag: str):
    out_csv = Path(f"../Final Metrics/round1_2c_predictions_{tag}.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(out_csv, index=False)
    print(f"Saved → {out_csv.resolve()}")


def _accuracy_markdown_text(report_df: pd.DataFrame, tag: str) -> str:
    lines = [
        "# Round 1 Benchmarking - Test 2c Accuracy Summary",
        "",
        f"Run tag: {tag}",
        "",
        "| Model | Alias | Accuracy | Errors |",
        "|---|---|---:|---:|",
    ]
    for row in report_df.itertuples(index=False):
        lines.append(
            f"| {row.Model} | {row.Alias} | {row.Accuracy:.4f} | {int(row.Errors)} |"
        )
    return "\n".join(lines) + "\n"


def save_accuracy_markdown(report_df: pd.DataFrame, tag: str):
    out_md = Path(f"../Final Metrics/round1_2c_accuracy_summary_{tag}.md")
    out_md_latest = Path("../Final Metrics/round1_2c_accuracy_summary.md")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    text = _accuracy_markdown_text(report_df, tag)
    out_md.write_text(text, encoding="utf-8")
    out_md_latest.write_text(text, encoding="utf-8")
    print(f"Saved → {out_md.resolve()}")
    print(f"Saved → {out_md_latest.resolve()}")


# ═════════════════════════════════════════════════════════════════════════════
# STEP 1 — Print the prompt for visual inspection
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PROMPT BEING USED (Test 2c):")
print("=" * 70)
print(PARTICLE_PREDICT_PROMPT)
print("  [+ STRICT_TAIL appended at runtime]")
print("=" * 70 + "\n")

# ═════════════════════════════════════════════════════════════════════════════
# STEP 2 — 1-sample test
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 60)
print("STEP 2: Running 1-sample test")
print("─" * 60)

sample_1 = masked_df.head(1).copy()
print("Sample sentence :", sample_1.iloc[0]["Text_Masked"][:120])
print("Ground truth    :", sample_1.iloc[0]["GT_Particle"])
print()

results_1 = run_2c(sample_1, desc="1-sample")
report_1 = accuracy_report(results_1)
save_results(results_1, "1sample")
save_accuracy_markdown(report_1, "1sample")

# ═════════════════════════════════════════════════════════════════════════════
# STEP 3 — 5-sample test
# Run this block after reviewing the 1-sample results above.
# Set env var SKIP_5SAMPLE=1 to skip (e.g. during debugging).
# ═════════════════════════════════════════════════════════════════════════════
if os.getenv("SKIP_5SAMPLE", "0") != "1":
    print("\n" + "─" * 60)
    print("STEP 3: Running 5-sample test")
    print("─" * 60)

    sample_5 = masked_df.head(5).copy()
    print("Ground truths:", sample_5["GT_Particle"].tolist())
    print()

    results_5 = run_2c(sample_5, desc="5-samples")
    report_5 = accuracy_report(results_5)
    save_results(results_5, "5samples")
    save_accuracy_markdown(report_5, "5samples")
else:
    print("SKIP_5SAMPLE=1 set; skipping 5-sample run.")

# ═════════════════════════════════════════════════════════════════════════════
# STEP 4 — Full dataset run
# Set env var SKIP_FULL=1 to skip (e.g. during debugging).
# ═════════════════════════════════════════════════════════════════════════════
if os.getenv("SKIP_FULL", "0") != "1":
    print("\n" + "─" * 60)
    print(f"STEP 4: Running full dataset ({len(masked_df)} rows)")
    print("─" * 60)

    results_full = run_2c(masked_df.copy(), desc="full")
    report_full = accuracy_report(results_full)
    save_results(results_full, "full")
    save_accuracy_markdown(report_full, "full")
else:
    print("SKIP_FULL=1 set; skipping full dataset run.")
