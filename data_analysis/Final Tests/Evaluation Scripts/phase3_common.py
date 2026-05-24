"""Shared infrastructure for Phase 3 tests (2d, 1b-CoT, 2b-CoT).

Reuses the model-runner pattern from round1_2a_particle_generation.py and
round2.py. Loads GOLD_187 + 16-cluster k-modes mapping to attach Macro_Function.
"""

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

try:
    from dotenv import load_dotenv
    for p in [Path(".env"), Path("../.env"), Path("../../.env"), Path("../../../.env")]:
        if p.exists():
            load_dotenv(p)
            break
except ImportError:
    pass

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


# ── Labels & definitions ──────────────────────────────────────────────────────
PARTICLE_LABELS = ["ke", "kan", "neutral"]

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
        "(surprise, irritation, humour, excitement, disbelief, etc.).",
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

CLUSTER_TO_MACRO = {
    0: "Assumed-Agreement Rhetorical Stance",
    6: "Assumed-Agreement Rhetorical Stance",
    9: "Assumed-Agreement Rhetorical Stance",
    1: "Neutral Declarative",
    4: "Information-Seeking Verification",
    10: "Information-Seeking Verification",
    13: "Information-Seeking Verification",
    2: "Affective Confirmation-Seeking Question",
    14: "Affective Confirmation-Seeking Question",
    15: "Affective Confirmation-Seeking Question",
    3: "Emphatic / Discourse-Marking",
    5: "Null Form Retaining Particle-Like Pragmatic Meaning",
    12: "Null Form Retaining Particle-Like Pragmatic Meaning",
    11: "Negative Rhetorical Challenge / Evaluation",
    7: "Negative Rhetorical Challenge / Evaluation",
    8: "Negative Rhetorical Challenge / Evaluation",
}

ATTRIBUTE_DESCRIPTIONS = {
    "Epistemic_Stance": {
        "Certain": "The speaker treats the statement as already true or established — no hedging, no doubt, full confidence.",
        "Uncertain": "The speaker sounds unsure, is making a guess, estimating, or checking whether something is the case.",
        "Neutral/Unclear": "No detectable certainty signal; the sentence does not lean toward confident assertion or uncertainty.",
        "Neutral / NA": "No detectable certainty signal; epistemic stance is not relevant here.",
    },
    "Particle_Position": {
        "Front": "The particle appears at the very start of the sentence, before any other content words.",
        "Middle/End": "The particle appears anywhere other than the front — mid-sentence or at the end.",
        "N/A": "No discourse particle position is applicable.",
    },
    "Listener_Agreement": {
        "Assumed Agreement": "The speaker treats the information as shared or obvious — the underlying tone is 'you already know this'.",
        "Confirmation Seeking": "The speaker is actively checking whether the listener agrees or can confirm something.",
        "Neutral/Unclear": "No clear orientation toward listener agreement; a plain statement or ambiguous stance.",
    },
    "Emotion": {
        "Positive": "The speaker expresses happiness, excitement, enthusiasm, satisfaction, humour, affection, or relief.",
        "Negative": "The speaker expresses frustration, annoyance, disappointment, sadness, anger, or bitterness.",
        "Neutral/Unclear": "No detectable emotional charge, or the emotion is genuinely ambiguous.",
    },
    "Question_Type": {
        "Declarative/Statement": "The sentence makes an assertion or conveys information — not structured as a question.",
        "Rhetorical Interrogative": "Phrased as a question but does not expect a genuine answer; used to make a point or emphasise something.",
        "Yes/No Interrogative": "A genuine question inviting the listener to confirm or deny; the speaker does not already know the answer.",
    },
}


# ── Data loading ──────────────────────────────────────────────────────────────
def mask_particle(text: str, particle: str) -> str:
    if particle == "neutral":
        return text
    masked, n = re.subn(rf"(?i)\b{re.escape(particle)}\b", "[___]", text, count=1)
    if n == 0:
        masked = re.sub(re.escape(particle), "[___]", text, count=1, flags=re.IGNORECASE)
    return masked


def _get_gt_particle(row) -> str:
    if row["Particle"] in ("kan", "ke", "neutral"):
        return row["Particle"]
    m = re.search(r"removed\s+(\w+)", str(row.get("Sentence_Type", "")), re.IGNORECASE)
    return m.group(1).lower() if m else ""


def load_gold_with_macro():
    gold_path = Path("../Datasets/GOLD_187.csv")
    cluster_path = Path("clustered_kmodes_discourse_context_16.csv")
    df = pd.read_csv(gold_path)
    cl = pd.read_csv(cluster_path)[["Text", "cluster"]].drop_duplicates("Text")
    df = df.merge(cl, on="Text", how="left")
    df["Macro_Function"] = df["cluster"].map(CLUSTER_TO_MACRO)
    df["Text_Masked"] = df.apply(lambda r: mask_particle(r["Text"], r["Particle"]), axis=1)
    df["GT_Particle"] = df.apply(_get_gt_particle, axis=1)
    df = df[df["Macro_Function"].notna()].reset_index(drop=True)
    df = df[df["GT_Particle"].isin(["ke", "kan", "neutral"])].reset_index(drop=True)
    return df


# ── Attribute block builder ───────────────────────────────────────────────────
def build_attribute_block(row) -> str:
    pairs = [
        ("Epistemic Stance",   "Epistemic_Stance"),
        ("Particle Position",  "Particle_Position"),
        ("Listener Agreement", "Listener_Agreement"),
        ("Emotion",            "Emotion"),
        ("Question Type",      "Question_Type"),
    ]
    lines = []
    for label, col in pairs:
        val = str(row[col]).strip()
        desc = ATTRIBUTE_DESCRIPTIONS[col].get(val, "")
        lines.append(f"- {label}: {val} — {desc}")
    return "\n".join(lines)


# ── Model setup (mirrors round1_2a) ───────────────────────────────────────────
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
    if host not in {"localhost", "127.0.0.1", "::1"}:
        return False, f"Ollama endpoint not reachable: {base_url}"
    ollama_cmd = shutil.which("ollama")
    if not ollama_cmd:
        return False, "Ollama is not installed."
    try:
        subprocess.Popen([ollama_cmd, "serve"], stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, start_new_session=True)
    except Exception as exc:
        return False, f"Failed to start ollama: {exc}"
    for _ in range(12):
        time.sleep(1)
        if _probe_ollama(base_url):
            return True, "Ollama auto-started."
    return False, "Ollama auto-start failed."


def build_model_runs():
    fallback = _extract_keys_from_notebook(Path("04_round1_1a_attribute_accuracy.ipynb"))
    OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY")    or fallback["openai"]
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY") or fallback["anthropic"]
    GEMINI_API_KEYS   = [k for k in [os.getenv("GEMINI_API_KEY"),
                                      os.getenv("GEMINI_API_KEY_2"),
                                      os.getenv("GEMINI_API_KEY_3")] + fallback["gemini"] if k]
    # dedupe preserving order
    GEMINI_API_KEYS   = list(dict.fromkeys(GEMINI_API_KEYS))
    DEEPSEEK_API_KEY  = os.getenv("DEEPSEEK_API_KEY")  or fallback["deepseek"]
    SEA_LION_API_KEY  = os.getenv("SEA_LION_API_KEY")  or fallback["sealion"]

    LLAMA_BASE_URL       = os.getenv("LLAMA_BASE_URL", "http://localhost:11434")
    LLAMA_STRONG_MODEL   = os.getenv("LLAMA_STRONG_MODEL", "llama3.2:7b")
    LLAMA_WEAK_MODEL     = os.getenv("LLAMA_WEAK_MODEL", "llama3.2:1b")
    SEA_LION_BASE_URL    = os.getenv("SEA_LION_BASE_URL", "https://api.sea-lion.ai/v1")
    SEA_LION_STRONG      = os.getenv("SEA_LION_STRONG_MODEL", "aisingapore/Llama-SEA-LION-v3.5-70B-R")
    SEA_LION_WEAK        = os.getenv("SEA_LION_WEAK_MODEL", "aisingapore/Gemma-SEA-LION-v4-27B-IT")
    GPT_STRONG_MODEL     = os.getenv("GPT_STRONG_MODEL", "gpt-5")
    GPT_WEAK_MODEL       = os.getenv("GPT_WEAK_MODEL", "gpt-5.4-mini")
    GPT_STRONG_EFFORT    = os.getenv("GPT_STRONG_REASONING_EFFORT", "minimal")
    GPT_WEAK_EFFORT      = os.getenv("GPT_WEAK_REASONING_EFFORT", "none")
    DEEPSEEK_STRONG      = os.getenv("DEEPSEEK_STRONG_MODEL", "deepseek-v4-pro")
    DEEPSEEK_WEAK        = os.getenv("DEEPSEEK_WEAK_MODEL", "deepseek-v4-flash")
    CLAUDE_STRONG        = os.getenv("CLAUDE_STRONG_MODEL", "claude-sonnet-4-6")
    CLAUDE_WEAK          = os.getenv("CLAUDE_WEAK_MODEL", "claude-haiku-4-5")
    GEMINI_STRONG        = os.getenv("GEMINI_STRONG_MODEL", "gemini-3.1-pro-preview")
    GEMINI_WEAK          = os.getenv("GEMINI_WEAK_MODEL", "gemini-3.1-flash-lite")
    GEMINI_STRONG_FB     = [m.strip() for m in os.getenv("GEMINI_STRONG_FALLBACK_MODELS",
                            "gemini-3.1-flash-lite").split(",") if m.strip()]

    runs = []
    if OPENAI_API_KEY:
        runs.append({"name": "gpt_strong", "provider": "openai_responses",
                     "api_key": OPENAI_API_KEY, "model": GPT_STRONG_MODEL,
                     "reasoning_effort": GPT_STRONG_EFFORT, "max_tokens": 1024,
                     "sleep": 0.5})
        runs.append({"name": "gpt_weak", "provider": "openai_responses",
                     "api_key": OPENAI_API_KEY, "model": GPT_WEAK_MODEL,
                     "reasoning_effort": GPT_WEAK_EFFORT, "sleep": 0.5})
    if ANTHROPIC_API_KEY:
        runs.append({"name": "claude_strong", "provider": "anthropic",
                     "api_key": ANTHROPIC_API_KEY, "model": CLAUDE_STRONG, "sleep": 0.5})
        runs.append({"name": "claude_weak", "provider": "anthropic",
                     "api_key": ANTHROPIC_API_KEY, "model": CLAUDE_WEAK, "sleep": 0.5})
    if GEMINI_API_KEYS:
        runs.append({"name": "gemini_strong", "provider": "gemini",
                     "api_keys": GEMINI_API_KEYS, "model": GEMINI_STRONG,
                     "fallback_models": GEMINI_STRONG_FB,
                     "max_tokens": 8192, "sleep": 0.5})
        runs.append({"name": "gemini_weak", "provider": "gemini",
                     "api_keys": GEMINI_API_KEYS, "model": GEMINI_WEAK,
                     "max_tokens": 8192, "sleep": 0.5})
    if DEEPSEEK_API_KEY:
        runs.append({"name": "deepseek_strong", "provider": "openai",
                     "api_key": DEEPSEEK_API_KEY, "base_url": "https://api.deepseek.com",
                     "model": DEEPSEEK_STRONG, "sleep": 0.3})
        runs.append({"name": "deepseek_weak", "provider": "openai",
                     "api_key": DEEPSEEK_API_KEY, "base_url": "https://api.deepseek.com",
                     "model": DEEPSEEK_WEAK, "sleep": 0.3})
    llama_ok, llama_status = _ensure_ollama_runtime(LLAMA_BASE_URL)
    print("Llama/Ollama status:", llama_status)
    if llama_ok:
        runs.append({"name": "llama_strong", "provider": "ollama",
                     "model": LLAMA_STRONG_MODEL, "base_url": LLAMA_BASE_URL, "sleep": 0.2})
        runs.append({"name": "llama_weak", "provider": "ollama",
                     "model": LLAMA_WEAK_MODEL, "base_url": LLAMA_BASE_URL, "sleep": 0.2})
    if SEA_LION_API_KEY:
        runs.append({"name": "sealion_strong", "provider": "openai",
                     "api_key": SEA_LION_API_KEY, "base_url": SEA_LION_BASE_URL,
                     "model": SEA_LION_STRONG, "max_tokens": 1024, "sleep": 0.3})
        runs.append({"name": "sealion_weak", "provider": "openai",
                     "api_key": SEA_LION_API_KEY, "base_url": SEA_LION_BASE_URL,
                     "model": SEA_LION_WEAK, "sleep": 0.3})
    if not runs:
        raise ValueError("No model keys/runtimes configured.")
    return runs


MODEL_MAX_WORKERS = {
    "claude_strong": 5,
    "claude_weak":   5,
    "sealion_strong": 1,
    "sealion_weak":   1,
}
PARALLEL_WORKERS = int(os.getenv("PARALLEL_WORKERS", "50"))


# ── LLM call infrastructure ───────────────────────────────────────────────────
def _utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def _is_fatal_error(error):
    text = str(error).lower()
    return any(m in text for m in [
        "model_not_found", "does not exist", "unsupported parameter",
        "invalid_request_error", "api key not valid", "permission",
        "authentication", "insufficient_quota",
    ])


def _build_client(run_cfg):
    provider = run_cfg["provider"]
    api_key = run_cfg.get("api_key")
    if provider == "openai_responses":
        return OpenAI(api_key=api_key)
    if provider == "openai":
        return OpenAI(api_key=api_key, base_url=run_cfg.get("base_url"))
    if provider == "anthropic":
        return anthropic.Anthropic(api_key=api_key)
    return None


def _call_gemini_with_fallback(run_cfg, prompt_text, constraint_text, max_tokens):
    last_error = None
    candidates = [run_cfg["model"]] + [m for m in run_cfg.get("fallback_models", [])
                                       if m != run_cfg["model"]]
    for model_name in candidates:
        for api_key in run_cfg.get("api_keys", []):
            try:
                client = genai.Client(api_key=api_key,
                                      http_options=genai_types.HttpOptions(timeout=120000))
                response = client.models.generate_content(
                    model=model_name,
                    config=genai_types.GenerateContentConfig(
                        system_instruction=constraint_text,
                        max_output_tokens=max_tokens, temperature=0),
                    contents=prompt_text,
                )
                raw = (response.text or "").strip()
                if raw:
                    return raw
                last_error = RuntimeError(f"Empty Gemini response: {model_name}")
            except Exception as e:
                last_error = e
                if _is_fatal_error(e):
                    break
    raise last_error


def call_llm_raw(run_cfg, prompt_text, constraint_text, max_tokens=8):
    """Returns raw model output string (or raises)."""
    provider = run_cfg["provider"]
    model = run_cfg["model"]
    client = _build_client(run_cfg)

    if provider == "openai_responses":
        req = {
            "model": model,
            "input": [
                {"role": "system", "content": constraint_text},
                {"role": "user", "content": prompt_text},
            ],
            "max_output_tokens": max(16, max_tokens),
        }
        if run_cfg.get("reasoning_effort"):
            req["reasoning"] = {"effort": run_cfg["reasoning_effort"]}
        response = client.responses.create(**req)
        return (getattr(response, "output_text", "") or "").strip()

    if provider == "openai":
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": constraint_text},
                {"role": "user", "content": prompt_text},
            ],
            max_completion_tokens=max_tokens,
        )
        return (response.choices[0].message.content or "").strip()

    if provider == "anthropic":
        response = client.messages.create(
            model=model, system=constraint_text, max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt_text}],
        )
        return response.content[0].text.strip()

    if provider == "gemini":
        return _call_gemini_with_fallback(run_cfg, prompt_text, constraint_text, max_tokens)

    if provider == "ollama":
        url = f"{run_cfg['base_url'].rstrip('/')}/api/generate"
        response = requests.post(
            url, timeout=120,
            json={"model": model, "prompt": f"{constraint_text}\n\n{prompt_text}",
                  "stream": False, "options": {"temperature": 0,
                                               "num_predict": max_tokens}},
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()

    raise ValueError(f"Unsupported provider: {provider}")


def predict_with_retry(run_cfg, prompt_text, constraint_text, parse_fn,
                       max_tokens=8, max_output_retries=4, log_sink=None,
                       log_meta=None):
    """Call model, parse output, retry on errors. Returns (label, raw_full_text)."""
    last_err = None
    for output_try in range(1, max_output_retries + 1):
        try:
            raw = call_llm_raw(run_cfg, prompt_text, constraint_text, max_tokens=max_tokens)
            label = parse_fn(raw)
            status = "ok" if str(label).strip() else "empty"
            if log_sink is not None:
                log_sink.append({
                    "timestamp": _utc_now_iso(), "model_alias": run_cfg["name"],
                    "provider": run_cfg["provider"], "model": run_cfg["model"],
                    "output_try": output_try, "input": prompt_text,
                    "constraint": constraint_text, "raw_output": str(raw),
                    "parsed_output": str(label), "status": status,
                    "meta": log_meta or {},
                })
            if not str(label).strip():
                raise RuntimeError("Empty parsed output")
            return label, raw
        except Exception as e:
            last_err = e
            if log_sink is not None:
                log_sink.append({
                    "timestamp": _utc_now_iso(), "model_alias": run_cfg["name"],
                    "provider": run_cfg["provider"], "model": run_cfg["model"],
                    "output_try": output_try, "input": prompt_text,
                    "constraint": constraint_text, "raw_output": "",
                    "parsed_output": "", "status": "error", "error": str(e),
                    "meta": log_meta or {},
                })
            if _is_fatal_error(e):
                return f"ERROR_FATAL: {e}", ""
            txt = str(e).lower()
            if "429" in txt or "rate_limit" in txt or "resource_exhausted" in txt:
                # Bound backoff: minute-rate caps recover quickly; daily quotas don't.
                if output_try >= 3:
                    print(f"  [persistent 429 — giving up on {run_cfg['name']}]", flush=True)
                    return f"ERROR: {e}", ""
                wait = 65 + random.uniform(0, 15)
                print(f"  [rate-limit backoff {wait:.0f}s — {run_cfg['name']}]", flush=True)
                time.sleep(wait)
            else:
                time.sleep(0.6 * output_try)
    return f"ERROR: {last_err}", ""


def run_parallel(tasks, desc="run"):
    if not tasks:
        return []
    sems = {}
    for t in tasks:
        name = t["model_name"]
        if name not in sems:
            sems[name] = threading.Semaphore(MODEL_MAX_WORKERS.get(name, PARALLEL_WORKERS))

    def _guarded(task):
        with sems[task["model_name"]]:
            logs = []
            # Per-model floor for max_tokens (e.g. sealion needs more)
            effective_tokens = max(task["max_tokens"],
                                   task["run_cfg"].get("max_tokens") or 0)
            label, raw = predict_with_retry(
                task["run_cfg"], task["prompt_text"], task["constraint_text"],
                task["parse_fn"], max_tokens=effective_tokens,
                log_sink=logs, log_meta=task.get("log_meta"),
            )
            return {"row_idx": task["row_idx"], "model_name": task["model_name"],
                    "prediction": label, "raw_full": raw, "logs": logs}

    workers = min(PARALLEL_WORKERS, len(tasks))
    if workers <= 1:
        return [_guarded(t) for t in tqdm(tasks, desc=desc, file=sys.stdout, dynamic_ncols=True)]
    out = []
    print(f"Launching {len(tasks)} tasks across {workers} threads", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_guarded, t) for t in tasks]
        for f in tqdm(as_completed(futs), total=len(futs), desc=desc,
                      file=sys.stdout, dynamic_ncols=True):
            out.append(f.result())
    return out


# ── Parsing helpers ───────────────────────────────────────────────────────────
_FINAL_ANSWER_RE = re.compile(r"final\s*answer\s*:\s*(.+?)(?:\n|$)", re.IGNORECASE | re.DOTALL)


def parse_particle_plain(raw):
    text = str(raw or "").strip().lower()
    text = re.sub(r"</?think>", "", text).strip()
    text = re.sub(r"^[\W_]+", "", text).strip()
    if text in {"ke", "kan", "neutral"}:
        return text
    for p in ["neutral", "kan", "ke"]:
        if re.search(rf"\b{p}\b", text):
            return p
    return ""


def parse_particle_cot(raw):
    text = str(raw or "")
    m = _FINAL_ANSWER_RE.search(text)
    if m:
        return parse_particle_plain(m.group(1))
    return parse_particle_plain(text)


def parse_macro_cot(raw):
    text = str(raw or "")
    m = _FINAL_ANSWER_RE.search(text)
    candidate = (m.group(1) if m else text).strip().strip(".").strip()
    cand_low = candidate.lower()
    # exact / substring match against canonical labels
    for label in MACRO_FUNCTION_LABELS:
        if label.lower() == cand_low:
            return label
    for label in MACRO_FUNCTION_LABELS:
        if label.lower() in cand_low or cand_low in label.lower():
            return label
    return candidate  # unparsed — store for inspection


# ── IO log persistence ────────────────────────────────────────────────────────
def save_io_logs(io_logs, test_name, run_id):
    out_dir = Path("../Final Metrics")
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(io_logs, ensure_ascii=False, indent=2)
    (out_dir / f"{test_name}_io_logs_{run_id}.json").write_text(payload, encoding="utf-8")
    (out_dir / f"{test_name}_io_logs.json").write_text(payload, encoding="utf-8")


# ── Reporting ─────────────────────────────────────────────────────────────────
def write_accuracy_report(results_df, model_runs, gt_col, pred_col_suffix,
                          test_name, group_col=None):
    """Writes overall accuracy + optional per-group breakdown markdown + CSV."""
    out_dir = Path("../Final Metrics")
    out_dir.mkdir(parents=True, exist_ok=True)
    model_version_map = {m["name"].upper(): m["model"] for m in model_runs}

    overall_rows = []
    for m in model_runs:
        col = f"{m['name']}_{pred_col_suffix}"
        if col not in results_df.columns:
            continue
        pred = results_df[col].astype(str)
        gt = results_df[gt_col].astype(str)
        acc = (gt == pred).mean()
        errors = pred.str.startswith("ERROR").sum()
        overall_rows.append({
            "Model": model_version_map.get(m["name"].upper(), m["name"]),
            "Accuracy": round(float(acc), 4),
            "Errors": int(errors),
        })

    if overall_rows:
        acc_df = pd.DataFrame(overall_rows).sort_values(["Accuracy", "Model"],
                                                        ascending=[False, True])
        display(acc_df)
        md = ["# " + test_name + " — Overall Accuracy", "",
              "| Model | Accuracy | Errors |", "|---|---|---|"]
        for _, r in acc_df.iterrows():
            md.append(f"| {r['Model']} | {r['Accuracy']:.4f} | {r['Errors']} |")

        if group_col and group_col in results_df.columns:
            md += ["", f"# {test_name} — Accuracy by {group_col}", ""]
            groups = sorted(results_df[group_col].dropna().unique())
            header = "| Model | " + " | ".join(groups) + " |"
            sep = "|---" * (len(groups) + 1) + "|"
            md += [header, sep]
            for m in model_runs:
                col = f"{m['name']}_{pred_col_suffix}"
                if col not in results_df.columns:
                    continue
                row_vals = [model_version_map.get(m["name"].upper(), m["name"])]
                for g in groups:
                    sub = results_df[results_df[group_col] == g]
                    if len(sub) == 0:
                        row_vals.append("—")
                    else:
                        acc = (sub[gt_col].astype(str) == sub[col].astype(str)).mean()
                        row_vals.append(f"{acc:.3f} (n={len(sub)})")
                md.append("| " + " | ".join(row_vals) + " |")

        (out_dir / f"{test_name}_accuracy_summary.md").write_text("\n".join(md) + "\n",
                                                                  encoding="utf-8")
        print(f"Saved accuracy summary → ../Final Metrics/{test_name}_accuracy_summary.md")


def worker_cfg(run_cfg):
    c = dict(run_cfg)
    c.pop("client", None)
    return c
