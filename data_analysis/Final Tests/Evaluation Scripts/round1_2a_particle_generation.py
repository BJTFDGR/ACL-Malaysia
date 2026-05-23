# %% [markdown]
# # Round 1 Benchmarking — Test 2a: Attribute-Constrained Particle Generation
#
# Given a sentence with a masked particle ([___]) + all 5 discourse-context attributes,
# ask each LLM to predict which particle fills the slot.
#
# | Setting | Value |
# |---|---|
# | Dataset | GOLD_187.csv (Natural rows only, particle masked for inference) |
# | Task | Attribute-Constrained Generation (Test 2a) |
# | Models | Same as Test 1a |
# | Particles | ke, kan |
# | Attributes supplied | Epistemic_Stance, Particle_Position, Listener_Agreement, Emotion, Question_Type |

# %%
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

IO_LOGS = []
RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
IO_LOG_JSON = Path(f"../Final Metrics/round1_2a_io_logs_{RUN_ID}.json")
IO_LOG_JSON_LATEST = Path("../Final Metrics/round1_2a_io_logs.json")
PARALLEL_WORKERS = int(os.getenv("PARALLEL_WORKERS", "50"))
# Per-model concurrent worker cap — respects API rate limits
MODEL_MAX_WORKERS = {
    "claude_strong": 5,
    "claude_weak":   5,   # Anthropic org limit: ~50 req/min
    "sealion_strong": 1,
    "sealion_weak":   1,  # SEA-LION hard cap: 10 req/min
}
MODEL_FATAL_ERRORS = {}

PARTICLE_LABELS = ["ke", "kan", "neutral"]

# ── Attribute descriptions (mirrors Test 1a prompts) ──────────────────────────
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

# ─────────────────────────────────────────────────────────────────────────────
# PROMPT — REVIEW THIS BEFORE RUNNING
# ─────────────────────────────────────────────────────────────────────────────
#
# The prompt presents the masked sentence followed by a structured breakdown
# of all five discourse-context attributes and their definitions, then asks
# the model to choose between exactly two particles: "ke" or "kan".
#
# Both "ke" and "kan" are colloquial Malay discourse particles:
#   ke  — signals genuine uncertainty or invites the listener to confirm;
#         commonly appears in yes/no questions or when the speaker is unsure.
#   kan — signals assumed shared knowledge, seeks light confirmation of
#         something the speaker already believes; often translated as "right?"
#
# The attribute block is generated dynamically per row. Example for one row:
#
#   Epistemic Stance: Certain
#     → The speaker treats the statement as already true — full confidence.
#   Particle Position: Middle/End
#     → The particle appears mid-sentence or at the end.
#   Listener Agreement: Assumed Agreement
#     → The speaker treats the information as shared or obvious.
#   Emotion: Negative
#     → The speaker expresses frustration, annoyance, or bitterness.
#   Question Type: Rhetorical Interrogative
#     → Phrased as a question but does not expect a genuine answer.
#
# ─────────────────────────────────────────────────────────────────────────────

PARTICLE_GEN_SYSTEM_CONSTRAINT = (
    "You are a linguist specialising in colloquial Malay discourse particles. "
    "You must output exactly one word — either \"ke\" or \"kan\" or \"neutral\" — and nothing else."
)

PARTICLE_GEN_STRICT_TAIL = (
    "\n\nReturn exactly one word from this set and nothing else: ke, kan, neutral"
)


def build_particle_gen_prompt(text_masked: str, row: pd.Series) -> str:
    """Build the attribute-aware generation prompt for one masked row."""
    attr_block_lines = []

    # Epistemic Stance
    es_val = str(row["Epistemic_Stance"]).strip()
    es_desc = ATTRIBUTE_DESCRIPTIONS["Epistemic_Stance"].get(es_val, "")
    attr_block_lines.append(
        f"{es_desc}"
    )

    # Particle Position
    pp_val = str(row["Particle_Position"]).strip()
    pp_desc = ATTRIBUTE_DESCRIPTIONS["Particle_Position"].get(pp_val, "")
    attr_block_lines.append(
        f"{pp_desc}"
    )

    # Listener Agreement
    la_val = str(row["Listener_Agreement"]).strip()
    la_desc = ATTRIBUTE_DESCRIPTIONS["Listener_Agreement"].get(la_val, "")
    attr_block_lines.append(
        f"{la_desc}"
    )

    # Emotion
    em_val = str(row["Emotion"]).strip()
    em_desc = ATTRIBUTE_DESCRIPTIONS["Emotion"].get(em_val, "")
    attr_block_lines.append(
        f"{em_desc}"
    )

    # Question Type
    qt_val = str(row["Question_Type"]).strip()
    qt_desc = ATTRIBUTE_DESCRIPTIONS["Question_Type"].get(qt_val, "")
    attr_block_lines.append(
        f"{qt_desc}"
    )

    attr_block = "\n".join(attr_block_lines)

    prompt = f"""\
You are given a Malay sentence in which one discourse particle has been replaced with [___].
Your task is to predict which particle, either “ke,” “kan,” or “neutral,” belongs in the [___] slot, such that the discourse-context attributes for this sentence are {attr_block}.

Particle meanings:
  ke  : signals genuine uncertainty or invites the listener to confirm something the speaker is unsure about.
  kan : signals assumed shared knowledge and seeks light confirmation of something the speaker already believes ("right?").
  neutral : indicates no particle.

Speaker:
  "{text_masked}"



Using the sentence context and the attributes above, which single particle — "ke" or "kan" or "neutral" — best fills [___]?"""

    return prompt


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


def _probe_ollama(base_url, timeout=2):
    try:
        tags_url = f"{base_url.rstrip('/')}/api/tags"
        resp = requests.get(tags_url, timeout=timeout)
        return resp.ok
    except Exception:
        return False


def _ensure_ollama_runtime(base_url):
    if _probe_ollama(base_url):
        return True, "Ollama endpoint is reachable."
    parsed = urlparse(base_url)
    host = (parsed.hostname or "").lower()
    is_local_target = host in {"localhost", "127.0.0.1", "::1"}
    ollama_cmd = shutil.which("ollama")
    if not ollama_cmd:
        return False, "Ollama is not installed."
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
    keys = {"openai": None, "anthropic": None, "gemini": [], "deepseek": None, "sealion": None}
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
            "deepseek": r"DeepSeek\s*:\s*(sk-[A-Za-z0-9_\-]+)",
            "sealion": r"SEA-LION\s*:\s*(sk-[A-Za-z0-9_\-]+)",
        }
        for k, pat in patterns.items():
            m = re.search(pat, text)
            if m:
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

LLAMA_BASE_URL = os.getenv("LLAMA_BASE_URL", "http://localhost:11434")
LLAMA_STRONG_MODEL = os.getenv("LLAMA_STRONG_MODEL", "llama3.2:7b")
LLAMA_WEAK_MODEL = os.getenv("LLAMA_WEAK_MODEL", "llama3.2:1b")
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
    MODEL_RUNS.append({"name": "gpt_strong", "provider": "openai_responses",
                        "client": OpenAI(api_key=OPENAI_API_KEY), "api_key": OPENAI_API_KEY,
                        "model": GPT_STRONG_MODEL, "max_tokens": 64,
                        "reasoning_effort": GPT_STRONG_REASONING_EFFORT, "sleep": 0.5})
    MODEL_RUNS.append({"name": "gpt_weak", "provider": "openai_responses",
                        "client": OpenAI(api_key=OPENAI_API_KEY), "api_key": OPENAI_API_KEY,
                        "model": GPT_WEAK_MODEL, "max_tokens": 64,
                        "reasoning_effort": GPT_WEAK_REASONING_EFFORT, "sleep": 0.5})

if ANTHROPIC_API_KEY:
    MODEL_RUNS.append({"name": "claude_strong", "provider": "anthropic",
                        "client": anthropic.Anthropic(api_key=ANTHROPIC_API_KEY),
                        "api_key": ANTHROPIC_API_KEY, "model": CLAUDE_STRONG_MODEL, "sleep": 0.5})
    MODEL_RUNS.append({"name": "claude_weak", "provider": "anthropic",
                        "client": anthropic.Anthropic(api_key=ANTHROPIC_API_KEY),
                        "api_key": ANTHROPIC_API_KEY, "model": CLAUDE_WEAK_MODEL, "sleep": 0.5})

if GEMINI_API_KEYS:
    MODEL_RUNS.append({"name": "gemini_strong", "provider": "gemini", "client": None,
                        "api_keys": GEMINI_API_KEYS, "model": GEMINI_STRONG_MODEL,
                        "fallback_models": GEMINI_STRONG_FALLBACK_MODELS, "sleep": 0.5})
    MODEL_RUNS.append({"name": "gemini_weak", "provider": "gemini", "client": None,
                        "api_keys": GEMINI_API_KEYS, "model": GEMINI_WEAK_MODEL, "sleep": 0.5})

if DEEPSEEK_API_KEY:
    MODEL_RUNS.append({"name": "deepseek_strong", "provider": "openai",
                        "client": OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com"),
                        "api_key": DEEPSEEK_API_KEY, "base_url": "https://api.deepseek.com",
                        "model": DEEPSEEK_STRONG_MODEL, "sleep": 0.3})
    MODEL_RUNS.append({"name": "deepseek_weak", "provider": "openai",
                        "client": OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com"),
                        "api_key": DEEPSEEK_API_KEY, "base_url": "https://api.deepseek.com",
                        "model": DEEPSEEK_WEAK_MODEL, "sleep": 0.3})

llama_available, llama_status = _ensure_ollama_runtime(LLAMA_BASE_URL)
print("Llama/Ollama status:", llama_status)
if llama_available:
    MODEL_RUNS.append({"name": "llama_strong", "provider": "ollama", "client": None,
                        "model": LLAMA_STRONG_MODEL, "sleep": 0.2, "base_url": LLAMA_BASE_URL})
    MODEL_RUNS.append({"name": "llama_weak", "provider": "ollama", "client": None,
                        "model": LLAMA_WEAK_MODEL, "sleep": 0.2, "base_url": LLAMA_BASE_URL})
else:
    print("Llama skipped:", llama_status)

if SEA_LION_API_KEY:
    MODEL_RUNS.append({"name": "sealion_strong", "provider": "openai",
                        "client": OpenAI(api_key=SEA_LION_API_KEY, base_url=SEA_LION_BASE_URL),
                        "api_key": SEA_LION_API_KEY, "base_url": SEA_LION_BASE_URL,
                        "model": SEA_LION_STRONG_MODEL, "max_tokens": 512, "sleep": 0.3})
    MODEL_RUNS.append({"name": "sealion_weak", "provider": "openai",
                        "client": OpenAI(api_key=SEA_LION_API_KEY, base_url=SEA_LION_BASE_URL),
                        "api_key": SEA_LION_API_KEY, "base_url": SEA_LION_BASE_URL,
                        "model": SEA_LION_WEAK_MODEL, "sleep": 0.3})

if not MODEL_RUNS:
    raise ValueError("No model keys/runtimes configured.")

print("Models to run:", [m["name"] for m in MODEL_RUNS])
MODEL_VERSION_MAP = {m["name"].upper(): m["model"] for m in MODEL_RUNS}

# ── Load data — all 187 rows, mask via Particle column ───────────────────────
DATA_PATH = Path("../Datasets/GOLD_187.csv")
df_full = pd.read_csv(DATA_PATH)

# Masking logic (per spec):
#   Particle='kan' → regex (?i)\bkan\b → [___]
#   Particle='ke'  → regex (?i)\bke\b  → [___]
#   Particle='neutral' → text already has [___], use as-is
def mask_particle(text: str, particle: str) -> str:
    if particle == "neutral":
        return text  # [___] already present
    masked, n = re.subn(rf"(?i)\b{re.escape(particle)}\b", "[___]", text, count=1)
    if n == 0:
        masked = re.sub(re.escape(particle), "[___]", text, count=1, flags=re.IGNORECASE)
    return masked

def _get_gt_particle(row) -> str:
    if row["Particle"] in ("kan", "ke"):
        return row["Particle"]
    if row["Particle"] == "neutral":
        return "neutral"
    m = re.search(r"removed\s+(\w+)", str(row["Sentence_Type"]), re.IGNORECASE)
    return m.group(1).lower() if m else ""

df_natural = df_full.copy().reset_index(drop=True)
df_natural["Text_Masked"] = df_natural.apply(
    lambda row: mask_particle(row["Text"], row["Particle"]), axis=1
)
df_natural["GT_Particle"] = df_natural.apply(_get_gt_particle, axis=1)
df_natural = df_natural[df_natural["GT_Particle"].isin(["kan", "ke", "neutral"])].reset_index(drop=True)

print(f"Total rows for Test 2a: {len(df_natural)}")
print(f"GT distribution:\n{df_natural['GT_Particle'].value_counts().to_string()}")

# Validate masking worked
unmasked = df_natural[~df_natural["Text_Masked"].str.contains(r"\[___\]", na=False)]
if len(unmasked) > 0:
    print(f"WARNING: {len(unmasked)} rows could not be masked:")
    for _, r in unmasked.iterrows():
        print(f"  Text: {r['Text'][:80]}  |  Particle: {r['Particle']}")

SAMPLE_N = int(os.getenv("SAMPLE_N", "5"))
eval_df = df_natural.head(SAMPLE_N).copy() if SAMPLE_N else df_natural.copy()
results = eval_df.copy()

print(f"\nEvaluating {len(eval_df)} rows (SAMPLE_N={SAMPLE_N})")
display(eval_df[["Text_Masked", "Particle"]].head(3))

# ── LLM helpers (same as Test 1a) ─────────────────────────────────────────────

def _extract_particle(raw):
    """Extract 'ke', 'kan', or 'neutral' from raw model output."""
    text = str(raw or "").strip().lower()
    # Strip thinking tags (e.g. </think>) and surrounding whitespace/special chars
    text = re.sub(r'</?think>', '', text).strip()
    text = re.sub(r'^[\W_]+', '', text).strip()  # strip leading non-word chars
    # Prefer exact match first
    if text in {"ke", "kan", "neutral"}:
        return text
    # Search for first occurrence as a whole word (neutral before kan/ke to avoid partial matches)
    for p in ["neutral", "kan", "ke"]:
        if re.search(rf"\b{p}\b", text):
            return p
    return text  # return as-is for error analysis


def _is_fatal_error(error):
    text = str(error).lower()
    return any(m in text for m in [
        "model_not_found", "does not exist", "unsupported parameter",
        "invalid_request_error", "api key not valid", "permission",
        "authentication", "insufficient_quota",
    ])


def _is_rate_limit_error(error):
    text = str(error).lower()
    return "429" in text or "rate_limit" in text or "ratelimit" in text or "too many requests" in text


def _is_transient_error(error):
    text = str(error).lower()
    return any(m in text for m in [
        "503", "service unavailable", "resource_exhausted",
        "429", "deadline", "timeout", "internal",
    ])


def _call_gemini_legacy(api_key, model_name, prompt_text, constraint_text):
    import google.generativeai as genai_legacy
    genai_legacy.configure(api_key=api_key)
    model = genai_legacy.GenerativeModel(model_name=model_name, system_instruction=constraint_text)
    response = model.generate_content(
        prompt_text,
        generation_config={"temperature": 0, "max_output_tokens": 8},
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
                client = genai.Client(api_key=api_key, http_options=genai_types.HttpOptions(timeout=120))
                response = client.models.generate_content(
                    model=model_name,
                    config=genai_types.GenerateContentConfig(
                        system_instruction=constraint_text, max_output_tokens=8, temperature=0,
                    ),
                    contents=prompt_text,
                )
                raw_text = (response.text or "").strip()
                if raw_text:
                    return raw_text
                last_error = RuntimeError(f"Empty Gemini response for model {model_name}")
            except Exception as error:
                last_error = error
                if _is_fatal_error(error):
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


def call_llm(run_cfg, prompt_text, constraint_text, retries=1, delay=1.0,
             output_try=1, log_meta=None, log_sink=None):
    for attempt in range(retries):
        try:
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
                    "max_output_tokens": run_cfg.get("max_tokens", 64),
                }
                reasoning_effort = run_cfg.get("reasoning_effort")
                if reasoning_effort:
                    req["reasoning"] = {"effort": reasoning_effort}
                response = client.responses.create(**req)
                raw = (getattr(response, "output_text", "") or "").strip()

            elif provider == "openai":
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": constraint_text},
                        {"role": "user", "content": prompt_text},
                    ],
                    max_completion_tokens=run_cfg.get("max_tokens", 8),
                )
                raw = (response.choices[0].message.content or "").strip()

            elif provider == "anthropic":
                response = client.messages.create(
                    model=model,
                    system=constraint_text,
                    max_tokens=8,
                    messages=[{"role": "user", "content": prompt_text}],
                )
                raw = response.content[0].text.strip()

            elif provider == "gemini":
                raw = _call_gemini_with_fallback(run_cfg, prompt_text, constraint_text)

            elif provider == "ollama":
                url = f"{run_cfg['base_url'].rstrip('/')}/api/generate"
                response = requests.post(
                    url,
                    json={"model": model, "prompt": f"{constraint_text}\n\n{prompt_text}",
                          "stream": False, "options": {"temperature": 0}},
                    timeout=60,
                )
                response.raise_for_status()
                raw = response.json().get("response", "").strip()

            else:
                raise ValueError(f"Unsupported provider: {provider}")

            label = _extract_particle(raw)
            status = "ok" if str(label).strip() else "empty"
            _append_io_log({
                "timestamp": _utc_now_iso(), "model_alias": run_cfg["name"],
                "provider": provider, "model": model, "output_try": output_try,
                "transport_try": attempt + 1, "input": prompt_text,
                "constraint": constraint_text, "raw_output": str(raw),
                "parsed_output": str(label), "status": status, "meta": log_meta or {},
            }, log_sink=log_sink)
            if not str(label).strip():
                raise RuntimeError("Empty model response")
            return label

        except Exception as e:
            _append_io_log({
                "timestamp": _utc_now_iso(), "model_alias": run_cfg["name"],
                "provider": run_cfg.get("provider"), "model": run_cfg.get("model"),
                "output_try": output_try, "transport_try": attempt + 1, "input": prompt_text,
                "constraint": constraint_text, "raw_output": "", "parsed_output": "",
                "status": "error", "error": str(e), "meta": log_meta or {},
            }, log_sink=log_sink)
            if _is_fatal_error(e):
                return f"ERROR_FATAL: {e}"
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
            else:
                return f"ERROR: {e}"


def predict_with_output_retry(run_cfg, prompt_text, constraint_text,
                               max_output_retries=6, delay=0.6, log_meta=None, log_sink=None):
    last_pred = "ERROR: Empty model response"
    for output_try in range(1, max_output_retries + 1):
        pred = call_llm(run_cfg, prompt_text, constraint_text, retries=1, delay=0.4,
                        output_try=output_try, log_meta=log_meta, log_sink=log_sink)
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
        task["run_cfg"], task["prompt_text"], task["constraint_text"],
        max_output_retries=task.get("max_output_retries", 3),
        delay=task.get("delay", 0.6), log_meta=task.get("log_meta"), log_sink=local_logs,
    )
    return {"row_idx": task["row_idx"], "model_name": task["model_name"],
            "prediction": pred, "logs": local_logs}


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
        return [_predict_task(task)
                for task in tqdm(tasks, total=len(tasks), desc=desc, file=sys.stdout, dynamic_ncols=True)]
    results_list = []
    print(f"Launching {len(tasks)} tasks across {worker_count} threads", flush=True)
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(_guarded, task) for task in tasks]
        for future in tqdm(as_completed(futures), total=len(futures), desc=desc, file=sys.stdout, dynamic_ncols=True):
            results_list.append(future.result())
    return results_list


# ── Smoke test (1 sample) ─────────────────────────────────────────────────────

def smoke_test_one_sample():
    """Run inference on a single sample to validate all models."""
    print("\n" + "=" * 70)
    print("TEST 2a — Smoke test: 1 sample")
    print("=" * 70)

    sample = eval_df.iloc[0]
    prompt_text = build_particle_gen_prompt(sample["Text_Masked"], sample) + PARTICLE_GEN_STRICT_TAIL

    print(f"\nSample text (masked): {sample['Text_Masked'][:100]}")
    print(f"Ground truth particle: {sample['Particle']}")
    print(f"\nPrompt sent to models:\n{'─'*60}")
    print(prompt_text)
    print('─' * 60)

    tasks = [
        {
            "row_idx": 0,
            "model_name": run_cfg["name"],
            "run_cfg": _worker_run_cfg(run_cfg),
            "prompt_text": prompt_text,
            "constraint_text": PARTICLE_GEN_SYSTEM_CONSTRAINT,
            "max_output_retries": 3,
            "delay": 0.2,
            "log_meta": {"phase": "smoke_test_2a"},
        }
        for run_cfg in MODEL_RUNS
    ]

    smoke_results = _run_parallel_predictions(tasks, desc="smoke_test_2a")
    smoke_map = {item["model_name"]: item for item in smoke_results}

    print(f"\nGround truth: {sample['Particle']}")
    for run_cfg in MODEL_RUNS:
        pred = smoke_map[run_cfg["name"]]["prediction"]
        correct = "✓" if pred == sample["Particle"] else "✗"
        if isinstance(pred, str) and pred.startswith("ERROR_FATAL"):
            MODEL_FATAL_ERRORS[run_cfg["name"]] = pred
        print(f"  {run_cfg['name']:20s}: {pred}  {correct}")

    IO_LOGS.extend(log for item in smoke_results for log in item["logs"])
    _save_io_logs()
    print(f"\nSmoke test complete. IO logs saved → {IO_LOG_JSON_LATEST}")


# ── Full evaluation run ───────────────────────────────────────────────────────

def run_particle_generation():
    """Run all configured models on all eval_df rows and append prediction columns."""
    print(f"\n{'─'*60}")
    print(f"  Task      : Particle Generation (Test 2a)")
    print(f"  Labels    : {PARTICLE_LABELS}")
    print(f"  N samples : {len(eval_df)}")
    print(f"{'─'*60}")

    tasks = []
    active_model_names = []

    for run_cfg in MODEL_RUNS:
        worker_cfg = _worker_run_cfg(run_cfg)
        fatal_note = MODEL_FATAL_ERRORS.get(run_cfg["name"])
        if fatal_note:
            results[f"{run_cfg['name']}_particle_gen"] = [fatal_note] * len(eval_df)
            print(f"  {run_cfg['name'].upper()} skipped → cached fatal error", flush=True)
            continue
        active_model_names.append(run_cfg["name"])
        for row_idx, row in enumerate(eval_df.itertuples(index=False)):
            row_series = eval_df.iloc[row_idx]
            prompt_text = (
                build_particle_gen_prompt(row_series["Text_Masked"], row_series)
                + PARTICLE_GEN_STRICT_TAIL
            )
            tasks.append({
                "row_idx": row_idx,
                "model_name": run_cfg["name"],
                "run_cfg": worker_cfg,
                "prompt_text": prompt_text,
                "constraint_text": PARTICLE_GEN_SYSTEM_CONSTRAINT,
                "max_output_retries": 6,
                "delay": 0.6,
                "log_meta": {"phase": "particle_gen_2a", "row_idx": row_idx},
            })

    parallel_results = _run_parallel_predictions(tasks, desc="particle_gen_2a")

    preds_by_model = {name: [None] * len(eval_df) for name in active_model_names}
    new_logs = []
    for item in parallel_results:
        preds_by_model[item["model_name"]][item["row_idx"]] = item["prediction"]
        new_logs.extend(item["logs"])

    for run_cfg in MODEL_RUNS:
        model_name = run_cfg["name"]
        if model_name not in preds_by_model:
            continue
        col = f"{model_name}_particle_gen"
        results[col] = preds_by_model[model_name]
        sample = preds_by_model[model_name][:3]
        print(f"  {model_name.upper()} done → sample: {sample}", flush=True)

    IO_LOGS.extend(new_logs)
    _save_io_logs()


def save_results():
    output_csv = Path("../Final Metrics/round1_2a_predictions.csv")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_csv, index=False)
    print(f"\nSaved predictions → {output_csv.resolve()}")

    # Accuracy report
    model_names = [m["name"] for m in MODEL_RUNS]
    gt = results["GT_Particle"]
    rows = []
    for model_name in model_names:
        col = f"{model_name}_particle_gen"
        if col not in results.columns:
            continue
        pred = results[col]
        acc = (gt == pred).mean()
        errors = pred.astype(str).str.startswith("ERROR").sum()
        rows.append({
            "Model": MODEL_VERSION_MAP.get(model_name.upper(), model_name.upper()),
            "Accuracy": round(float(acc), 4),
            "Errors": int(errors),
        })

    if rows:
        acc_df = pd.DataFrame(rows).sort_values(["Accuracy", "Model"], ascending=[False, True])
        print("\n" + "=" * 55)
        print("  Test 2a — Particle Generation Accuracy")
        print("=" * 55)
        display(acc_df)

        # Write markdown summary
        md_path = Path("../Final Metrics/round1_2a_accuracy_summary.md")
        header = "| Model | Accuracy | Errors |"
        sep = "|---|---|---|"
        lines = [header, sep]
        for _, r in acc_df.iterrows():
            lines.append(f"| {r['Model']} | {r['Accuracy']:.4f} | {r['Errors']} |")
        md_path.write_text("# Test 2a — Particle Generation Accuracy\n\n" + "\n".join(lines) + "\n",
                            encoding="utf-8")
        print(f"Saved markdown summary → {md_path.resolve()}")

    display(results[["Text_Masked", "GT_Particle"] +
                     [c for c in results.columns if c.endswith("_particle_gen")]].head(SAMPLE_N))


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__" or True:
    # Step 1: smoke test on 1 sample
    smoke_test_one_sample()

    if os.getenv("STOP_AFTER_SMOKE", "0") == "1":
        _save_io_logs()
        print("STOP_AFTER_SMOKE=1 — exiting after smoke test.")
        raise SystemExit(0)

    # Step 2: full run on SAMPLE_N samples
    run_particle_generation()
    save_results()
