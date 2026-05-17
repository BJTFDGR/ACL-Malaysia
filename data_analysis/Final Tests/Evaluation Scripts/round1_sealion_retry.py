"""
Targeted retry for Llama-SEA-LION-v3.5-70B-R error rows in Test 2a and 2c.

Actions:
  1. Fix the 93 </think>-wrapped rows in 2a (extract particle after </think>).
  2. Retry the 8 actual ERROR rows for sealion_strong in both 2a and 2c.
  3. Recalculate per-model accuracy and update the markdown summary files.
"""

import json
import os
import re
import sys
import time
import random
from pathlib import Path

import pandas as pd
from openai import OpenAI

# ── Config ────────────────────────────────────────────────────────────────────
FINAL_METRICS = Path("../Final Metrics")

CSV_2A = FINAL_METRICS / "round1_2a_predictions.csv"
CSV_2C = FINAL_METRICS / "round1_2c_predictions_full.csv"
MD_2A  = FINAL_METRICS / "round1_2a_accuracy_summary.md"
MD_2C  = FINAL_METRICS / "round1_2c_accuracy_summary_full.md"

SEA_LION_BASE_URL    = os.getenv("SEA_LION_BASE_URL", "https://api.sea-lion.ai/v1")
SEA_LION_STRONG_MODEL = "aisingapore/Llama-SEA-LION-v3.5-70B-R"

MAX_RETRIES = 8
RATE_LIMIT_SLEEP = 70  # seconds to wait after a 429

# ── Load API key ──────────────────────────────────────────────────────────────
def _extract_keys_from_notebook(nb_path: Path):
    try:
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        text = "\n".join(
            "\n".join(cell.get("source", []))
            for cell in nb.get("cells", [])
            if cell.get("cell_type") == "markdown"
        )
        m = re.search(r"SEA-LION\s*:\s*(sk-[A-Za-z0-9_\-]+)", text)
        if m:
            return m.group(1).strip()
    except Exception:
        pass
    return None

SEA_LION_API_KEY = (
    os.getenv("SEA_LION_API_KEY")
    or _extract_keys_from_notebook(Path("04_round1_1a_attribute_accuracy.ipynb"))
)
if not SEA_LION_API_KEY:
    print("ERROR: No SEA-LION API key found.", file=sys.stderr)
    sys.exit(1)

client = OpenAI(api_key=SEA_LION_API_KEY, base_url=SEA_LION_BASE_URL)

# ── Helpers ───────────────────────────────────────────────────────────────────
PARTICLE_LABELS = {"kan", "ke"}

def _extract_label(raw: str) -> str:
    """Extract 'kan' or 'ke' from raw LLM output."""
    text = str(raw or "").strip().lower()
    # Strip <think>...</think> blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # Prefer whole-word match
    for label in ["kan", "ke"]:
        if re.search(r"\b" + re.escape(label) + r"\b", text):
            return label
    # Substring fallback
    for label in ["kan", "ke"]:
        if label in text:
            return label
    return str(raw or "").strip()


def _is_error_value(val: str) -> bool:
    return str(val).startswith("ERROR:")


def _call_sealion(prompt_text: str, constraint_text: str) -> str:
    """Call sealion_strong with retries. Returns the raw response or ERROR:..."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=SEA_LION_STRONG_MODEL,
                messages=[
                    {"role": "system", "content": constraint_text},
                    {"role": "user", "content": prompt_text},
                ],
                max_completion_tokens=512,
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as e:
            err_str = str(e)
            is_rate_limit = "429" in err_str or "rate_limit" in err_str.lower()
            print(f"  [attempt {attempt}/{MAX_RETRIES}] Error: {err_str[:120]}")
            if attempt == MAX_RETRIES:
                return f"ERROR: {err_str}"
            sleep_secs = (RATE_LIMIT_SLEEP + random.randint(0, 15)) if is_rate_limit else 5
            print(f"  Sleeping {sleep_secs}s...")
            time.sleep(sleep_secs)
    return "ERROR: max retries exceeded"


# ── 2a: fix </think> wrapped rows + retry 8 error rows ───────────────────────
print("=" * 60)
print("Test 2a: fixing </think> rows and retrying 8 error rows")
print("=" * 60)

# Delayed import of 2a prompt builder to avoid running the full script
sys.path.insert(0, str(Path(".").resolve()))

# Inline the prompt builder from round1_2a (avoids executing the full script)
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

PARTICLE_GEN_SYSTEM_CONSTRAINT = (
    "You are a linguist specialising in colloquial Malay discourse particles. "
    "You must output exactly one word — either \"ke\" or \"kan\" — and nothing else."
)
PARTICLE_GEN_STRICT_TAIL = "\n\nReturn exactly one word from this set and nothing else: ke, kan"


def build_particle_gen_prompt(text_masked: str, row: pd.Series) -> str:
    def _get_desc(attr, val):
        return ATTRIBUTE_DESCRIPTIONS.get(attr, {}).get(str(val).strip(), "")

    attr_block = "\n".join([
        f"  Epistemic Stance: {row['Epistemic_Stance']}\n    → {_get_desc('Epistemic_Stance', row['Epistemic_Stance'])}",
        f"  Particle Position: {row['Particle_Position']}\n    → {_get_desc('Particle_Position', row['Particle_Position'])}",
        f"  Listener Agreement: {row['Listener_Agreement']}\n    → {_get_desc('Listener_Agreement', row['Listener_Agreement'])}",
        f"  Emotion: {row['Emotion']}\n    → {_get_desc('Emotion', row['Emotion'])}",
        f"  Question Type: {row['Question_Type']}\n    → {_get_desc('Question_Type', row['Question_Type'])}",
    ])

    return f"""\
You are given a Malay sentence in which one discourse particle has been replaced with [___].
Your task is to predict which particle — either "ke" or "kan" — belongs in the [___] slot.

Sentence: "{text_masked}"

Discourse-context attributes for this sentence:
{attr_block}

Particle meanings:
  ke  : signals genuine uncertainty or invites the listener to confirm something the speaker is unsure about.
  kan : signals assumed shared knowledge and seeks light confirmation of something the speaker already believes ("right?").

Given the sentence and its discourse-context attributes, the single most likely particle to fill [___] is:"""


df2a = pd.read_csv(CSV_2A)
col_2a = "sealion_strong_particle_gen"

# Step 1: fix </think> wrapped rows
think_mask = df2a[col_2a].str.contains("</think>", na=False)
n_think = think_mask.sum()
print(f"Fixing {n_think} </think>-wrapped rows in 2a...")
for idx in df2a.index[think_mask]:
    raw = df2a.at[idx, col_2a]
    # Strip everything up to and including </think>
    cleaned = re.sub(r".*?</think>", "", raw, flags=re.DOTALL).strip()
    # Also strip leading control/whitespace chars
    cleaned = cleaned.lstrip()
    extracted = _extract_label(cleaned)
    df2a.at[idx, col_2a] = extracted
    print(f"  row {idx}: '{raw[:60]}...' → '{extracted}'")

# Step 2: retry actual ERROR rows
error_mask_2a = df2a[col_2a].str.startswith("ERROR:", na=False)
error_idx_2a = df2a.index[error_mask_2a].tolist()
print(f"\nRetrying {len(error_idx_2a)} ERROR rows in 2a: {error_idx_2a}")
for idx in error_idx_2a:
    row = df2a.loc[idx]
    prompt = build_particle_gen_prompt(row["Text_Masked"], row) + PARTICLE_GEN_STRICT_TAIL
    print(f"\n  Row {idx} (GT={row['Particle']}) — calling sealion_strong...")
    raw = _call_sealion(prompt, PARTICLE_GEN_SYSTEM_CONSTRAINT)
    result = _extract_label(raw)
    print(f"  Raw: {raw[:80]!r} → Extracted: {result!r}")
    df2a.at[idx, col_2a] = result

# Save updated 2a CSV
df2a.to_csv(CSV_2A, index=False)
print(f"\nSaved updated 2a CSV → {CSV_2A}")


# ── 2c: retry 8 error rows ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Test 2c: retrying 8 ERROR rows")
print("=" * 60)

PARTICLE_PREDICT_PROMPT = """\
You are a linguist specialising in colloquial Malay discourse particles. \
A discourse particle has been removed from the Malay sentence below and replaced with [___]. \
Your task is to predict which particle, either "ke" or "kan", belongs in that slot.

Particle meanings:
  ke  : signals genuine uncertainty or invites the listener to confirm something the speaker is unsure about.
  kan : signals assumed shared knowledge and seeks light confirmation of something the speaker already believes ("right?").


Speaker: "{TEXT}"


Given the two candidate particles "kan" and "ke", the single most likely particle to fill [___] is:\
"""
CONSTRAINT_2C = (
    "You must output exactly one word — either \"kan\" or \"ke\" — and nothing else."
)
STRICT_TAIL_2C = "\n\nReturn exactly one word: kan  or  ke"

df2c = pd.read_csv(CSV_2C)
col_2c = "sealion_strong_Particle"

error_mask_2c = df2c[col_2c].str.startswith("ERROR:", na=False)
error_idx_2c = df2c.index[error_mask_2c].tolist()
print(f"Retrying {len(error_idx_2c)} ERROR rows in 2c: {error_idx_2c}")

for idx in error_idx_2c:
    row = df2c.loc[idx]
    prompt = PARTICLE_PREDICT_PROMPT.format(TEXT=row["Text"]) + STRICT_TAIL_2C
    print(f"\n  Row {idx} (GT={row['GT_Particle']}) — calling sealion_strong...")
    raw = _call_sealion(prompt, CONSTRAINT_2C)
    result = _extract_label(raw)
    print(f"  Raw: {raw[:80]!r} → Extracted: {result!r}")
    df2c.at[idx, col_2c] = result

df2c.to_csv(CSV_2C, index=False)
print(f"\nSaved updated 2c CSV → {CSV_2C}")


# ── Recalculate accuracy ───────────────────────────────────────────────────────
def calc_accuracy(df, pred_col, gt_col):
    valid = df[pred_col].isin(["kan", "ke"])
    correct = (df.loc[valid, pred_col] == df.loc[valid, gt_col]).sum()
    errors = (~valid).sum()
    total = len(df)
    acc = correct / total if total > 0 else 0.0
    return acc, int(errors)


MODEL_COLS_2A = [
    ("gpt-5",                         "gpt_strong",    "gpt_strong_particle_gen"),
    ("gpt-5.4-mini",                  "gpt_weak",      "gpt_weak_particle_gen"),
    ("claude-sonnet-4-6",               "claude_strong", "claude_strong_particle_gen"),
    ("claude-haiku-4-5",              "claude_weak",   "claude_weak_particle_gen"),
    ("gemini-3.1-pro-preview",        "gemini_strong", "gemini_strong_particle_gen"),
    ("gemini-3.1-flash-lite",         "gemini_weak",   "gemini_weak_particle_gen"),
    ("deepseek-v4-pro",               "deepseek_strong","deepseek_strong_particle_gen"),
    ("deepseek-v4-flash",             "deepseek_weak", "deepseek_weak_particle_gen"),
    ("aisingapore/Llama-SEA-LION-v3.5-70B-R", "sealion_strong", "sealion_strong_particle_gen"),
    ("aisingapore/Gemma-SEA-LION-v4-27B-IT",  "sealion_weak",   "sealion_weak_particle_gen"),
]

MODEL_COLS_2C = [
    ("gpt-5",                         "gpt_strong",    "gpt_strong_Particle"),
    ("gpt-5.4-mini",                  "gpt_weak",      "gpt_weak_Particle"),
    ("claude-sonnet-4-6",             "claude_strong", "claude_strong_Particle"),
    ("claude-haiku-4-5",              "claude_weak",   "claude_weak_Particle"),
    ("gemini-3.1-pro-preview",        "gemini_strong", "gemini_strong_Particle"),
    ("gemini-3.1-flash-lite",         "gemini_weak",   "gemini_weak_Particle"),
    ("deepseek-v4-pro",               "deepseek_strong","deepseek_strong_Particle"),
    ("deepseek-v4-flash",             "deepseek_weak", "deepseek_weak_Particle"),
    ("aisingapore/Llama-SEA-LION-v3.5-70B-R", "sealion_strong", "sealion_strong_Particle"),
    ("aisingapore/Gemma-SEA-LION-v4-27B-IT",  "sealion_weak",   "sealion_weak_Particle"),
]

print("\n" + "=" * 60)
print("Recalculating accuracy tables")
print("=" * 60)

# 2a accuracy
rows_2a = []
for model_name, alias, col in MODEL_COLS_2A:
    if col not in df2a.columns:
        continue
    acc, errs = calc_accuracy(df2a, col, "Particle")
    rows_2a.append({"Model": model_name, "Alias": alias, "Accuracy": acc, "Errors": errs})

acc_2a = pd.DataFrame(rows_2a).sort_values("Accuracy", ascending=False).reset_index(drop=True)
print("\n  Test 2a:")
print(acc_2a.to_string(index=False))

# 2c accuracy
rows_2c = []
for model_name, alias, col in MODEL_COLS_2C:
    if col not in df2c.columns:
        continue
    acc, errs = calc_accuracy(df2c, col, "GT_Particle")
    rows_2c.append({"Model": model_name, "Alias": alias, "Accuracy": acc, "Errors": errs})

acc_2c = pd.DataFrame(rows_2c).sort_values("Accuracy", ascending=False).reset_index(drop=True)
print("\n  Test 2c:")
print(acc_2c.to_string(index=False))


# ── Write markdown tables ──────────────────────────────────────────────────────
def df_to_md_table(df: pd.DataFrame) -> str:
    cols = df.columns.tolist()
    header = "| " + " | ".join(cols) + " |"
    sep    = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                cells.append(f"{v:.4f}")
            else:
                cells.append(str(v))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep] + rows)


md_2a_content = f"# Test 2a — Particle Generation Accuracy\n\n{df_to_md_table(acc_2a[['Model','Accuracy','Errors']])}\n"
MD_2A.write_text(md_2a_content, encoding="utf-8")
print(f"\nSaved 2a markdown → {MD_2A}")

md_2c_content = f"# Test 2c — Unconstrained Baseline Particle Prediction\n\n{df_to_md_table(acc_2c)}\n"
MD_2C.write_text(md_2c_content, encoding="utf-8")
print(f"Saved 2c markdown → {MD_2C}")

print("\nDone.")
