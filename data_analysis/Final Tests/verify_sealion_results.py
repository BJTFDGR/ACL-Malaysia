"""
Verification script: re-compute Gemma-SEA-LION and Llama-SEA-LION accuracy
across all 6 tasks from raw predictions CSVs.

Models:
  sealion_weak   = Gemma-SEA-LION-v4-27B-IT
  sealion_strong = Llama-SEA-LION-v3.5-70B-R

Methodology notes:
- Task 1a: denom = rows where GT is non-NaN (NaN predictions counted as wrong).
  Llama MS columns contain raw CoT reasoning chains; labels are extracted before
  scoring using the same approach as round1_sealion_retry.py.
- Tasks 2a/2c/1b/1c/2b: exact-match, both sides non-NaN.
"""

import re
import pandas as pd
from pathlib import Path

METRICS_DIR = Path(__file__).parent / "Final Metrics"

# ── Label normalisation (same logic as the evaluation notebook) ───────────────
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


def normalize(val, label_set, attribute=None):
    if pd.isna(val):
        return None   # signal: no prediction
    raw = str(val).strip()
    v = raw.lower()
    if v in {"na", "n/a", "n.a.", "none", "tiada", "tidak ada"} and "N/A" in label_set:
        return "N/A"
    for lbl in sorted(label_set, key=len, reverse=True):
        if lbl.lower() == v:
            return lbl
    if attribute in ATTRIBUTE_ALIASES:
        for alias, canonical in ATTRIBUTE_ALIASES[attribute].items():
            if alias == v:
                return canonical
    return raw   # unknown value → will count as wrong


def extract_label_from_cot(raw: str, label_set: list, attribute: str) -> str:
    """
    Extract a valid label from a Llama CoT reasoning chain.
    Strips <think>...</think>, then searches for the label keywords.
    Returns the canonical label string or the raw value if extraction fails.
    """
    if pd.isna(raw):
        return None
    text = str(raw).strip()
    # Strip <think> blocks (with Unicode escapes like ĊĊ</think>)
    text = re.sub(r"</?think>", " ", text, flags=re.IGNORECASE)
    # Strip non-ASCII (ĊĊ, Ġ, etc. are Unicode control/space chars)
    text_ascii = re.sub(r"[^\x00-\x7F]+", " ", text).strip()

    # Build alias → canonical mapping (lower-cased)
    candidates = {}
    for lbl in label_set:
        candidates[lbl.lower()] = lbl
    if attribute in ATTRIBUTE_ALIASES:
        for alias, canonical in ATTRIBUTE_ALIASES[attribute].items():
            candidates[alias.lower()] = canonical

    # Search from the END of the text (final answer usually at end)
    search_text = text_ascii.lower()
    # Try longest match first to avoid "ke" matching inside "mencari pengesahan" etc.
    for alias in sorted(candidates.keys(), key=len, reverse=True):
        if re.search(r"\b" + re.escape(alias) + r"\b", search_text):
            return candidates[alias]
    return raw   # unchanged → will count as wrong


def simple_acc(df, gt_col, pred_col):
    """Exact-match accuracy, ignoring rows where gt or pred is NaN."""
    mask = df[gt_col].notna() & df[pred_col].notna()
    sub = df[mask]
    if len(sub) == 0:
        return float("nan"), 0
    return (sub[gt_col].str.strip() == sub[pred_col].str.strip()).mean(), len(sub)


# ─────────────────────────────────────────────────────────────────────────────
# Task 1a — Particle Attribute Classification
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("TASK 1a — Particle Attribute Classification")
print("=" * 65)
print("  Methodology: denom = rows where GT is non-NaN.")
print("  NaN predictions count as wrong. CoT outputs extracted.")

df1a = pd.read_csv(METRICS_DIR / "round1_1a_predictions.csv")

models = {"sealion_strong": "Llama-SEA-LION-v3.5-70B-R", "sealion_weak": "Gemma-SEA-LION-v4-27B-IT"}
EVAL_VARIANTS = {"EN": "", "MS": "_ms"}


def score_1a_attr(df, gt_col, pred_col, attr, labels, is_cot=False):
    """
    Score attribute classification.
    Denom = rows where GT is non-NaN.
    NaN or unparseable predictions count as wrong.
    """
    gt_raw = df[gt_col]
    pred_raw = df[pred_col]

    rows = []
    for gt_val, pred_val in zip(gt_raw, pred_raw):
        gt_norm = normalize(gt_val, labels, attr)
        if gt_norm is None:
            continue   # skip rows where GT is missing
        if is_cot:
            pred_norm = extract_label_from_cot(pred_val, labels, attr)
            pred_norm = normalize(pred_norm, labels, attr) if pred_norm else None
        else:
            pred_norm = normalize(pred_val, labels, attr)
        correct = (pred_norm is not None) and (gt_norm == pred_norm)
        rows.append(int(correct))

    if not rows:
        return float("nan"), 0
    return sum(rows) / len(rows), len(rows)


def has_cot(series):
    """True if the column contains Llama CoT reasoning chains."""
    sample = series.dropna().astype(str)
    return sample.str.contains(r"</think>", regex=False).any()


for variant, suffix in EVAL_VARIANTS.items():
    print(f"\n  [{variant}]")
    for model_key, model_label in models.items():
        accs = {}
        for attr, labels in ATTRIBUTE_LABEL_MAP.items():
            gt_col = attr
            pred_col = f"{model_key}_{attr}{suffix}"
            if pred_col not in df1a.columns:
                accs[attr] = float("nan")
                continue
            is_cot = has_cot(df1a[pred_col])
            acc, n = score_1a_attr(df1a, gt_col, pred_col, attr, labels, is_cot)
            accs[attr] = round(acc, 4)
            cot_flag = " [CoT extracted]" if is_cot else ""
            print(f"      {attr:<22} {acc:.4f}  (n={n}){cot_flag}")
        macro = pd.Series(accs).mean()
        print(f"    {model_label}")
        print(f"      {'MACRO AVG':<22} {macro:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# Task 2a — Particle Generation
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("TASK 2a — Particle Generation")
print("=" * 65)

df2a = pd.read_csv(METRICS_DIR / "round1_2a_predictions.csv")
for model_key, model_label in models.items():
    pred_col = f"{model_key}_particle_gen"
    acc, n = simple_acc(df2a, "GT_Particle", pred_col)
    print(f"  {model_label}: {acc:.4f}  (n={n})")

# ─────────────────────────────────────────────────────────────────────────────
# Task 2c — Particle Prediction
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("TASK 2c — Particle Prediction")
print("=" * 65)

df2c = pd.read_csv(METRICS_DIR / "round1_2c_predictions_full.csv")
for model_key, model_label in models.items():
    pred_col = f"{model_key}_Particle"
    acc, n = simple_acc(df2c, "GT_Particle", pred_col)
    print(f"  {model_label}: {acc:.4f}  (n={n})")

# ─────────────────────────────────────────────────────────────────────────────
# Task 1b — Macro-Function Unassisted
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("TASK 1b — Macro-Function Unassisted")
print("=" * 65)

df1b = pd.read_csv(METRICS_DIR / "round2_1b_predictions.csv")
for model_key, model_label in models.items():
    pred_col = f"{model_key}_macro_1b"
    acc, n = simple_acc(df1b, "Macro_Function", pred_col)
    print(f"  {model_label}: {acc:.4f}  (n={n})")

# ─────────────────────────────────────────────────────────────────────────────
# Task 1c — Macro-Function Attribute-Assisted
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("TASK 1c — Macro-Function Attribute-Assisted")
print("=" * 65)

df1c = pd.read_csv(METRICS_DIR / "round2_1c_predictions.csv")
for model_key, model_label in models.items():
    pred_col = f"{model_key}_macro_1c"
    acc, n = simple_acc(df1c, "Macro_Function", pred_col)
    print(f"  {model_label}: {acc:.4f}  (n={n})")

# ─────────────────────────────────────────────────────────────────────────────
# Task 2b — Function-Constrained Particle Generation
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("TASK 2b — Function-Constrained Particle Generation")
print("=" * 65)

df2b = pd.read_csv(METRICS_DIR / "round2_2b_predictions.csv")
for model_key, model_label in models.items():
    pred_col = f"{model_key}_particle_2b"
    acc, n = simple_acc(df2b, "GT_Particle", pred_col)
    print(f"  {model_label}: {acc:.4f}  (n={n})")

# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SUMMARY vs REPORTED VALUES")
print("=" * 65)
reported = {
    # (task, model_key): reported_value
    ("1a_EN_macro", "sealion_weak"):   round((0.7166 + 0.6116 + 0.5241 + 0.6898 + 0.6150) / 5, 4),
    ("1a_EN_macro", "sealion_strong"): round((0.3476 + 0.8678 + 0.2781 + 0.5401 + 0.4064) / 5, 4),
    ("1a_MS_macro", "sealion_weak"):   round((0.6791 + 0.7217 + 0.5294 + 0.7112 + 0.5989) / 5, 4),
    ("1a_MS_macro", "sealion_strong"): round((0.4332 + 0.5536 + 0.4332 + 0.3690 + 0.4171) / 5, 4),
    ("2a",  "sealion_weak"):   0.5027,
    ("2a",  "sealion_strong"): 0.4706,
    ("2c",  "sealion_weak"):   0.3369,
    ("2c",  "sealion_strong"): 0.3316,
    ("1b",  "sealion_weak"):   0.2299,
    ("1b",  "sealion_strong"): 0.1230,
    ("1c",  "sealion_weak"):   0.5882,
    ("1c",  "sealion_strong"): 0.4011,
    ("2b",  "sealion_weak"):   0.4866,
    ("2b",  "sealion_strong"): 0.4171,
}

def recompute_1a_macro(df, model_key, suffix):
    accs = []
    for attr, labels in ATTRIBUTE_LABEL_MAP.items():
        pred_col = f"{model_key}_{attr}{suffix}"
        if pred_col not in df.columns:
            continue
        is_cot = has_cot(df[pred_col])
        acc, n = score_1a_attr(df, attr, pred_col, attr, labels, is_cot)
        accs.append(acc)
    return round(sum(accs) / len(accs), 4) if accs else float("nan")

recomputed = {
    ("1a_EN_macro", "sealion_weak"):   recompute_1a_macro(df1a, "sealion_weak", ""),
    ("1a_EN_macro", "sealion_strong"): recompute_1a_macro(df1a, "sealion_strong", ""),
    ("1a_MS_macro", "sealion_weak"):   recompute_1a_macro(df1a, "sealion_weak", "_ms"),
    ("1a_MS_macro", "sealion_strong"): recompute_1a_macro(df1a, "sealion_strong", "_ms"),
    ("2a",  "sealion_weak"):   round(simple_acc(df2a, "GT_Particle", "sealion_weak_particle_gen")[0], 4),
    ("2a",  "sealion_strong"): round(simple_acc(df2a, "GT_Particle", "sealion_strong_particle_gen")[0], 4),
    ("2c",  "sealion_weak"):   round(simple_acc(df2c, "GT_Particle", "sealion_weak_Particle")[0], 4),
    ("2c",  "sealion_strong"): round(simple_acc(df2c, "GT_Particle", "sealion_strong_Particle")[0], 4),
    ("1b",  "sealion_weak"):   round(simple_acc(df1b, "Macro_Function", "sealion_weak_macro_1b")[0], 4),
    ("1b",  "sealion_strong"): round(simple_acc(df1b, "Macro_Function", "sealion_strong_macro_1b")[0], 4),
    ("1c",  "sealion_weak"):   round(simple_acc(df1c, "Macro_Function", "sealion_weak_macro_1c")[0], 4),
    ("1c",  "sealion_strong"): round(simple_acc(df1c, "Macro_Function", "sealion_strong_macro_1c")[0], 4),
    ("2b",  "sealion_weak"):   round(simple_acc(df2b, "GT_Particle", "sealion_weak_particle_2b")[0], 4),
    ("2b",  "sealion_strong"): round(simple_acc(df2b, "GT_Particle", "sealion_strong_particle_2b")[0], 4),
}

print(f"\n{'Task':<16} {'Model':<12} {'Reported':>10} {'Recomputed':>12} {'Match':>7}")
print("-" * 65)
all_match = True
for key in sorted(reported.keys()):
    task, model = key
    rep = reported[key]
    rec = recomputed[key]
    match = abs(rep - rec) < 0.0001 if not pd.isna(rec) else False
    if not match:
        all_match = False
    flag = "OK" if match else "MISMATCH"
    print(f"  {task:<14} {model:<12} {rep:>10.4f} {rec:>12.4f}  {flag}")

print()
if all_match:
    print("All values match. Results verified.")
else:
    print("Some mismatches found — see rows marked MISMATCH above.")
