"""
Retry all gpt_strong ERROR rows across 1b, 1c, 2b with max_tokens=32768.
Updates the predictions CSVs and regenerates accuracy .md + confusion matrices.
"""
import json, os, re, sys, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

# ── paths ────────────────────────────────────────────────────────────────────
SCRIPTS_DIR   = Path(__file__).parent
FINAL_METRICS = SCRIPTS_DIR / "../Final Metrics"
NB_PATH       = SCRIPTS_DIR / "04_round1_1a_attribute_accuracy.ipynb"

# ── API key ──────────────────────────────────────────────────────────────────
def _get_openai_key():
    key = os.getenv("OPENAI_API_KEY")
    if key:
        return key
    try:
        nb = json.loads(NB_PATH.read_text())
        text = "\n".join(
            "\n".join(c.get("source", []))
            for c in nb.get("cells", []) if c.get("cell_type") == "markdown"
        )
        m = re.search(r"GPT\s*\(general\s*\)\s*:\s*(sk-[A-Za-z0-9_\-]+)", text)
        if m: return m.group(1).strip()
    except Exception:
        pass
    raise RuntimeError("No OpenAI API key found")

OPENAI_API_KEY = _get_openai_key()
client = OpenAI(api_key=OPENAI_API_KEY)
MODEL             = os.getenv("GPT_STRONG_MODEL", "gpt-5")
MAX_TOKENS        = int(os.getenv("GPT_RETRY_MAX_TOKENS", "8192"))
REASONING_EFFORT  = os.getenv("GPT_REASONING_EFFORT", "minimal")

print(f"Retrying gpt_strong ({MODEL}) with max_tokens={MAX_TOKENS}, reasoning_effort={REASONING_EFFORT}")

# ── label sets ───────────────────────────────────────────────────────────────
MACRO_FUNCTION_LABELS = [
    "Assumed-Agreement Rhetorical Stance",
    "Neutral Declarative",
    "Information-Seeking Verification",
    "Affective Confirmation-Seeking Question",
    "Emphatic / Discourse-Marking",
    "Null Form Retaining Particle-Like Pragmatic Meaning",
    "Negative Rhetorical Challenge / Evaluation",
]
PARTICLE_LABELS = ["ke", "kan", "neutral"]

# ── helpers ──────────────────────────────────────────────────────────────────
def _call(system, prompt, retries=6):
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": prompt},
                ],
                max_completion_tokens=MAX_TOKENS,
                reasoning_effort=REASONING_EFFORT,
            )
            raw = (resp.choices[0].message.content or "").strip()
            if raw:
                return raw
        except Exception as e:
            err = str(e)
            if "429" in err or "rate_limit" in err.lower():
                wait = 65 + attempt * 10
                print(f"  rate-limit, sleeping {wait}s …", flush=True)
                time.sleep(wait)
            elif attempt == retries - 1:
                return f"ERROR: {e}"
            else:
                time.sleep(2 ** attempt)
    return "ERROR: Empty after retries"

def _extract_macro(raw):
    text = raw.lower().strip()
    for label in MACRO_FUNCTION_LABELS:
        if label.lower() == text:
            return label
    for label in sorted(MACRO_FUNCTION_LABELS, key=len, reverse=True):
        if label.lower() in text:
            return label
    return raw.strip()

def _extract_particle(raw):
    text = raw.lower().strip()
    if text in {"ke", "kan", "neutral"}:
        return text
    for p in ["neutral", "kan", "ke"]:
        if re.search(rf"\b{p}\b", text):
            return p
    return raw.strip()

# ── per-test retry ────────────────────────────────────────────────────────────
def retry_test(tag, gt_col, col_suffix, label_set, prompt_col, system_text,
               extractor_fn, build_prompt_fn):
    csv_path = FINAL_METRICS / f"round2_{tag}_predictions.csv"
    df = pd.read_csv(csv_path)
    col = f"gpt_strong_{col_suffix}"
    if col not in df.columns:
        print(f"  [{tag}] column {col} not found, skipping")
        return

    error_mask = df[col].astype(str).str.startswith("ERROR")
    error_rows = df.index[error_mask].tolist()
    print(f"\n[{tag}] {len(error_rows)} error rows to retry …")
    if not error_rows:
        return

    def task(idx):
        row = df.loc[idx]
        prompt = build_prompt_fn(row)
        raw    = _call(system_text, prompt)
        pred   = extractor_fn(raw)
        return idx, pred

    with ThreadPoolExecutor(max_workers=10) as ex:
        futures = {ex.submit(task, idx): idx for idx in error_rows}
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc=f"gpt_strong_{tag}", file=sys.stdout):
            idx, pred = fut.result()
            df.at[idx, col] = pred

    df.to_csv(csv_path, index=False)
    print(f"  Saved → {csv_path.resolve()}")

    # accuracy
    gt   = df[gt_col].astype(str).str.strip()
    pred = df[col].astype(str).str.strip()
    acc  = (gt == pred).mean()
    errs = pred.str.startswith("ERROR").sum()
    print(f"  gpt_strong {tag}: accuracy={acc:.4f}  errors={errs}")

    # confusion matrix
    mask = gt.isin(label_set)
    cm   = pd.crosstab(gt[mask], pred[mask], rownames=["Actual"], colnames=["Predicted"])
    cm   = cm.reindex(index=label_set, columns=label_set, fill_value=0)
    cm.to_csv(FINAL_METRICS / f"round2_{tag}_cm_gpt_strong.csv")
    print(f"  Confusion matrix updated.")

# ── prompt builders (inline, no round2 import needed) ────────────────────────
MACRO_FUNCTION_DEFINITIONS = {
    "Assumed-Agreement Rhetorical Stance":
        "Speaker presents proposition as already obvious/shared knowledge; listener is expected to align rather than genuinely answer.",
    "Neutral Declarative":
        "Plain informational statements with minimal discourse pressure or stance marking.",
    "Information-Seeking Verification":
        "Genuine request for verification or clarification; speaker leaves room for disagreement.",
    "Affective Confirmation-Seeking Question":
        "Speaker seeks confirmation while simultaneously expressing affect (surprise, irritation, humour, excitement, disbelief, etc.)",
    "Emphatic / Discourse-Marking":
        "Particle functions less as a literal confirmation marker and more as a discourse-management or emphasis device.",
    "Null Form Retaining Particle-Like Pragmatic Meaning":
        "Pragmatic meaning associated with particles remains inferable even after overt particle removal.",
    "Negative Rhetorical Challenge / Evaluation":
        "Speaker uses rhetorical questioning to criticise, challenge, mock, or negatively evaluate a proposition rather than genuinely seek information.",
}
ATTRIBUTE_DESCRIPTIONS = {
    "Epistemic_Stance": {
        "Certain":         "The speaker treats the statement as already true or established — no hedging, no doubt, full confidence.",
        "Uncertain":       "The speaker sounds unsure, is making a guess, estimating, or checking whether something is the case.",
        "Neutral/Unclear": "No detectable certainty signal; the sentence does not lean toward confident assertion or uncertainty.",
        "Neutral / NA":    "No detectable certainty signal; epistemic stance is not relevant here.",
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
        "Declarative/Statement":    "The sentence makes an assertion or conveys information — not structured as a question.",
        "Rhetorical Interrogative": "Phrased as a question but does not expect a genuine answer; used to make a point or emphasise something.",
        "Yes/No Interrogative":     "A genuine question inviting the listener to confirm or deny; the speaker does not already know the answer.",
    },
}

MACRO_LABEL_LIST_TEXT = "\n\n".join(
    f"{l}: {MACRO_FUNCTION_DEFINITIONS[l]}" for l in MACRO_FUNCTION_LABELS
)
SYSTEM_1B = (
    "You are a linguist specialising in colloquial Malay discourse pragmatics. "
    "You must output exactly one label from the provided list and nothing else."
)
SYSTEM_1C = SYSTEM_1B
SYSTEM_2B = (
    "You are a linguist specialising in colloquial Malay discourse particles. "
    'You must output exactly one word — either "ke" or "kan" or "neutral" — and nothing else.'
)

def _build_attr_block(row):
    lines = []
    for attr, key in [
        ("Epistemic_Stance", "Epistemic_Stance"),
        ("Particle_Position", "Particle_Position"),
        ("Listener_Agreement", "Listener_Agreement"),
        ("Emotion", "Emotion"),
        ("Question_Type", "Question_Type"),
    ]:
        val  = str(row.get(attr, "")).strip()
        desc = ATTRIBUTE_DESCRIPTIONS[key].get(val, "")
        lines.append(f"{val}: {desc}" if desc else val)
    return "\n".join(lines)

def prompt_1b(row):
    return (
        "You are a linguist specialising in colloquial Malay discourse pragmatics. "
        "Your task is to read the Malay sentence below and identify the primary communicative role "
        "the utterance plays in interaction, beyond its literal propositional content.\n\n"
        "Referring to the following seven labels and their definitions to make your decision:\n\n"
        + MACRO_LABEL_LIST_TEXT
        + f'\n\nSpeaker: "{row["Text"]}"\n\n'
        "Considering what the speaker is communicatively doing with this utterance, their stance, "
        "their orientation toward the listener, and the function the sentence serves in interaction, "
        "which of the seven labels best captures its primary discourse role?\n"
        "The most likely, single label is:"
    )

def prompt_1c(row):
    attr_block = _build_attr_block(row)
    return (
        "You are a linguist specialising in colloquial Malay discourse pragmatics. "
        "Your task is to read the Malay sentence below and identify the primary communicative role "
        "the utterance plays in interaction, beyond its literal propositional content.\n\n"
        "You are provided with the following human-annotated attribute labels for this sentence "
        f"as additional context:\n{attr_block}\n\n"
        "Use these attributes to inform your decision, but base your final label on the overall "
        "communicative function of the utterance.\n\n"
        "Referring to the following seven labels and their definitions to make your decision:\n\n"
        + MACRO_LABEL_LIST_TEXT
        + f'\n\nSpeaker: "{row["Text"]}"\n\n'
        "Considering what the speaker is communicatively doing with this utterance, their stance, "
        "their orientation toward the listener, and the function the sentence serves in interaction, "
        "which of the seven labels best captures its primary discourse role?\n"
        "The most likely, single label is:"
    )

def prompt_2b(row):
    macro = str(row.get("Macro_Function", "")).strip()
    defn  = MACRO_FUNCTION_DEFINITIONS.get(macro, "")
    block = f"  {macro}: {defn}" if defn else f"  {macro}"
    text_masked = str(row.get("Text_Masked", row.get("Text", ""))).strip()
    return (
        "You are given a Malay sentence in which one discourse particle has been replaced with [___].\n"
        "Your task is to predict which particle — \"ke\", \"kan\", or \"neutral\" — belongs in the "
        "[___] slot, such that the sentence is consistent with the primary communicative role the "
        "utterance plays in interaction, beyond its literal propositional content:\n"
        f"{block}\n\n\n"
        "Particle meanings:\n"
        "  ke      : signals genuine uncertainty or invites the listener to confirm something the speaker is unsure about.\n"
        "  kan     : signals assumed shared knowledge and seeks light confirmation of something the speaker already believes (\"right?\").\n"
        "  neutral : indicates no particle.\n\n"
        f'Speaker:\n  "{text_masked}"\n\n'
        "Using the sentence context and the macro-function above, which single particle — "
        '"ke" or "kan" or "neutral" — best fills [___]?\n\n'
        "Return exactly one word from this set and nothing else: ke, kan, neutral"
    )

# ── run retries ───────────────────────────────────────────────────────────────
retry_test("1b", "Macro_Function", "macro_1b",   MACRO_FUNCTION_LABELS,
           "Text", SYSTEM_1B, _extract_macro, prompt_1b)
retry_test("1c", "Macro_Function", "macro_1c",   MACRO_FUNCTION_LABELS,
           "Text", SYSTEM_1C, _extract_macro, prompt_1c)
retry_test("2b", "GT_Particle",   "particle_2b", PARTICLE_LABELS,
           "Text_Masked", SYSTEM_2B, _extract_particle, prompt_2b)

print("\nAll gpt_strong retries complete.")
