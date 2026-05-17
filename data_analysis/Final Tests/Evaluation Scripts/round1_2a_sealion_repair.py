"""
Repair sealion_strong in round1_2a_predictions.csv:
  1. Re-extract labels from stored raw output (strip </think> + non-ASCII junk)
  2. Retry the 1 ERROR row (429) via API
  3. Save updated CSV + recalculate accuracy summary
"""
import os, re, sys, time, random
from pathlib import Path

import pandas as pd
import requests

PRED_CSV = Path("../Final Metrics/round1_2a_predictions.csv")
SUMM_MD  = Path("../Final Metrics/round1_2a_accuracy_summary.md")
COL      = "sealion_strong_particle_gen"
VALID    = {"ke", "kan", "neutral"}

# ── Load ─────────────────────────────────────────────────────────────────────
df = pd.read_csv(PRED_CSV)
print(f"Loaded {len(df)} rows")
print("Before fix:", df[COL].value_counts(dropna=False).head(5).to_string())

# ── Step 1: Re-extract from stored raw output ─────────────────────────────────
def _clean(raw: str) -> str:
    text = str(raw or "").strip()
    text = re.sub(r"</?think>", " ", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)   # strip all non-ASCII (ċ etc.)
    text = text.strip().lower()
    if text in VALID:
        return text
    for p in ["neutral", "kan", "ke"]:
        if re.search(rf"\b{p}\b", text):
            return p
    return raw  # keep original if still unparseable

fixable = ~df[COL].isin(VALID) & ~df[COL].astype(str).str.startswith("ERROR")
df.loc[fixable, COL] = df.loc[fixable, COL].apply(_clean)
print(f"\nRe-extracted {fixable.sum()} rows from stored raw output")

# ── Step 2: Retry ERROR rows via API ─────────────────────────────────────────
error_rows = df[df[COL].astype(str).str.startswith("ERROR")]
print(f"ERROR rows to retry: {len(error_rows)}")

if len(error_rows) > 0:
    from openai import OpenAI
    from round1_2a_particle_generation import (
        build_particle_gen_prompt, PARTICLE_GEN_STRICT_TAIL,
        PARTICLE_GEN_SYSTEM_CONSTRAINT, df_natural,
        SEA_LION_API_KEY, SEA_LION_BASE_URL, SEA_LION_STRONG_MODEL,
    )

    client = OpenAI(api_key=SEA_LION_API_KEY, base_url=SEA_LION_BASE_URL)

    for idx in error_rows.index:
        row = df_natural.iloc[idx]
        prompt_text = build_particle_gen_prompt(row["Text_Masked"], row) + PARTICLE_GEN_STRICT_TAIL
        constraint = PARTICLE_GEN_SYSTEM_CONSTRAINT

        print(f"\nRetrying row {idx} (GT={row['GT_Particle']})...")
        for attempt in range(8):
            try:
                resp = client.chat.completions.create(
                    model=SEA_LION_STRONG_MODEL,
                    messages=[
                        {"role": "system", "content": constraint},
                        {"role": "user", "content": prompt_text},
                    ],
                    max_completion_tokens=64,
                    temperature=0,
                )
                raw = (resp.choices[0].message.content or "").strip()
                label = _clean(raw)
                if label in VALID:
                    df.at[idx, COL] = label
                    print(f"  → {label}")
                    break
                else:
                    print(f"  attempt {attempt+1}: unparseable: {raw[:60]!r}")
                    time.sleep(5)
            except Exception as e:
                err = str(e)
                if "429" in err or "rate" in err.lower():
                    wait = 70 + random.uniform(0, 15)
                    print(f"  rate-limit, backoff {wait:.0f}s")
                    time.sleep(wait)
                else:
                    print(f"  error: {err}")
                    time.sleep(10)
        else:
            print(f"  FAILED after all retries, leaving as ERROR")

# ── Step 3: Save updated CSV ─────────────────────────────────────────────────
df.to_csv(PRED_CSV, index=False)
print(f"\nSaved updated predictions → {PRED_CSV.resolve()}")
print("After fix:", df[COL].value_counts().to_string())

# ── Step 4: Recalculate accuracy ──────────────────────────────────────────────
try:
    from round1_2a_particle_generation import MODEL_RUNS, MODEL_VERSION_MAP
except Exception as e:
    print(f"Warning: could not import MODEL_RUNS: {e}")
    MODEL_RUNS = None

if MODEL_RUNS:
    gt = df["GT_Particle"]
    rows = []
    for m in MODEL_RUNS:
        col = f"{m['name']}_particle_gen"
        if col not in df.columns:
            continue
        pred = df[col]
        acc = (gt == pred).mean()
        errors = pred.astype(str).str.startswith("ERROR").sum()
        rows.append({
            "Model": MODEL_VERSION_MAP.get(m["name"].upper(), m["name"].upper()),
            "Accuracy": round(float(acc), 4),
            "Errors": int(errors),
        })
    acc_df = pd.DataFrame(rows).sort_values(["Accuracy", "Model"], ascending=[False, True])
    print("\n" + "=" * 55)
    print("  Test 2a — Particle Generation Accuracy (updated)")
    print("=" * 55)
    print(acc_df.to_string(index=False))

    header = "| Model | Accuracy | Errors |"
    sep    = "|---|---|---|"
    lines  = [header, sep]
    for _, r in acc_df.iterrows():
        lines.append(f"| {r['Model']} | {r['Accuracy']:.4f} | {r['Errors']} |")
    SUMM_MD.write_text(
        "# Test 2a — Particle Generation Accuracy\n\n" + "\n".join(lines) + "\n",
        encoding="utf-8"
    )
    print(f"Saved updated summary → {SUMM_MD.resolve()}")
