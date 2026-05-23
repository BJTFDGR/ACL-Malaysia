"""Test 2b-CoT — CoT + Function-Constrained Particle Generation.

Given a masked Malay sentence + the macro-function label, ask each LLM to
reason step-by-step, then output one of "ke", "kan", or "neutral".

Env:
  SAMPLE_N=0           full rows
  SAMPLE_N=5           first 5 rows (smoke)
  STOP_AFTER_SMOKE=1   exit after 1-row smoke test
"""

import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from phase3_common import (
    MACRO_FUNCTION_DEFINITIONS, build_model_runs, load_gold_with_macro,
    parse_particle_cot, run_parallel, save_io_logs, worker_cfg,
    write_accuracy_report,
)


SYSTEM_MSG = (
    "You are a linguist specialising in colloquial Malay discourse particles. "
    'Reason step-by-step, then output exactly one particle — "ke", "kan", or "neutral".'
)


def build_prompt(row) -> str:
    macro = row["Macro_Function"]
    macro_def = MACRO_FUNCTION_DEFINITIONS[macro]
    return (
        "You are given a Malay sentence in which one discourse particle has been replaced with [___].\n\n"
        "The macro-pragmatic function of this utterance is:\n"
        f"  {macro}: {macro_def}\n\n"
        "Particle meanings:\n"
        "  ke      : signals genuine uncertainty or invites the listener to confirm something the speaker is unsure about.\n"
        '  kan     : signals assumed shared knowledge and seeks light confirmation of something the speaker already believes ("right?").\n'
        "  neutral : indicates no particle.\n\n"
        f'Speaker:\n  "{row["Text_Masked"]}"\n\n'
        "Think step-by-step:\n"
        "  1. What is the speaker communicatively doing, given the macro-function above?\n"
        "  2. Does the masked slot call for genuine uncertainty (ke), assumed shared knowledge (kan), or no particle (neutral)?\n"
        "  3. Which choice best fits both the sentence flow and the macro-function?\n\n"
        "Format your response EXACTLY as follows, with no extra text before or after:\n\n"
        "Reasoning: <2–4 sentences of step-by-step analysis>\n"
        "Final answer: <ke | kan | neutral>"
    )


def main():
    RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    SAMPLE_N = int(os.getenv("SAMPLE_N", "5"))

    df = load_gold_with_macro()
    eval_df = df.head(SAMPLE_N).copy() if SAMPLE_N else df.copy()
    print(f"Test 2b-CoT — evaluating {len(eval_df)} rows")

    model_runs = build_model_runs()
    print("Models:", [m["name"] for m in model_runs])

    io_logs = []

    sample = eval_df.iloc[0]
    smoke_prompt = build_prompt(sample)
    print("\n=== SMOKE PROMPT ===\n" + smoke_prompt + "\n====================\n")
    smoke_tasks = [{
        "row_idx": 0, "model_name": m["name"], "run_cfg": worker_cfg(m),
        "prompt_text": smoke_prompt, "constraint_text": SYSTEM_MSG,
        "parse_fn": parse_particle_cot, "max_tokens": 512,
        "log_meta": {"phase": "smoke_2b_cot"},
    } for m in model_runs]
    smoke_res = run_parallel(smoke_tasks, desc="smoke_2b_cot")
    fatal = set()
    for r in smoke_res:
        if isinstance(r["prediction"], str) and r["prediction"].startswith("ERROR_FATAL"):
            fatal.add(r["model_name"])
        print(f"  {r['model_name']:20s} → {r['prediction']}")
        io_logs.extend(r["logs"])
    save_io_logs(io_logs, "round2_2b_cot", RUN_ID)

    if os.getenv("STOP_AFTER_SMOKE", "0") == "1":
        return

    tasks = []
    active = []
    for m in model_runs:
        if m["name"] in fatal:
            continue
        active.append(m["name"])
        for i in range(len(eval_df)):
            row = eval_df.iloc[i]
            tasks.append({
                "row_idx": i, "model_name": m["name"], "run_cfg": worker_cfg(m),
                "prompt_text": build_prompt(row), "constraint_text": SYSTEM_MSG,
                "parse_fn": parse_particle_cot, "max_tokens": 512,
                "log_meta": {"phase": "test_2b_cot", "row_idx": i},
            })

    results = run_parallel(tasks, desc="test_2b_cot")
    preds = {name: [None] * len(eval_df) for name in active}
    reasons = {name: [""] * len(eval_df) for name in active}
    for r in results:
        preds[r["model_name"]][r["row_idx"]] = r["prediction"]
        reasons[r["model_name"]][r["row_idx"]] = r["raw_full"]
        io_logs.extend(r["logs"])

    out = eval_df.copy()
    for m in model_runs:
        if m["name"] in preds:
            out[f"{m['name']}_particle_2b_cot"] = preds[m["name"]]
            out[f"{m['name']}_reasoning"] = reasons[m["name"]]

    out_path = Path("../Final Metrics/round2_2b_cot_predictions.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"\nSaved → {out_path.resolve()}")

    save_io_logs(io_logs, "round2_2b_cot", RUN_ID)
    write_accuracy_report(out, model_runs, gt_col="GT_Particle",
                          pred_col_suffix="particle_2b_cot", test_name="Test_2b_CoT",
                          group_col="Macro_Function")


if __name__ == "__main__":
    main()
