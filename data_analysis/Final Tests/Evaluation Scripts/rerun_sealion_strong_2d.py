"""Re-run sealion_strong on Test 2d with bumped max_tokens.

Fixes the issue where the Llama-SEA-LION-v3.5-70B-R reasoning model was being
cut off at 1024 tokens before producing a final answer, then mis-parsed as
"neutral" due to the prompt's particle-definition block.
"""
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from phase3_common import (build_model_runs, load_gold_with_macro,
                           parse_particle_plain, predict_with_retry,
                           save_io_logs, worker_cfg, write_accuracy_report)
from round1_2d_particle_generation_both import build_prompt, SYSTEM_MSG


def main():
    csv_path = Path("../Final Metrics/round1_2d_predictions.csv")
    df = pd.read_csv(csv_path)
    gold = load_gold_with_macro()
    assert len(df) == len(gold)

    model_runs = build_model_runs()
    ss = next((m for m in model_runs if m["name"] == "sealion_strong"), None)
    if ss is None:
        raise RuntimeError("sealion_strong not configured")
    print(f"Re-running {len(df)} rows on {ss['model']} (max_tokens={ss['max_tokens']})")

    io_logs = []
    col = "sealion_strong_particle_gen_2d"
    for i in range(len(df)):
        row = gold.iloc[i]
        prompt = build_prompt(row)
        label, raw = predict_with_retry(
            worker_cfg(ss), prompt, SYSTEM_MSG, parse_particle_plain,
            max_tokens=16384, max_output_retries=4,
            log_sink=io_logs,
            log_meta={"phase": "rerun_sealion_strong_2d", "row_idx": int(i)},
        )
        df.at[i, col] = label
        if (i + 1) % 10 == 0 or i + 1 == len(df):
            print(f"  {i+1}/{len(df)}  last={label}", flush=True)

    df.to_csv(csv_path, index=False)
    print(f"\nPatched CSV → {csv_path}")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    save_io_logs(io_logs, "round1_2d_sealion_rerun", run_id)
    write_accuracy_report(df, model_runs, gt_col="GT_Particle",
                          pred_col_suffix="particle_gen_2d",
                          test_name="Test_2d", group_col="Macro_Function")


if __name__ == "__main__":
    main()
