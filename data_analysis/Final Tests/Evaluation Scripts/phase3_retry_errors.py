"""Retry ERROR cells in the three Phase 3 prediction CSVs.

For each test:
  - load predictions CSV
  - find columns whose values start with "ERROR" (skip "ERROR_FATAL")
  - rebuild prompt for each (row, model), re-issue with extra retries
  - patch CSV in place, regenerate accuracy summary
"""

import re
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from phase3_common import (
    build_model_runs, load_gold_with_macro, parse_macro_cot, parse_particle_cot,
    parse_particle_plain, predict_with_retry, save_io_logs, worker_cfg,
    write_accuracy_report,
)
from round1_2d_particle_generation_both import build_prompt as build_2d, SYSTEM_MSG as SYS_2D
from round2_1b_cot import build_prompt as build_1b, SYSTEM_MSG as SYS_1B
from round2_2b_cot import build_prompt as build_2b, SYSTEM_MSG as SYS_2B


VALID_PARTICLE = {"ke", "kan", "neutral"}
VALID_MACRO = {
    "Assumed-Agreement Rhetorical Stance", "Neutral Declarative",
    "Information-Seeking Verification", "Affective Confirmation-Seeking Question",
    "Emphatic / Discourse-Marking",
    "Null Form Retaining Particle-Like Pragmatic Meaning",
    "Negative Rhetorical Challenge / Evaluation",
}

TESTS = {
    "round1_2d": {
        "csv":          "../Final Metrics/round1_2d_predictions.csv",
        "pred_suffix":  "particle_gen_2d",
        "build_prompt": build_2d,
        "system_msg":   SYS_2D,
        "parse_fn":     parse_particle_plain,
        "max_tokens":   16,
        "gt_col":       "GT_Particle",
        "test_name":    "Test_2d",
        "save_reason":  False,
        "valid_set":    VALID_PARTICLE,
    },
    "round2_1b_cot": {
        "csv":          "../Final Metrics/round2_1b_cot_predictions.csv",
        "pred_suffix":  "macro_1b_cot",
        "build_prompt": build_1b,
        "system_msg":   SYS_1B,
        "parse_fn":     parse_macro_cot,
        "max_tokens":   512,
        "gt_col":       "Macro_Function",
        "test_name":    "Test_1b_CoT",
        "save_reason":  True,
        "valid_set":    None,  # set below
    },
    "round2_2b_cot": {
        "csv":          "../Final Metrics/round2_2b_cot_predictions.csv",
        "pred_suffix":  "particle_2b_cot",
        "build_prompt": build_2b,
        "system_msg":   SYS_2B,
        "parse_fn":     parse_particle_cot,
        "max_tokens":   512,
        "gt_col":       "GT_Particle",
        "test_name":    "Test_2b_CoT",
        "save_reason":  True,
        "valid_set":    None,  # set below
    },
}


def retry_one_test(key, cfg, model_runs, gold_df):
    csv_path = Path(cfg["csv"])
    df = pd.read_csv(csv_path)
    print(f"\n=== {cfg['test_name']}  rows={len(df)} ===")
    io_logs = []
    model_by_name = {m["name"]: m for m in model_runs}

    for col in [c for c in df.columns if c.endswith(f"_{cfg['pred_suffix']}")]:
        model_name = col[: -len(f"_{cfg['pred_suffix']}")]
        if model_name not in model_by_name:
            continue
        m = model_by_name[model_name]
        s = df[col].astype(str)
        is_err = s.str.startswith("ERROR") & ~s.str.startswith("ERROR_FATAL")
        is_parsefail = ~s.isin(cfg["valid_set"]) & ~s.str.startswith("ERROR")
        mask = is_err | is_parsefail
        idxs = df[mask].index.tolist()
        if not idxs:
            continue
        print(f"  {model_name}: retrying {len(idxs)} rows", flush=True)
        for j, i in enumerate(idxs, 1):
            row = gold_df.iloc[i]
            prompt = cfg["build_prompt"](row)
            label, raw = predict_with_retry(
                worker_cfg(m), prompt, cfg["system_msg"], cfg["parse_fn"],
                max_tokens=cfg["max_tokens"], max_output_retries=6,
                log_sink=io_logs,
                log_meta={"phase": f"retry_{key}", "row_idx": int(i),
                          "model": model_name},
            )
            df.at[i, col] = label
            if cfg["save_reason"]:
                reason_col = f"{model_name}_reasoning"
                if reason_col in df.columns:
                    df.at[i, reason_col] = raw
            if j % 10 == 0 or j == len(idxs):
                print(f"    {model_name} {j}/{len(idxs)}  last={label[:60]}", flush=True)

    df.to_csv(csv_path, index=False)
    print(f"  patched CSV → {csv_path}")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    save_io_logs(io_logs, f"{key}_retry", run_id)
    write_accuracy_report(df, model_runs, gt_col=cfg["gt_col"],
                          pred_col_suffix=cfg["pred_suffix"],
                          test_name=cfg["test_name"],
                          group_col="Macro_Function")


def main():
    TESTS["round2_1b_cot"]["valid_set"] = VALID_MACRO
    TESTS["round2_2b_cot"]["valid_set"] = VALID_PARTICLE
    gold = load_gold_with_macro()
    model_runs = build_model_runs()
    for key, cfg in TESTS.items():
        retry_one_test(key, cfg, model_runs, gold)


if __name__ == "__main__":
    main()
