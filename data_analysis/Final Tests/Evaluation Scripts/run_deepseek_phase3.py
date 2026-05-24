"""Run deepseek_strong and deepseek_weak on all three Phase-3 tests.

Appends columns to existing predictions CSVs in-place; does not touch other
models' data.
"""
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from phase3_common import (build_model_runs, load_gold_with_macro,
                           parse_macro_cot, parse_particle_cot,
                           parse_particle_plain, run_parallel,
                           save_io_logs, worker_cfg, write_accuracy_report)
from round1_2d_particle_generation_both import build_prompt as build_2d, SYSTEM_MSG as SYS_2D
from round2_1b_cot import build_prompt as build_1b, SYSTEM_MSG as SYS_1B
from round2_2b_cot import build_prompt as build_2b, SYSTEM_MSG as SYS_2B


TESTS = [
    dict(key="round1_2d",     csv="../Final Metrics/round1_2d_predictions.csv",
         pred_suffix="particle_gen_2d",  build_prompt=build_2d, system_msg=SYS_2D,
         parse_fn=parse_particle_plain,  max_tokens=16,
         gt_col="GT_Particle", test_name="Test_2d", save_reason=False),
    dict(key="round2_1b_cot", csv="../Final Metrics/round2_1b_cot_predictions.csv",
         pred_suffix="macro_1b_cot",     build_prompt=build_1b, system_msg=SYS_1B,
         parse_fn=parse_macro_cot,       max_tokens=2048,
         gt_col="Macro_Function", test_name="Test_1b_CoT", save_reason=True),
    dict(key="round2_2b_cot", csv="../Final Metrics/round2_2b_cot_predictions.csv",
         pred_suffix="particle_2b_cot",  build_prompt=build_2b, system_msg=SYS_2B,
         parse_fn=parse_particle_cot,    max_tokens=2048,
         gt_col="GT_Particle", test_name="Test_2b_CoT", save_reason=True),
]


def main():
    gold = load_gold_with_macro()
    model_runs = build_model_runs()
    deepseek = [m for m in model_runs if m["name"].startswith("deepseek")]
    if not deepseek:
        raise RuntimeError("DeepSeek not configured")
    print("DeepSeek models:", [m["name"] for m in deepseek])

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    io_logs = []

    for cfg in TESTS:
        print(f"\n=== {cfg['test_name']} ===")
        df = pd.read_csv(cfg["csv"])
        tasks = []
        for m in deepseek:
            for i in range(len(df)):
                tasks.append({
                    "row_idx": i, "model_name": m["name"],
                    "run_cfg": worker_cfg(m),
                    "prompt_text": cfg["build_prompt"](gold.iloc[i]),
                    "constraint_text": cfg["system_msg"],
                    "parse_fn": cfg["parse_fn"],
                    "max_tokens": cfg["max_tokens"],
                    "log_meta": {"phase": f"deepseek_{cfg['key']}", "row_idx": i},
                })
        results = run_parallel(tasks, desc=f"deepseek_{cfg['key']}")
        preds = {m["name"]: [None] * len(df) for m in deepseek}
        reasons = {m["name"]: [""] * len(df) for m in deepseek}
        for r in results:
            preds[r["model_name"]][r["row_idx"]] = r["prediction"]
            reasons[r["model_name"]][r["row_idx"]] = r["raw_full"]
            io_logs.extend(r["logs"])

        for m in deepseek:
            df[f"{m['name']}_{cfg['pred_suffix']}"] = preds[m["name"]]
            if cfg["save_reason"]:
                df[f"{m['name']}_reasoning"] = reasons[m["name"]]
        df.to_csv(cfg["csv"], index=False)
        print(f"  patched CSV → {cfg['csv']}")
        write_accuracy_report(df, model_runs, gt_col=cfg["gt_col"],
                              pred_col_suffix=cfg["pred_suffix"],
                              test_name=cfg["test_name"],
                              group_col="Macro_Function")
    save_io_logs(io_logs, "phase3_deepseek", run_id)


if __name__ == "__main__":
    main()
