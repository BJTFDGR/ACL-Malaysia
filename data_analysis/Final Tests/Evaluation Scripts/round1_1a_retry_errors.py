"""
round1_1a_retry_errors.py
─────────────────────────
Re-runs only the cells in round1_1a_predictions.csv that contain an ERROR
value for the four rate-limited models:
  - claude_strong  (claude-opus-4-7,            ≤50 req/min)
  - claude_weak    (claude-haiku-4-5,            ≤50 req/min)
  - sealion_strong (Llama-SEA-LION-v3.5-70B-R,  ≤10 req/min)
  - sealion_weak   (Gemma-SEA-LION-v4-27B-IT,   ≤10 req/min)

Retries both EN (no suffix) and MS (_ms suffix) columns.
After patching, re-writes the accuracy summary markdown (with error counts).

Usage:
  cd "data_analysis/Final Tests/Evaluation Scripts"
  source /home/xitongzhang/Maylie/.venv/bin/activate
  CLAUDE_WORKERS=20 SEALION_WORKERS=8 python3 round1_1a_retry_errors.py 2>&1 | tee "../Final Metrics/round1_1a_retry_errors.log"
"""

import json
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import anthropic
import pandas as pd
from openai import OpenAI
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────
PREDICTIONS_CSV = Path("../Final Metrics/round1_1a_predictions.csv")
ACCURACY_MD     = Path("../Final Metrics/round1_1a_accuracy_summary.md")
ACCURACY_MD_ERRORS = Path("../Final Metrics/round1_1a_accuracy_summary_with_errors.md")
RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
LOG_JSON = Path(f"../Final Metrics/round1_1a_retry_errors_{RUN_ID}.json")

PARALLEL_WORKERS = int(os.getenv("PARALLEL_WORKERS", "10"))

# ── Extract API keys from notebook (same logic as main script) ────────────────
def _extract_keys_from_notebook(nb_path: Path):
    keys = {"anthropic": None, "sealion": None}
    try:
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        text = "\n".join(
            "\n".join(cell.get("source", []))
            for cell in nb.get("cells", [])
            if cell.get("cell_type") == "markdown"
        )
        m = re.search(r"Claude\s*:\s*(sk-ant-[A-Za-z0-9_\-]+)", text)
        if m:
            keys["anthropic"] = m.group(1).strip()
        m = re.search(r"SEA-LION\s*:\s*(sk-[A-Za-z0-9_\-]+)", text)
        if m:
            keys["sealion"] = m.group(1).strip()
    except Exception:
        pass
    return keys

notebook_path = Path("04_round1_1a_attribute_accuracy.ipynb")
fallback = _extract_keys_from_notebook(notebook_path)

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY") or fallback["anthropic"]
SEA_LION_API_KEY  = os.getenv("SEA_LION_API_KEY")  or fallback["sealion"]
SEA_LION_BASE_URL = os.getenv("SEA_LION_BASE_URL", "https://api.sea-lion.ai/v1")

if not ANTHROPIC_API_KEY:
    raise ValueError("No Anthropic API key found.")
if not SEA_LION_API_KEY:
    raise ValueError("No SEA-LION API key found.")
CLAUDE_STRONG_MODEL = os.getenv("CLAUDE_STRONG_MODEL", "claude-sonnet-4-6")
CLAUDE_WEAK_MODEL   = os.getenv("CLAUDE_WEAK_MODEL",   "claude-haiku-4-5")
SEA_LION_STRONG_MODEL = os.getenv("SEA_LION_STRONG_MODEL", "aisingapore/Llama-SEA-LION-v3.5-70B-R")
SEA_LION_WEAK_MODEL   = os.getenv("SEA_LION_WEAK_MODEL",   "aisingapore/Gemma-SEA-LION-v4-27B-IT")

# Only the 4 rate-limited models are retried.
RETRY_MODELS = [
    {
        "name": "claude_strong",
        "provider": "anthropic",
        "api_key": ANTHROPIC_API_KEY,
        "model": CLAUDE_STRONG_MODEL,
        "workers": int(os.getenv("CLAUDE_WORKERS", "20")),
    },
    {
        "name": "claude_weak",
        "provider": "anthropic",
        "api_key": ANTHROPIC_API_KEY,
        "model": CLAUDE_WEAK_MODEL,
        "workers": int(os.getenv("CLAUDE_WORKERS", "20")),
    },
    {
        "name": "sealion_strong",
        "provider": "openai",
        "api_key": SEA_LION_API_KEY,
        "base_url": SEA_LION_BASE_URL,
        "model": SEA_LION_STRONG_MODEL,
        "max_tokens": 512,
        "workers": int(os.getenv("SEALION_WORKERS", "8")),
    },
    {
        "name": "sealion_weak",
        "provider": "openai",
        "api_key": SEA_LION_API_KEY,
        "base_url": SEA_LION_BASE_URL,
        "model": SEA_LION_WEAK_MODEL,
        "workers": int(os.getenv("SEALION_WORKERS", "8")),
    },
]

MODEL_VERSION_MAP = {m["name"]: m["model"] for m in RETRY_MODELS}

# ── Attribute definitions ─────────────────────────────────────────────────────
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

# Malay prompt templates (from main script)
MS_PROMPTS = {
    "Epistemic_Stance": """\
Anda adalah ahli bahasa yang pakar dalam bahasa Melayu kolokial. Tugas anda ialah membaca ayat Melayu di bawah dan menentukan tahap keyakinan penutur terhadap maklumat yang disampaikan.
Rujuk tiga label berikut dan definisinya untuk membuat keputusan anda:
Certain: Penutur menganggap pernyataan itu sudah benar atau sudah ditetapkan. Tiada keraguan, dan tiada semakan. Penutur menegaskan maklumat dengan penuh keyakinan.
Uncertain: Penutur kedengaran tidak pasti, membuat andaian, menganggarkan, atau menyemak sama ada sesuatu itu benar. Perkataan seperti "agaknya," "kot," atau partikel soalan yang mencari pengesahan adalah isyarat biasa.
Neutral/NA: Ayat itu tidak membawa sebarang isyarat keyakinan yang boleh dikesan dalam mana-mana arah. Ini berlaku untuk penerangan neutral, arahan, atau ayat di mana keyakinan tidak relevan.
Penutur: "{TEXT}"
Dengan mengambil kira tiga label "Certain, Uncertain, Neutral/NA", label tunggal yang paling mungkin untuk ujaran penutur ialah:""",

    "Particle_Position": """\
Anda adalah ahli bahasa yang pakar dalam bahasa Melayu kolokial. Tugas anda ialah membaca ayat Melayu di bawah dan menentukan di mana partikel wacana muncul dalam ayat tersebut.
Rujuk tiga label berikut dan definisinya untuk membuat keputusan anda:
Front: Partikel muncul di bahagian paling awal ayat, sebelum mana-mana perkataan kandungan lain.
Middle/End: Partikel muncul di mana-mana selain bahagian hadapan — di tengah ayat, sebelum perkataan terakhir, atau di akhir ayat.
N/A: Tiada partikel wacana hadir dalam ayat (contohnya, ruang partikel ditunjukkan sebagai "[___]" atau ayat itu memang tidak mengandungi partikel).
Penutur: "{TEXT}"
Dengan mengambil kira tiga label "Front, Middle/End, N/A", label tunggal yang paling mungkin untuk ujaran penutur ialah:""",

    "Listener_Agreement": """\
Anda adalah ahli bahasa yang pakar dalam bahasa Melayu kolokial. Tugas anda ialah membaca ayat Melayu di bawah dan menentukan bagaimana penutur mengorientasikan diri terhadap pendengar dari segi pengetahuan bersama atau persetujuan.
Rujuk tiga label berikut dan definisinya untuk membuat keputusan anda:
Assumed Agreement: Penutur menganggap maklumat itu sudah dikongsi atau sudah jelas kepada pendengar. Ayat itu dibentangkan sebagai asas bersama — nada yang mendasarinya ialah "anda sudah tahu ini" atau "sudah tentu ini benar." Tiada permintaan pengesahan secara eksplisit.
Confirmation Seeking: Penutur secara aktif menyemak sama ada pendengar bersetuju, mengetahui, atau boleh mengesahkan maklumat tersebut. Ayat itu menjemput atau meminta pengesahan pendengar sebelum penutur dapat meneruskan dengan keyakinan.
Neutral/Unclear: Ayat itu tidak menunjukkan sebarang orientasi yang jelas terhadap persetujuan pendengar. Ini berlaku untuk pernyataan biasa, arahan, atau kes di mana pendirian interpersonal terhadap persetujuan adalah kabur atau tidak hadir.
Penutur: "{TEXT}"
Dengan mengambil kira tiga label "Assumed Agreement, Confirmation Seeking, Neutral/Unclear", label tunggal yang paling mungkin untuk ujaran penutur ialah:""",

    "Emotion": """\
Anda adalah ahli bahasa yang pakar dalam bahasa Melayu kolokial. Tugas anda ialah membaca ayat Melayu di bawah dan menentukan nada emosi penutur.
Rujuk tiga label berikut dan definisinya untuk membuat keputusan anda:
Positive: Penutur menyatakan kegembiraan, keseronokan, antusias, kepuasan, humor, kasih sayang, kelegaan, atau sebarang perasaan positif yang jelas. Ini termasuk usikan ringan atau sarkasme jenaka yang bernada mesra.
Negative: Penutur menyatakan kekecewaan, kejengkelan, kekecewaan, kesedihan, kemarahan, kepahitan, atau sebarang perasaan negatif yang jelas. Ini termasuk sarkasme yang bersifat bermusuhan atau pahit.
Neutral/Unclear: Ayat itu tidak membawa sebarang muatan emosi yang boleh dikesan dalam mana-mana arah, atau emosi itu benar-benar samar-samar dan tidak boleh dikelaskan dengan boleh dipercayai sebagai positif atau negatif.
Penutur: "{TEXT}"
Dengan mengambil kira tiga label "Positive, Negative, Neutral/Unclear", label tunggal yang paling mungkin untuk ujaran penutur ialah:""",

    "Question_Type": """\
Anda adalah ahli bahasa yang pakar dalam bahasa Melayu kolokial. Tugas anda ialah membaca ayat Melayu di bawah dan menentukan fungsi ayat utamanya.
Rujuk tiga label berikut dan definisinya untuk membuat keputusan anda:
Declarative/Statement: Ayat itu membuat penegasan atau menyampaikan maklumat. Ia menerangkan situasi, menyatakan fakta, atau meluahkan pandangan. Ia tidak dibentuk sebagai soalan, walaupun ia berakhir dengan partikel.
Rhetorical Interrogative: Ayat itu digubal sebagai soalan tetapi tidak mengharapkan jawapan sebenar daripada pendengar. Ia digunakan untuk membuat penegasan, meluahkan emosi, atau menekankan sesuatu — penutur sudah menunjukkan jawapan melalui soalan itu sendiri.
Yes/No Interrogative: Ayat itu adalah soalan tulen yang menjemput pendengar untuk mengesahkan atau menafikan sesuatu. Penutur tidak mengetahui jawapannya dan sedang mencari respons ya-atau-tidak yang sebenar.
Penutur: "{TEXT}"
Dengan mengambil kira tiga label "Declarative/Statement, Rhetorical Interrogative, Yes/No Interrogative", label tunggal yang paling mungkin untuk ujaran penutur ialah:""",
}

MS_CONSTRAINT = "Anda mesti mengeluarkan tepat satu label daripada set label yang diberi dan tiada teks lain."

# ── English prompt templates ──────────────────────────────────────────────────
EN_PROMPTS = {
    "Epistemic_Stance": """\
You are a linguist specialising in colloquial Malay. Your task is to read the Malay sentence below and decide how certain the speaker sounds about the information they are conveying.
Referring to the following three labels and their definitions to make your decision:
Certain: The speaker treats the statement as already true or established. There is no hedging, no doubt, and no checking. The speaker is asserting the information with full confidence.
Uncertain: The speaker sounds unsure, is making a guess, is estimating, or is checking whether something is the case. Words like "agaknya" (I think/probably), "kot" (maybe), or question particles that probe for confirmation are typical signals.
Neutral/NA: The sentence does not carry any detectable certainty signal in either direction. This applies to neutral descriptions, commands, or sentences where certainty is simply not relevant.
Speaker: "{TEXT}"
Given the three labels "Certain, Uncertain, Neutral/NA", the most likely, single label of the speaker's utterance is:""",

    "Particle_Position": """\
You are a linguist specialising in colloquial Malay. Your task is to read the Malay sentence below and decide where the discourse particle appears in it.
Referring to the following three labels and their definitions to make your decision:
Front: The particle appears at the very start of the sentence, before any other content words.
Middle/End: The particle appears anywhere other than the front — mid-sentence, before the final word, or at the end.
N/A: No discourse particle is present in the sentence (e.g. the particle slot is shown as "[___]" or the sentence simply contains no particle).
Speaker: "{TEXT}"
Given the three labels "Front, Middle/End, N/A", the most likely, single label of the speaker's utterance is:""",

    "Listener_Agreement": """\
You are a linguist specialising in colloquial Malay. Your task is to read the Malay sentence below and decide how the speaker is orienting toward the listener in terms of shared knowledge or agreement.
Referring to the following three labels and their definitions to make your decision:
Assumed Agreement: The speaker treats the information as already shared or obvious to the listener. The sentence is presented as common ground — the underlying tone is "you already know this" or "of course this is true". No explicit confirmation is being requested.
Confirmation Seeking: The speaker is actively checking whether the listener agrees, knows, or can confirm the information. The sentence invites or requests the listener's validation before the speaker can proceed with confidence.
Neutral/Unclear: The sentence does not show any clear orientation toward listener agreement. This applies to plain statements, commands, or cases where the interpersonal stance toward agreement is ambiguous or absent.
Speaker: "{TEXT}"
Given the three labels "Assumed Agreement, Confirmation Seeking, Neutral/Unclear", the most likely, single label of the speaker's utterance is:""",

    "Emotion": """\
You are a linguist specialising in colloquial Malay. Your task is to read the Malay sentence below and decide the emotional tone of the speaker.
Referring to the following three labels and their definitions to make your decision:
Positive: The speaker expresses happiness, excitement, enthusiasm, satisfaction, humour, affection, relief, or any other clearly positive feeling. This includes light-hearted teasing or playful sarcasm that is warm in tone.
Negative: The speaker expresses frustration, annoyance, disappointment, sadness, anger, bitterness, or any other clearly negative feeling. This includes hostile or bitter sarcasm.
Neutral/Unclear: The sentence carries no detectable emotional charge in either direction, or the emotion is genuinely ambiguous and cannot be reliably classified as positive or negative.
Speaker: "{TEXT}"
Given the three labels "Positive, Negative, Neutral/Unclear", the most likely, single label of the speaker's utterance is:""",

    "Question_Type": """\
You are a linguist specialising in colloquial Malay. Your task is to read the Malay sentence below and decide its primary sentence function.
Referring to the following three labels and their definitions to make your decision:
Declarative/Statement: The sentence makes an assertion or conveys information. It describes a situation, states a fact, or expresses a view. It is not structured as a question, even if it ends with a particle.
Rhetorical Interrogative: The sentence is phrased as a question but does not expect a genuine answer from the listener. It is used to make a point, express emotion, or emphasise something — the speaker already implies the answer through the question itself.
Yes/No Interrogative: The sentence is a genuine question that invites the listener to confirm or deny something. The speaker does not already know the answer and is seeking a real yes-or-no response.
Speaker: "{TEXT}"
Given the three labels "Declarative/Statement, Rhetorical Interrogative, Yes/No Interrogative", the most likely, single label of the speaker's utterance is:""",
}

EN_CONSTRAINT = "You must output exactly one label from the provided set and nothing else."

IO_LOGS = []


def _utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def _extract_label(raw, label_set):
    text = str(raw or "").strip()
    for label in sorted(label_set, key=len, reverse=True):
        if label.lower() in text.lower():
            return label
    return text


def _is_fatal_error(error):
    text = str(error).lower()
    fatal_markers = [
        "model_not_found", "does not exist", "unsupported parameter",
        "unsupported value", "invalid_request_error", "api key not valid",
        "permission", "authentication", "insufficient_quota", "quota",
    ]
    return any(m in text for m in fatal_markers)


def _is_transient_error(error):
    text = str(error).lower()
    return any(m in text for m in ["503", "429", "timeout", "unavailable", "resource_exhausted", "deadline"])


def _build_client(run_cfg):
    provider = run_cfg["provider"]
    if provider == "anthropic":
        return anthropic.Anthropic(api_key=run_cfg["api_key"])
    if provider == "openai":
        return OpenAI(api_key=run_cfg["api_key"], base_url=run_cfg.get("base_url"))
    raise ValueError(f"Unknown provider: {provider}")


def call_llm(run_cfg, prompt_text, label_set, constraint_text, retries=3, delay=1.5, log_sink=None):
    sink = IO_LOGS if log_sink is None else log_sink
    for attempt in range(retries):
        try:
            provider = run_cfg["provider"]
            client = _build_client(run_cfg)
            model = run_cfg["model"]

            if provider == "anthropic":
                response = client.messages.create(
                    model=model,
                    system=constraint_text,
                    max_tokens=12,
                    messages=[{"role": "user", "content": prompt_text}],
                )
                raw = response.content[0].text.strip()
            elif provider == "openai":
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": constraint_text},
                        {"role": "user", "content": prompt_text},
                    ],
                    max_completion_tokens=run_cfg.get("max_tokens", 32),
                )
                raw = (response.choices[0].message.content or "").strip()
            else:
                raise ValueError(f"Unsupported provider: {provider}")

            label = _extract_label(raw, label_set)
            sink.append({
                "timestamp": _utc_now_iso(), "model": model,
                "raw_output": raw, "parsed_output": str(label), "status": "ok",
            })
            if not str(label).strip():
                raise RuntimeError("Empty model response")
            return label

        except Exception as e:
            sink.append({
                "timestamp": _utc_now_iso(), "model": run_cfg.get("model"),
                "raw_output": "", "parsed_output": "", "status": "error", "error": str(e),
            })
            if _is_fatal_error(e):
                return f"ERROR_FATAL: {e}"
            if attempt < retries - 1:
                wait = delay * (attempt + 1)
                time.sleep(wait)
            else:
                return f"ERROR: {e}"


def _predict_task(task):
    local_logs = []
    label_set = task["label_set"]
    constraint = task["constraint_text"]
    prompt = task["prompt_text"]
    run_cfg = task["run_cfg"]

    for output_try in range(1, 4):
        pred = call_llm(run_cfg, prompt, label_set, constraint, retries=2, delay=1.5, log_sink=local_logs)
        if pred and not pred.startswith("ERROR"):
            return {"row_idx": task["row_idx"], "col": task["col"], "prediction": pred, "logs": local_logs}
        if output_try < 3:
            time.sleep(0.8 * output_try)

    return {"row_idx": task["row_idx"], "col": task["col"], "prediction": pred, "logs": local_logs}


def _run_parallel_predictions(tasks, desc, workers):
    if not tasks:
        return []
    worker_count = min(workers, len(tasks))
    if worker_count <= 1:
        return [_predict_task(t) for t in tqdm(tasks, desc=desc, file=sys.stdout)]
    results = []
    print(f"Launching {len(tasks)} tasks, {worker_count} workers — {desc}", flush=True)
    with ProcessPoolExecutor(max_workers=worker_count) as ex:
        futs = [ex.submit(_predict_task, t) for t in tasks]
        for fut in tqdm(as_completed(futs), total=len(futs), desc=desc, file=sys.stdout):
            results.append(fut.result())
    return results


def _normalize_for_scoring(val, label_set, attribute=None):
    if pd.isna(val):
        return val
    raw = str(val).strip()
    lowered = raw.lower()
    if lowered in {"na", "n/a", "n.a.", "none", "tiada", "tidak ada"} and "N/A" in label_set:
        return "N/A"
    for lbl in sorted(label_set, key=len, reverse=True):
        if lbl.lower() == lowered:
            return lbl
    if attribute in ATTRIBUTE_ALIASES:
        for alias, canonical in ATTRIBUTE_ALIASES[attribute].items():
            if alias == lowered:
                return canonical
    return raw


# ── Step 1: Write pre-retry markdown with error counts ───────────────────────
def _write_accuracy_md_with_errors(df, path):
    all_model_names = [m["name"] for m in RETRY_MODELS]
    # Include all models that appear in predictions CSV
    all_names_in_csv = []
    for attr in ATTRIBUTE_LABEL_MAP:
        for col in df.columns:
            if col.endswith(f"_{attr}_ms"):
                name = col[: -(len(attr) + 4)]
                if name not in all_names_in_csv:
                    all_names_in_csv.append(name)

    lines = ["## MS — Accuracy with Error Counts\n"]
    for attr, label_set in ATTRIBUTE_LABEL_MAP.items():
        lines.append(f"### {attr}\n")
        lines.append("| Model | Accuracy | Errors |")
        lines.append("|---|---|---|")
        gt = df[attr].apply(lambda x: _normalize_for_scoring(x, label_set, attr))
        rows = []
        for name in all_names_in_csv:
            col = f"{name}_{attr}_ms"
            if col not in df.columns:
                continue
            pred_raw = df[col]
            pred = pred_raw.apply(lambda x: _normalize_for_scoring(x, label_set, attr))
            acc = (gt == pred).mean()
            errs = pred_raw.astype(str).str.startswith("ERROR").sum()
            model_ver = MODEL_VERSION_MAP.get(name, name)
            rows.append((acc, model_ver, f"{acc:.2%}", int(errs)))
        rows.sort(key=lambda r: -r[0])
        for _, model_ver, acc_str, errs in rows:
            lines.append(f"| {model_ver} | {acc_str} | {errs} |")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Written: {path.resolve()}", flush=True)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    df = pd.read_csv(PREDICTIONS_CSV)
    print(f"Loaded {len(df)} rows from {PREDICTIONS_CSV}", flush=True)

    # Write pre-retry snapshot with error counts
    _write_accuracy_md_with_errors(df, ACCURACY_MD_ERRORS)
    print("\n--- Pre-retry error counts ---", flush=True)
    for m in RETRY_MODELS:
        for lang_suffix in ["", "_ms"]:
            for attr in ATTRIBUTE_LABEL_MAP:
                col = f"{m['name']}_{attr}{lang_suffix}"
                if col in df.columns:
                    n = df[col].astype(str).str.startswith("ERROR").sum()
                    if n:
                        print(f"  {col}: {n} errors", flush=True)

    # Process each model sequentially (different rate limits)
    total_patched = 0
    for run_cfg in RETRY_MODELS:
        model_name = run_cfg["name"]
        workers = run_cfg.get("workers", PARALLEL_WORKERS)
        print(f"\n{'='*60}", flush=True)
        print(f"  Model: {model_name} ({run_cfg['model']})  workers={workers}", flush=True)
        print(f"{'='*60}", flush=True)

        # Build tasks for all ERROR cells — both EN and MS variants
        tasks = []
        for lang_suffix, prompt_map, constraint in [
            ("",    EN_PROMPTS, EN_CONSTRAINT),
            ("_ms", MS_PROMPTS, MS_CONSTRAINT),
        ]:
            for attr, label_set in ATTRIBUTE_LABEL_MAP.items():
                col = f"{model_name}_{attr}{lang_suffix}"
                if col not in df.columns:
                    continue
                error_mask = df[col].astype(str).str.startswith("ERROR")
                error_rows = df.index[error_mask].tolist()
                if not error_rows:
                    lang_label = "EN" if lang_suffix == "" else "MS"
                    print(f"  [OK]   {lang_label} {attr}: 0 errors", flush=True)
                    continue
                lang_label = "EN" if lang_suffix == "" else "MS"
                print(f"  [RETRY] {lang_label} {attr}: {len(error_rows)} errors", flush=True)

                if lang_suffix == "_ms":
                    strict_tail = "\n\nPulangkan tepat satu label daripada set ini dan tiada teks lain: " + ", ".join(label_set)
                else:
                    strict_tail = "\n\nOutput exactly one label from this set and nothing else: " + ", ".join(label_set)

                for row_idx in error_rows:
                    text = df.loc[row_idx, "Text"]
                    tasks.append({
                        "row_idx": row_idx,
                        "col": col,
                        "run_cfg": {k: v for k, v in run_cfg.items() if k != "workers"},
                        "prompt_text": prompt_map[attr].format(TEXT=text) + strict_tail,
                        "label_set": label_set,
                        "constraint_text": constraint,
                    })

        if not tasks:
            print(f"  No errors to retry for {model_name}.", flush=True)
            continue

        print(f"  Total tasks: {len(tasks)}", flush=True)
        retry_results = _run_parallel_predictions(
            tasks,
            desc=f"{model_name} retry",
            workers=workers,
        )

        # Patch the dataframe
        patched = 0
        still_error = 0
        for item in retry_results:
            new_pred = item["prediction"]
            if new_pred and not new_pred.startswith("ERROR"):
                df.at[item["row_idx"], item["col"]] = new_pred
                patched += 1
            else:
                still_error += 1
        total_patched += patched
        print(f"  Patched: {patched}  Still ERROR: {still_error}", flush=True)

        # Save after each model in case of interruption
        df.to_csv(PREDICTIONS_CSV, index=False)
        print(f"  Saved updated predictions → {PREDICTIONS_CSV.resolve()}", flush=True)

        IO_LOGS.extend(log for item in retry_results for log in item["logs"])

    # Save IO logs
    LOG_JSON.parent.mkdir(parents=True, exist_ok=True)
    LOG_JSON.write_text(json.dumps(IO_LOGS, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nIO logs saved → {LOG_JSON.resolve()}", flush=True)

    # Re-write updated accuracy markdown (with error counts)
    _write_accuracy_md_with_errors(df, ACCURACY_MD_ERRORS)

    # Print final accuracy report
    print(f"\n{'='*60}", flush=True)
    print("  Final MS Accuracy (post-retry)", flush=True)
    print(f"{'='*60}", flush=True)

    all_names_in_csv = []
    for attr in ATTRIBUTE_LABEL_MAP:
        for col in df.columns:
            if col.endswith(f"_{attr}_ms"):
                name = col[: -(len(attr) + 4)]
                if name not in all_names_in_csv:
                    all_names_in_csv.append(name)

    for attr, label_set in ATTRIBUTE_LABEL_MAP.items():
        print(f"\n  {attr}:", flush=True)
        gt = df[attr].apply(lambda x: _normalize_for_scoring(x, label_set, attr))
        rows = []
        for name in all_names_in_csv:
            col = f"{name}_{attr}_ms"
            if col not in df.columns:
                continue
            pred_raw = df[col]
            pred = pred_raw.apply(lambda x: _normalize_for_scoring(x, label_set, attr))
            acc = (gt == pred).mean()
            errs = pred_raw.astype(str).str.startswith("ERROR").sum()
            model_ver = MODEL_VERSION_MAP.get(name, name)
            rows.append((acc, model_ver, errs))
        rows.sort(key=lambda r: -r[0])
        print(f"  {'Model':<45} {'Accuracy':>10}  {'Errors':>6}", flush=True)
        for acc, model_ver, errs in rows:
            print(f"  {model_ver:<45} {acc:>10.2%}  {errs:>6}", flush=True)

    print(f"\nTotal cells patched: {total_patched}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
