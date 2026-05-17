#!/usr/bin/env python3
"""Re-filter JSON data files using updated regex patterns, then report counts."""
import json, re
from pathlib import Path

BASE = Path(__file__).parent

_RAW = [
    r"\bkan+n*[\?\!\.\~,]*\b",
    r"(?m)(?:\bker+[\?\!\.\~,]*\b|\bka+[\?\!\.\~,]*\b|\bke\b(?!-\d)(?=\s*[?.!,😂😅🤣😭]*\s*$))",
    r"\b(eh+h*|ek+k*)[\?\!\.\~,]*\b",
    r"(?<!#)\b(ko|ku)t+t*[\?\!\.\~,]*\b",
]
_PAT   = [re.compile(p, re.IGNORECASE) for p in _RAW]
_NAMES = ["kan", "ke/ka", "eh/ek", "kot/kut"]

def matched(text):
    return [n for p, n in zip(_PAT, _NAMES) if p.search(text)] if text else []

for fname in ["reddit_data.json", "twitter_data.json"]:
    path = BASE / fname
    data = json.load(open(path))
    kept, dropped_ke, dropped_kot, dropped_both = [], 0, 0, 0
    for rec in data:
        old = set(rec.get("matched_patterns", []))
        new = matched(rec["text"])
        if "ke/ka" in old and "ke/ka" not in new:
            dropped_ke += 1
        if "kot/kut" in old and "kot/kut" not in new:
            dropped_kot += 1
        if new:
            rec["matched_patterns"] = new
            kept.append(rec)
        else:
            dropped_both += 1
    print(f"\n{fname}: {len(data)} → {len(kept)} kept  ({len(data)-len(kept)} removed)")
    print(f"  ke/ka newly excluded : {dropped_ke}")
    print(f"  kot/kut newly excluded: {dropped_kot}")
    print(f"  records with NO match: {dropped_both}")
    json.dump(kept, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"  saved.")

print("\nDone.")
