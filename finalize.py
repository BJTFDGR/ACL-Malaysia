#!/usr/bin/env python3
"""Merge reddit_data.json + twitter_data.json → combined_data.json with summary."""
import json, re
from pathlib import Path
from datetime import datetime, timezone

OUTPUT_DIR = Path(__file__).parent
_RAW = [
    r"\bkan+n*[\?\!\.\~,]*\b",
    r"(?m)(?:\bker+[\?\!\.\~,]*\b|\bka+[\?\!\.\~,]*\b|\bke\b(?!-\d)(?=\s*[?.!,😂😅🤣😭]*\s*$))",
    r"\b(eh+h*|ek+k*)[\?\!\.\~,]*\b",
    r"(?<!#)\b(ko|ku)t+t*[\?\!\.\~,]*\b",
    r"\bke\b(?!-\d)(?=\s+(?:lah|la|je|pun)\b)",
]
_PAT   = [re.compile(p, re.IGNORECASE) for p in _RAW]
_NAMES = ["kan", "ke/ka", "eh/ek", "kot/kut", "ke_clausal"]

reddit_path  = OUTPUT_DIR / "reddit_data.json"
twitter_path = OUTPUT_DIR / "twitter_data.json"
combined_path = OUTPUT_DIR / "combined_data.json"

reddit_data  = json.load(open(reddit_path))  if reddit_path.exists()  else []
twitter_data = json.load(open(twitter_path)) if twitter_path.exists() else []

combined = reddit_data[:3000] + twitter_data[:3000]
with open(combined_path, "w", encoding="utf-8") as f:
    json.dump(combined, f, ensure_ascii=False, indent=2)

print("=" * 60)
print(f"Reddit   : {len(reddit_data[:3000]):>5} → reddit_data.json")
print(f"Twitter  : {len(twitter_data[:3000]):>5} → twitter_data.json")
print(f"Combined : {len(combined):>5} → combined_data.json")
print("Pattern breakdown (combined):")
for name in _NAMES:
    cnt = sum(1 for r in combined if name in r.get("matched_patterns", []))
    print(f"  {name:<8}: {cnt:>5}")
print("Sources:")
for src in ["reddit", "twitter"]:
    cnt = sum(1 for r in combined if r.get("source") == src)
    print(f"  {src:<10}: {cnt:>5}")
print("=" * 60)
