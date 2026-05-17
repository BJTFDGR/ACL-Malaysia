#!/usr/bin/env python3
"""Live progress monitor for twitter_data.json — runs in terminal."""
import json, time, sys
from pathlib import Path

TARGET = 3000
path = Path(__file__).parent / "twitter_data.json"

prev = 0
start_t = time.time()
start_n = None

print(f"Watching {path} — target {TARGET}")
print("-" * 60)

while True:
    try:
        with open(path) as f:
            d = json.load(f)
        n = len(d)
        if start_n is None:
            start_n = n
        elapsed = time.time() - start_t
        gained = n - start_n
        rate = gained / elapsed * 60 if elapsed > 5 else 0
        pct = n / TARGET * 100
        bar_len = 40
        filled = int(bar_len * n / TARGET)
        bar = "█" * filled + "░" * (bar_len - filled)
        eta_str = ""
        if rate > 0:
            eta_min = (TARGET - n) / (rate / 60)
            eta_str = f"  ETA {eta_min:.0f}min"
        indicator = " +" if n > prev else "   "
        print(f"\r[{bar}] {n}/{TARGET} ({pct:.1f}%){indicator}  +{gained} this run  {rate:.1f}/min{eta_str}   ", end="", flush=True)
        prev = n
        if n >= TARGET:
            print(f"\n✓ Reached {TARGET}!")
            break
    except Exception as e:
        print(f"\r[waiting for data file...] {e}   ", end="", flush=True)
    time.sleep(5)
