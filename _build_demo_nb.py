#!/usr/bin/env python3
"""Generate demo.ipynb — one-click demo notebook for Maylie."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {
    "display_name": "Python 3 (base)",
    "language": "python",
    "name": "base",
}

cells = []

# ── 0. Title ─────────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""\
# Maylie — Malaysian Particle Corpus: Live Collection Demo

**Purpose:** Fresh mini-scrape (Reddit + Twitter/X) with live linguistic filtering.
**Particles:** `kan` · `ke/ka` · `eh/ek` · `kot/kut`
**Demo target:** 20 matched records per platform (~2–3 min to run)

> Run all cells with **Kernel → Restart & Run All**."""))

# ── 1. Setup ─────────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("## 1. Setup"))

cells.append(nbf.v4.new_code_cell("""\
import sys, json, re, time, random, logging
from pathlib import Path
import pandas as pd
from IPython.display import display

ROOT = Path(".")
sys.path.insert(0, str(ROOT))

import scraper  # import helpers from the main collector

DEMO_TARGET  = 20                          # records per platform (keep small for demo)
DEMO_REDDIT  = ROOT / "demo_reddit.json"
DEMO_TWITTER = ROOT / "demo_twitter.json"

logging.basicConfig(format="%(asctime)s  %(message)s", datefmt="%H:%M:%S", level=logging.INFO)
print(f"Setup complete.  Demo target: {DEMO_TARGET} records per platform.")"""))

# ── 2. Patterns ───────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""\
## 2. Linguistic Filters — Regex Patterns

Four Malay discourse particles, each with refined detection logic.

| Particle | Key constraint | Example match |
|----------|---------------|---------------|
| **kan** | Word boundary prevents matching names (e.g. *Kannappan*) | `tau kan?` |
| **ke/ka** | `ke` must be **utterance-final**; `ke-2` ordinals excluded | `nak pergi ke?` |
| **eh/ek** | Covers elongated forms (`ehh`, `ekkk`) | `betul ke eh` |
| **kot/kut** | Negative lookbehind `(?<!#)` removes hashtag noise | `kot lah` |"""))

cells.append(nbf.v4.new_code_cell("""\
_RAW = [
    r"\\bkan+n*[\\?\\!\\.\\~,]*\\b",
    r"(?m)(?:\\bker+[\\?\\!\\.\\~,]*\\b|\\bka+[\\?\\!\\.\\~,]*\\b|\\bke\\b(?!-\\d)(?=\\s*[?.!,\U0001f602\U0001f605\U0001f923\U0001f62d]*\\s*$))",
    r"\\b(eh+h*|ek+k*)[\\?\\!\\.\\~,]*\\b",
    r"(?<!#)\\b(ko|ku)t+t*[\\?\\!\\.\\~,]*\\b",
]
_PAT   = [re.compile(p, re.IGNORECASE) for p in _RAW]
_NAMES = ["kan", "ke/ka", "eh/ek", "kot/kut"]

def matched(text):
    return [n for p, n in zip(_PAT, _NAMES) if p.search(text)] if text else []

# --- live test cases --------------------------------------------------------
TESTS = [
    # (sentence,                    should_match, note)
    ("tau kan?",                    True,  "kan  PASS  discourse particle"),
    ("Dr Kannappan hadir hari ini", False, "kan  PASS  inside a name — word boundary blocks"),
    ("nak pergi ke?",               True,  "ke   PASS  utterance-final before ?"),
    ("dia pergi ke sekolah lah",    False, "ke   PASS  prepositional — not utterance-final"),
    ("tahniah ke-2 pemenang",       False, "ke   PASS  ordinal ke-2 excluded"),
    ("betul ke \U0001f602",         True,  "ke   PASS  utterance-final before emoji"),
    ("eh, betul ke?",               True,  "eh   PASS  affective particle"),
    ("kot lah dia",                 True,  "kot  PASS  epistemic particle"),
    ("#kot trending harini",        False, "kot  PASS  inside hashtag — lookbehind blocks"),
]

rows = []
for text, expect, note in TESTS:
    hits  = matched(text)
    ok    = bool(hits) == expect
    rows.append({"sentence": text, "particles": hits or "—", "expected": expect, "ok": "PASS" if ok else "FAIL", "note": note})

df = pd.DataFrame(rows)[["sentence", "particles", "ok"]]
display(df)"""))

# ── 3. Reddit ─────────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""\
## 3. Reddit Collection Demo

Searches r/malaysia, r/malaysians, r/bolehland, r/KualaLumpur, r/learnmalay
using particle-related search terms, then filters with the regex patterns above."""))

cells.append(nbf.v4.new_code_cell("""\
from tqdm.notebook import tqdm
from datetime import datetime, timezone

reddit_demo, seen_r = [], set()
session_r  = scraper.make_reddit_session()

pbar = tqdm(total=DEMO_TARGET, desc="Reddit")

for sub in scraper.SUBREDDITS:
    if len(reddit_demo) >= DEMO_TARGET:
        break
    for term in scraper.REDDIT_SEARCH_TERMS:
        if len(reddit_demo) >= DEMO_TARGET:
            break
        after = None
        while len(reddit_demo) < DEMO_TARGET:
            params = {"q": term, "sort": "new", "type": "link",
                      "limit": 100, "raw_json": 1, "restrict_sr": 1}
            if after:
                params["after"] = after
            try:
                js = scraper._reddit_get(session_r,
                    f"https://oauth.reddit.com/r/{sub}/search.json", params)
            except Exception as e:
                break
            children = js.get("data", {}).get("children", [])
            after    = js.get("data", {}).get("after")
            new = 0
            for child in children:
                if len(reddit_demo) >= DEMO_TARGET:
                    break
                d   = child.get("data", {})
                pid = d.get("id", "")
                if not pid or f"p_{pid}" in seen_r:
                    continue
                seen_r.add(f"p_{pid}")
                title    = d.get("title", "")
                selftext = d.get("selftext", "")
                text     = f"{title} {selftext}".strip()
                if scraper.is_malay(text):
                    reddit_demo.append(scraper.make_record(
                        text, "reddit",
                        id=f"p_{pid}", type="post",
                        subreddit=d.get("subreddit", sub),
                        title=title,
                        url=f"https://reddit.com{d.get('permalink', '')}",
                        author=d.get("author", "[deleted]"),
                        score=d.get("score", 0),
                        num_comments=d.get("num_comments", 0),
                        created_utc=datetime.fromtimestamp(
                            d.get("created_utc", 0), tz=timezone.utc).isoformat(),
                    ))
                    pbar.update(1)
                    new += 1
            if not after or not children:
                break
            time.sleep(random.uniform(1.0, 2.0))

pbar.close()
json.dump(reddit_demo, open(DEMO_REDDIT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"Reddit demo complete: {len(reddit_demo)} records → {DEMO_REDDIT}")"""))

# ── 4. Twitter ────────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""\
## 4. Twitter/X Collection Demo

Paginates through curated high-volume Malay-language accounts via GraphQL UserTweets."""))

cells.append(nbf.v4.new_code_cell("""\
twitter_demo, seen_t = [], set()
session_t = scraper.make_tw_session()

pbar = tqdm(total=DEMO_TARGET, desc="Twitter")

for screen_name in scraper.MALAY_ACCOUNTS:
    if len(twitter_demo) >= DEMO_TARGET:
        break
    user_id = scraper._get_twitter_user_id(session_t, screen_name)
    if not user_id:
        continue
    cursor, empty = None, 0
    while len(twitter_demo) < DEMO_TARGET:
        vars_ = {"userId": user_id, "count": 100,
                 "includePromotedContent": False,
                 "withQuotedTweets": True, "withVoice": False}
        if cursor:
            vars_["cursor"] = cursor
        code, js = scraper._tw_gql(session_t, "UserTweets", vars_)
        if code != 200 or not js:
            break
        tweets, cursor = scraper._parse_user_tweets(js)
        if not tweets:
            empty += 1
            if empty >= 3:
                break
            continue
        empty = 0
        for tw in tweets:
            if len(twitter_demo) >= DEMO_TARGET:
                break
            tid, text = tw.get("id", ""), tw.get("text", "")
            if not tid or tid in seen_t or not text:
                continue
            seen_t.add(tid)
            if not scraper.is_malay(text):
                continue
            twitter_demo.append(scraper.make_record(
                text, "twitter",
                id=tid, author=tw.get("author", screen_name),
                display_name=tw.get("display_name", ""),
                created_at=tw.get("created_at", ""),
                lang=tw.get("lang", ""),
                retweet_count=tw.get("retweet_count", 0),
                favorite_count=tw.get("favorite_count", 0),
                url=f"https://x.com/{tw.get('author', screen_name)}/status/{tid}",
                account_scraped=screen_name,
            ))
            pbar.update(1)
        if not cursor:
            break
        time.sleep(random.uniform(1.0, 2.0))
    time.sleep(random.uniform(1.0, 2.0))

pbar.close()
json.dump(twitter_demo, open(DEMO_TWITTER, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"Twitter demo complete: {len(twitter_demo)} records → {DEMO_TWITTER}")"""))

# ── 5. Results ────────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("## 5. Results"))

cells.append(nbf.v4.new_code_cell("""\
combined = reddit_demo + twitter_demo

# ── summary table ──────────────────────────────────────────────────────────
print("=" * 50)
print(f"  Reddit  : {len(reddit_demo):>3} records")
print(f"  Twitter : {len(twitter_demo):>3} records")
print(f"  Total   : {len(combined):>3} records")
print()
print("  Particle breakdown:")
for name in ["kan", "ke/ka", "eh/ek", "kot/kut"]:
    cnt = sum(1 for r in combined if name in r.get("matched_patterns", []))
    pct = cnt / len(combined) * 100 if combined else 0
    bar = "#" * int(pct / 5)
    print(f"    {name:<8} {cnt:>3}  ({pct:4.0f}%)  {bar}")
print("=" * 50)

# ── sample records ──────────────────────────────────────────────────────────
print()
rows = []
for rec in combined:
    rows.append({
        "platform":  rec["source"],
        "particles": " + ".join(rec.get("matched_patterns", [])),
        "text":      rec["text"][:110] + ("…" if len(rec["text"]) > 110 else ""),
    })

df = pd.DataFrame(rows)
pd.set_option("display.max_colwidth", 120)
display(df)"""))

nb.cells = cells

out = "demo.ipynb"
nbf.write(nb, out)
print(f"Written: {out}  ({len(cells)} cells)")
