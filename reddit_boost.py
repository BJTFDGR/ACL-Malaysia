#!/usr/bin/env python3
"""
Reddit boost: adds more subreddits + global search + comment fetching
to top up reddit_data.json from 1855 → 3000.
Run in parallel with the main Twitter scraper.
"""
import json, re, time, random, logging
from datetime import datetime, timezone
from pathlib import Path
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

TARGET_REDDIT = 1000
OUTPUT_DIR    = Path(__file__).parent
REDDIT_COOKIES_FILE = OUTPUT_DIR / "www.reddit.com_cookies.txt"

_RAW = [
    r"\bkan+n*[\?\!\.\~,]*\b",
    r"(?m)(?:\bker+[\?\!\.\~,]*\b|\bka+[\?\!\.\~,]*\b|\bke\b(?!-\d)(?=\s*[?.!,😂😅🤣😭]*\s*$))",
    r"\b(eh+h*|ek+k*)[\?\!\.\~,]*\b",
    r"(?<!#)\b(ko|ku)t+t*[\?\!\.\~,]*\b",
    r"\bke\b(?!-\d)(?=\s+(?:lah|la|je|pun)\b)",
]
_PAT   = [re.compile(p, re.IGNORECASE) for p in _RAW]
_NAMES = ["kan", "ke/ka", "eh/ek", "kot/kut", "ke_clausal"]

_INDONESIAN_RE = re.compile(
    r"\b(aing|aja|ayo|banget|bgt|belom|ben|bener|bilang|bisa|cakep|capek|"
    r"cewek|cowok|de|det|dong|doang|elu|emang|emg|en|enggak|er|ga|gak|"
    r"gausah|gede|gua|gue|gueh|gw|gwe|heb|houden|ih|ik|iya|kagak|kalian|"
    r"kantor|kerjaan|kalo|klo|ko|kok|libur|loe|lu|mau|maar|mensen|milih|"
    r"ngapain|ngga|nggak|nih|pake|sampe|sih|temen|udah|udh|voor|wkwk|yo|zien)\b",
    re.IGNORECASE
)

def matched(text):
    return [n for p, n in zip(_PAT, _NAMES) if p.search(text)] if text else []

def is_malay(text):
    return bool(matched(text)) and not _INDONESIAN_RE.search(text)

def make_record(text, source, **meta):
    return {"text": text.strip(), "source": source, "matched_patterns": matched(text),
            "collected_at": datetime.now(timezone.utc).isoformat(), **meta}

def load_json(path):
    return json.load(open(path, encoding="utf-8")) if path.exists() else []

def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_netscape_cookies(path):
    cookies = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) >= 7:
                cookies[parts[5]] = parts[6]
    return cookies

def make_reddit_session():
    cookies = load_netscape_cookies(REDDIT_COOKIES_FILE)
    token = cookies.get("token_v2", "")
    s = requests.Session()
    for name, value in cookies.items():
        s.cookies.set(name, value, domain=".reddit.com")
    s.headers.update({
        "User-Agent":    "script:MalayDataCollector:1.0 (by /u/temp_research)",
        "Authorization": f"Bearer {token}",
        "Accept":        "application/json",
        "Accept-Language": "en-US,en;q=0.9",
    })
    return s

@retry(stop=stop_after_attempt(6), wait=wait_exponential(min=3, max=90),
       retry=retry_if_exception_type((requests.RequestException, ValueError)),
       before_sleep=before_sleep_log(log, logging.WARNING))
def _reddit_get(session, url, params):
    r = session.get(url, params=params, timeout=20)
    if r.status_code == 429:
        wait = int(r.headers.get("Retry-After", 60))
        log.warning(f"Reddit 429 — sleeping {wait}s")
        time.sleep(wait)
        raise requests.RequestException("rate-limited")
    if r.status_code == 401:
        log.warning("Reddit 401 — token may have expired")
        raise requests.RequestException("auth-401")
    r.raise_for_status()
    return r.json()

def _flatten_comments(listing, depth=0):
    results = []
    if depth > 4:
        return results
    for child in listing.get("data", {}).get("children", []):
        if child.get("kind") == "t1":
            d = child["data"]
            results.append(d)
            if isinstance(d.get("replies"), dict):
                results.extend(_flatten_comments(d["replies"], depth + 1))
    return results

# Extended subreddit list
SUBREDDITS_EXTENDED = [
    # New subreddits from annotators (searched first)
    "OkeyRakanMalaysia", "Ajar_Malaysia", "Sabah", "trulyMalaysians",
    # Malaysian subreddits
    "malaysia", "malaysians", "bolehland", "KualaLumpur", "learnmalay",
    "Malay", "bahasamelayu",
    # Other SEA subreddits with Malay content
    "singapore", "indonesia",
]

# Extended search terms (more phrase-based, higher hit rate)
SEARCH_TERMS_EXTENDED = [
    # Direct particle phrases (high hit rate)
    "lah kan", "betul kan", "eh kan", "tau kan", "kan je", "kan ke",
    "boleh ke", "nak ke", "ke mana", "pergi ke", "balik ke", "datang ke",
    "eh betul", "eh takpe", "eh okay", "eh bagus", "eh comel",
    "kot la", "eh kot", "lawak kot", "pelik kot",
    "macam mana", "macam tu", "macam ni",
    # Common Malay phrases with particles
    "rasa macam", "tak boleh ke", "boleh tak", "tak pe", "oklah",
    "betul ke", "serious ke", "ok ke", "ok la", "ok kan",
    "kan dah", "kan best", "kan bagus", "kan comel",
    # Basic particles
    "kan", "eh", "kot", "ke tak",
    # Everyday phrases
    "makan ke", "tidur ke", "kerja ke",
    "dah makan", "dah lah", "apahal",
    "weh", "wei", "bro",
]

# Global search terms (no restrict_sr) — these are distinctive enough to be Malay
GLOBAL_TERMS = [
    "lah kan", "betul kan", "macam mana eh", "eh kot",
    "boleh ke tak", "lawak kot", "tau kan", "kan je",
    "balik ke malaysia", "pergi ke mana",
]

def collect_reddit_boost(target=TARGET_REDDIT):
    path     = OUTPUT_DIR / "reddit_data_new.json"
    old_path = OUTPUT_DIR / "reddit_data.json"
    data = load_json(path)           # resume new file if it exists
    seen = {r["id"] for r in load_json(old_path)} | {r["id"] for r in data}
    log.info(f"Reddit boost: {len(data)} new records so far (target {target})")

    session    = make_reddit_session()
    pbar       = tqdm(total=target, initial=len(data), desc="Reddit+", unit="rec")
    since_save = 0
    SAVE_EVERY = 50

    def checkpoint():
        nonlocal since_save
        save_json(data, path)
        since_save = 0
        log.info(f"  ckpt → {len(data)} records")

    def add_text(text, pid, d, sub):
        nonlocal since_save
        if pid in seen or not is_malay(text):
            return False
        seen.add(pid)
        data.append(make_record(text, "reddit", id=pid, type="post" if pid.startswith("p_") else "comment",
                                subreddit=sub, text_preview=text[:50], score=d.get("score", 0),
                                created_utc=datetime.fromtimestamp(d.get("created_utc", 0), tz=timezone.utc).isoformat()))
        pbar.update(1)
        since_save += 1
        return True

    def search_subreddit(sub, term, global_search=False):
        after = None
        empty_pages = 0
        while len(data) < target:
            params = {"q": term, "sort": "new", "type": "link", "limit": 100, "raw_json": 1}
            if not global_search:
                params["restrict_sr"] = 1
            if after:
                params["after"] = after
            base = "https://oauth.reddit.com/search.json" if global_search else f"https://oauth.reddit.com/r/{sub}/search.json"
            try:
                js = _reddit_get(session, base, params)
            except Exception as e:
                log.warning(f"{'global' if global_search else f'r/{sub}'} '{term}': {e}")
                break

            children = js.get("data", {}).get("children", [])
            after    = js.get("data", {}).get("after")
            if not children:
                break

            new_this = 0
            for child in children:
                if len(data) >= target:
                    break
                d   = child.get("data", {})
                pid = d.get("id", "")
                if not pid:
                    continue
                sub2     = d.get("subreddit", sub)
                title    = d.get("title", "")
                selftext = d.get("selftext", "")
                # Try title + selftext as post record
                combined = f"{title} {selftext}".strip()
                if add_text(combined, f"p_{pid}", d, sub2):
                    new_this += 1

                # Also try to fetch comments for posts with many comments
                # (disabled to save API budget — re-enable if needed)

                if since_save >= SAVE_EVERY:
                    checkpoint()
                time.sleep(random.uniform(0.05, 0.15))

            empty_pages = 0 if new_this else empty_pages + 1
            if empty_pages >= 2 or not after:
                break

            time.sleep(random.uniform(6.0, 8.0))

    # Phase 1: Extended subreddits with extended terms
    log.info("Phase 1: Extended subreddits")
    for sub in SUBREDDITS_EXTENDED:
        if len(data) >= target:
            break
        for term in SEARCH_TERMS_EXTENDED:
            if len(data) >= target:
                break
            search_subreddit(sub, term)

    # Phase 2: Global search (no restrict_sr)
    if len(data) < target:
        log.info("Phase 2: Global Reddit search")
        for term in GLOBAL_TERMS:
            if len(data) >= target:
                break
            search_subreddit("", term, global_search=True)

    # Phase 3: Fetch comments from top posts (if still short)
    if len(data) < target:
        log.info(f"Phase 3: Comment harvest (need {target - len(data)} more)")
        for sub in ["malaysia", "malaysians"]:
            if len(data) >= target:
                break
            # Get top posts (not searched yet)
            for sort in ["top", "hot"]:
                if len(data) >= target:
                    break
                try:
                    js = _reddit_get(session, f"https://oauth.reddit.com/r/{sub}/{sort}.json",
                                     {"limit": 100, "raw_json": 1, "t": "month"})
                    children = js.get("data", {}).get("children", [])
                    for child in children:
                        if len(data) >= target:
                            break
                        d   = child.get("data", {})
                        pid = d.get("id", "")
                        sub2 = d.get("subreddit", sub)
                        if not pid or d.get("num_comments", 0) < 5:
                            continue
                        # Fetch comments
                        try:
                            cr = _reddit_get(session, f"https://oauth.reddit.com/r/{sub}/comments/{pid}.json",
                                            {"limit": 200, "depth": 5, "raw_json": 1})
                            if isinstance(cr, list) and len(cr) >= 2:
                                comments = _flatten_comments(cr[1])
                                for c in comments:
                                    if len(data) >= target:
                                        break
                                    cid  = c.get("id", "")
                                    body = c.get("body", "")
                                    if cid and body and body not in ("[deleted]", "[removed]"):
                                        add_text(body, f"c_{cid}", c, sub2)
                                        time.sleep(0.05)
                            if since_save >= SAVE_EVERY:
                                checkpoint()
                        except Exception as e:
                            log.debug(f"Comment fetch error: {e}")
                        time.sleep(random.uniform(6.5, 8.5))
                except Exception as e:
                    log.warning(f"r/{sub} {sort}: {e}")

    pbar.close()
    save_json(data[:target], path)
    log.info(f"Reddit boost done: {min(len(data), target)}/{target}")

if __name__ == "__main__":
    collect_reddit_boost(TARGET_REDDIT)
