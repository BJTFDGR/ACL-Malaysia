#!/usr/bin/env python3
"""
Twitter search scraper — uses Playwright to search for Malay particle terms.
Navigates to x.com/search?q=<term>&f=live and collects SearchTimeline API responses.
Bypasses the blocked API search endpoint by going through the browser.
"""
import asyncio, json, re, logging, sys
from datetime import datetime, timezone
from pathlib import Path
from playwright.async_api import async_playwright, TimeoutError as PWTimeout

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

OUTPUT_DIR   = Path(__file__).parent
COOKIES_FILE = OUTPUT_DIR / "x.com_cookies.txt"
OUTPUT_FILE  = OUTPUT_DIR / "twitter_data.json"
TARGET       = 3000
SAVE_EVERY   = 30

_RAW = [
    r"\bkan+n*[\?\!\.\~,]*\b",
    r"(?m)(?:\bker+[\?\!\.\~,]*\b|\bka+[\?\!\.\~,]*\b|\bke\b(?!-\d)(?=\s*[?.!,\U0001F602\U0001F605\U0001F923\U0001F62D]*\s*$))",
    r"\b(eh+h*|ek+k*)[\?\!\.\~,]*\b",
    r"(?<!#)\b(ko|ku)t+t*[\?\!\.\~,]*\b",
    r"\bke\b(?!-\d)(?=\s+(?:lah|la|je|pun)\b)",
]
_PAT   = [re.compile(p, re.IGNORECASE) for p in _RAW]
_NAMES = ["kan", "ke/ka", "eh/ek", "kot/kut", "ke_clausal"]

def matched(text):
    return [n for p, n in zip(_PAT, _NAMES) if p.search(text)] if text else []
def is_malay(text):
    return bool(matched(text))
def make_record(text, **meta):
    return {"text": text.strip(), "source": "twitter", "matched_patterns": matched(text),
            "collected_at": datetime.now(timezone.utc).isoformat(), **meta}
def load_json(path):
    return json.load(open(path, encoding="utf-8")) if path.exists() else []
def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_cookies(path):
    cookies = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"): continue
            parts = line.split("\t")
            if len(parts) >= 7:
                domain, flag, path_, secure, expiry, name, value = parts[:7]
                if name == "__cf_bm": continue
                c = {"domain": domain if domain.startswith(".") else "." + domain,
                     "path": path_, "name": name, "value": value,
                     "secure": secure.lower() == "true",
                     "sameSite": "None" if secure.lower() == "true" else "Lax"}
                if expiry and expiry.strip() not in ("0",""):
                    try: c["expires"] = int(float(expiry))
                    except: pass
                cookies.append(c)
    return cookies

def _parse_search_timeline(js):
    """Extract tweets from SearchTimeline API response."""
    tweets = []
    try:
        instructions = (js.get("data",{}).get("search_by_raw_query",{})
                          .get("search_timeline",{}).get("timeline",{})
                          .get("instructions",[]))
        for instr in instructions:
            for entry in instr.get("entries", []):
                content = entry.get("content", {})
                # Direct tweet entry
                item = content.get("itemContent", {})
                if item.get("itemType") == "TimelineTweet":
                    result = item.get("tweet_results", {}).get("result", {})
                    legacy = result.get("legacy", {})
                    user_leg = (result.get("core", {}).get("user_results", {})
                                      .get("result", {}).get("legacy", {}))
                    full_text = legacy.get("full_text") or legacy.get("text", "")
                    tid = legacy.get("id_str", "")
                    if full_text and tid:
                        tweets.append({
                            "id": tid, "text": full_text,
                            "created_at": legacy.get("created_at", ""),
                            "screen_name": user_leg.get("screen_name", ""),
                            "retweet_count": legacy.get("retweet_count", 0),
                            "favorite_count": legacy.get("favorite_count", 0),
                        })
                # Module items (carousel)
                for item2 in content.get("items", []):
                    item_c = item2.get("item", {}).get("itemContent", {})
                    if item_c.get("itemType") == "TimelineTweet":
                        result = item_c.get("tweet_results", {}).get("result", {})
                        legacy = result.get("legacy", {})
                        user_leg = (result.get("core", {}).get("user_results", {})
                                          .get("result", {}).get("legacy", {}))
                        full_text = legacy.get("full_text") or legacy.get("text", "")
                        tid = legacy.get("id_str", "")
                        if full_text and tid:
                            tweets.append({
                                "id": tid, "text": full_text,
                                "created_at": legacy.get("created_at", ""),
                                "screen_name": user_leg.get("screen_name", ""),
                                "retweet_count": legacy.get("retweet_count", 0),
                                "favorite_count": legacy.get("favorite_count", 0),
                            })
    except Exception as e:
        log.debug(f"parse error: {e}")
    return tweets

# Search terms covering all 5 particle patterns
SEARCH_TERMS = [
    # kan — confirmation particle
    "betul kan", "kan best", "kan dah", "best kan",
    "lawak kan", "comel kan", "sedap kan", "pelik kan",
    # ke/ka — question particle (utterance-final)
    "betul ke", "serius ke", "lawak ke", "best ke",
    "nak pergi ke", "dah makan ke",
    # eh/ek — exclamation/surprise
    "eh betul", "eh kenapa", "eh comel", "eh lawak",
    "betul eh", "serius eh",
    # kot — uncertainty/hedge
    "lawak kot", "lambat kot", "penat kot", "best kot",
    "malas kot", "lapar kot",
    # ke_clausal — ke lah/la/je/pun
    "ke lah", "ke la", "ke je", "ke pun",
    # combinations
    "kan lah", "eh dah", "betul ke lah",
]


async def search_term(page, term, data, seen, target):
    """Search one term and collect results by scrolling."""
    import urllib.parse
    url = f"https://x.com/search?q={urllib.parse.quote(term)}&src=typed_query&f=live"
    log.info(f"Searching: '{term}'")

    def is_search(response):
        return "SearchTimeline" in response.url and "/graphql/" in response.url

    new_count = 0
    RATE_LIMIT_SLEEP = 780

    try:
        async with page.expect_response(is_search, timeout=30000) as resp_info:
            await page.goto(url, wait_until="domcontentloaded", timeout=30000)
        resp = await resp_info.value
        if resp.status == 429:
            log.warning(f"  429 on search — sleeping {RATE_LIMIT_SLEEP}s")
            await asyncio.sleep(RATE_LIMIT_SLEEP)
            return new_count
        body = await resp.body()
        js = json.loads(body)
        tweets = _parse_search_timeline(js)
    except PWTimeout:
        log.warning(f"  '{term}': timeout (possibly no SearchTimeline response)")
        return new_count
    except Exception as e:
        log.warning(f"  '{term}': error: {e}")
        return new_count

    def process_tweets(tweets):
        nonlocal new_count
        for tw in tweets:
            if len(data) >= target: break
            tid, text = tw["id"], tw["text"]
            if tid in seen: continue
            seen.add(tid)
            if not is_malay(text): continue
            data.append(make_record(text, id=tid,
                                    screen_name=tw["screen_name"],
                                    created_at=tw["created_at"],
                                    retweet_count=tw["retweet_count"],
                                    favorite_count=tw["favorite_count"],
                                    search_term=term))
            new_count += 1

    process_tweets(tweets)
    log.info(f"  '{term}': page 1 → {new_count} new records")

    # Scroll to load more
    since_save = 0
    for scroll_i in range(15):
        if len(data) >= target: break
        try:
            async with page.expect_response(is_search, timeout=10000) as resp_info:
                await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            resp = await resp_info.value
            if resp.status == 429:
                log.warning(f"  429 while scrolling — sleeping {RATE_LIMIT_SLEEP}s")
                await asyncio.sleep(RATE_LIMIT_SLEEP)
                break
            body = await resp.body()
            js = json.loads(body)
            tweets = _parse_search_timeline(js)
            prev = new_count
            process_tweets(tweets)
            added = new_count - prev
            since_save += added
            if since_save >= SAVE_EVERY:
                save_json(data, OUTPUT_FILE)
                since_save = 0
                log.info(f"  checkpoint → {len(data)} Twitter records")
        except PWTimeout:
            break  # no more results
        except Exception:
            break
        await asyncio.sleep(1.5)

    log.info(f"  '{term}' done: +{new_count} new records  total={len(data)}")
    return new_count


async def main():
    data = load_json(OUTPUT_FILE)
    seen = {r["id"] for r in data}
    log.info(f"Starting from {len(data)} records, target {TARGET}")

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        ctx = await browser.new_context(
            viewport={"width": 1280, "height": 900},
            user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        )
        await ctx.add_cookies(load_cookies(COOKIES_FILE))
        page = await ctx.new_page()

        for term in SEARCH_TERMS:
            if len(data) >= TARGET:
                log.info("Target reached!")
                break
            await search_term(page, term, data, seen, TARGET)
            save_json(data, OUTPUT_FILE)
            n = len(data)
            pct = n / TARGET * 100
            bar = "█" * int(40 * n / TARGET) + "░" * (40 - int(40 * n / TARGET))
            log.info(f"  Progress: [{bar}] {n}/{TARGET} ({pct:.1f}%)")
            await asyncio.sleep(3)

        await browser.close()

    save_json(data, OUTPUT_FILE)
    log.info(f"Done: {len(data)} records")

if __name__ == "__main__":
    asyncio.run(main())
