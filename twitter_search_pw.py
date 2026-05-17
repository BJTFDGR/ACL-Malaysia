#!/usr/bin/env python3
"""
Twitter search-based Playwright scraper.
Searches for Malay keyword phrases and captures SearchTimeline API responses.
Supplements twitter_data.json (avoids duplicates via seen IDs).
"""
import asyncio, json, re, logging, urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from playwright.async_api import async_playwright, TimeoutError as PWTimeout
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

TARGET_TWITTER = 750
OUTPUT_DIR     = Path(__file__).parent
TWITTER_COOKIES_FILE = OUTPUT_DIR / "x.com_cookies.txt"
SAVE_EVERY     = 10

_RAW = [
    r"\bkan+n*[\?\!\.\~,]*\b",
    r"(?m)(?:\bker+[\?\!\.\~,]*\b|\bka+[\?\!\.\~,]*\b|\bke\b(?!-\d)(?=\s*[?.!,😂😅🤣😭]*\s*$))",
    r"\b(eh+h*|ek+k*)[\?\!\.\~,]*\b",
    r"(?<!#)\b(ko|ku)t+t*[\?\!\.\~,]*\b",
]
_PAT   = [re.compile(p, re.IGNORECASE) for p in _RAW]
_NAMES = ["kan", "ke/ka", "eh/ek", "kot/kut"]

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
def make_record(text, **meta):
    return {"text": text.strip(), "source": "twitter", "matched_patterns": matched(text),
            "collected_at": datetime.now(timezone.utc).isoformat(), **meta}
def load_json(path):
    return json.load(open(path, encoding="utf-8")) if path.exists() else []
def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_cookies_for_playwright(path):
    cookies = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"): continue
            parts = line.split("\t")
            if len(parts) >= 7:
                domain, flag, path_, secure, expiry, name, value = parts[:7]
                if name == "__cf_bm": continue
                cookie = {
                    "domain": domain if domain.startswith(".") else "." + domain,
                    "path": path_, "name": name, "value": value,
                    "secure": secure.lower() == "true",
                    "sameSite": "None" if secure.lower() == "true" else "Lax",
                }
                if expiry and expiry.strip() not in ("0", ""):
                    try: cookie["expires"] = int(float(expiry))
                    except: pass
                cookies.append(cookie)
    return cookies

def _parse_search_timeline(js):
    """Parse SearchTimeline GraphQL response — same entry structure as UserTweets."""
    tweets, cursor = [], None
    try:
        instructions = (js.get("data", {}).get("search_by_raw_query", {})
                          .get("search_timeline", {}).get("timeline", {})
                          .get("instructions", []))
        for instr in instructions:
            if instr.get("type") != "TimelineAddEntries": continue
            for entry in instr.get("entries", []):
                eid = entry.get("entryId", "")
                content = entry.get("content", {})
                if "tweet-" in eid:
                    result = content.get("itemContent", {}).get("tweet_results", {}).get("result", {})
                    legacy = result.get("legacy", {})
                    user_leg = (result.get("core", {}).get("user_results", {})
                                      .get("result", {}).get("legacy", {}))
                    full_text = legacy.get("full_text") or legacy.get("text", "")
                    if full_text:
                        tweets.append({"id": legacy.get("id_str", ""), "text": full_text,
                                       "created_at": legacy.get("created_at", ""),
                                       "screen_name": user_leg.get("screen_name", ""),
                                       "retweet_count": legacy.get("retweet_count", 0),
                                       "favorite_count": legacy.get("favorite_count", 0)})
                elif "cursor-bottom" in eid:
                    val = content.get("value", "")
                    if val: cursor = val
    except Exception as e:
        log.debug(f"search parse error: {e}")
    return tweets, cursor

# Malay search queries — each should produce tweets matching our patterns
MALAY_QUERIES = [
    "takkan lah",
    "lah kan",
    "betul kan",
    "memang lah kan",
    "ye ke tak",
    "ye ke",
    "eh kan",
    "lah kot",
    "kot lah",
    "kut lah",
    "eh betul ke",
    "kan dah",
    "dah lah",
    "mana lah kan",
    "takkan kot",
    "eleh kan",
    "alah kan",
    "haish kan",
    "mane kan",
    "nak kan",
    "tau kan",
    "tahu kan",
    "betul ke",
    "ek kan",
    "lah ke",
]

async def scrape_search(page, query, data, seen, pbar, since_save_ref, target):
    """Search Twitter for a Malay query and collect matching tweets."""
    path = OUTPUT_DIR / "twitter_data_new.json"
    RATE_LIMIT_SLEEP = 780

    def is_search(response):
        return "SearchTimeline" in response.url and "/graphql/" in response.url

    encoded = urllib.parse.quote(query)
    search_url = f"https://x.com/search?q={encoded}&src=typed_query&f=live"

    pages_done = 0
    new_total = 0

    def process_js(js):
        nonlocal pages_done, new_total
        tweets, cursor = _parse_search_timeline(js)
        pages_done += 1
        for tw in tweets:
            if len(data) >= target: break
            tid, text = tw["id"], tw["text"]
            if tid in seen or not is_malay(text): continue
            seen.add(tid)
            data.append(make_record(text, id=tid, screen_name=tw["screen_name"],
                                    created_at=tw["created_at"],
                                    retweet_count=tw["retweet_count"],
                                    favorite_count=tw["favorite_count"]))
            pbar.update(1); since_save_ref[0] += 1; new_total += 1
        if since_save_ref[0] >= SAVE_EVERY:
            save_json(data, path); since_save_ref[0] = 0
            log.info(f"  ckpt → {len(data)} Twitter records saved")
        return cursor

    # Navigate to search page
    try:
        async with page.expect_response(is_search, timeout=60000) as resp_info:
            await page.goto(search_url, wait_until="domcontentloaded", timeout=30000)
        response = await resp_info.value
        if response.status == 429:
            log.warning(f"  search '{query}': 429 rate limited — sleeping {RATE_LIMIT_SLEEP}s")
            await asyncio.sleep(RATE_LIMIT_SLEEP)
            # retry once
            try:
                async with page.expect_response(is_search, timeout=60000) as resp_info:
                    await page.goto(search_url, wait_until="domcontentloaded", timeout=30000)
                response = await resp_info.value
            except PWTimeout:
                log.warning(f"  search '{query}': timeout on retry")
                return
        if response.status != 200:
            log.warning(f"  search '{query}': HTTP {response.status}")
            return
        body = await response.body()
        first_js = json.loads(body)
    except PWTimeout:
        log.warning(f"  search '{query}': timeout waiting for SearchTimeline")
        return
    except Exception as e:
        log.warning(f"  search '{query}': {type(e).__name__}: {e}")
        return

    process_js(first_js)

    # Scroll loop
    MAX_SCROLLS = 80
    no_response_scrolls = 0

    for scroll_i in range(MAX_SCROLLS):
        if len(data) >= target: break
        if no_response_scrolls >= 8: break

        try:
            async with page.expect_response(is_search, timeout=15000) as resp_info:
                await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")

            response = await resp_info.value
            if response.status == 429:
                log.warning(f"  search scroll {scroll_i}: 429 — sleeping {RATE_LIMIT_SLEEP}s")
                await asyncio.sleep(RATE_LIMIT_SLEEP)
                no_response_scrolls += 1
                continue
            body = await response.body()
            js = json.loads(body)
            process_js(js)
            no_response_scrolls = 0

        except PWTimeout:
            no_response_scrolls += 1
            await asyncio.sleep(1)
        except Exception as e:
            log.debug(f"  search scroll {scroll_i}: {type(e).__name__}: {e}")
            no_response_scrolls += 1

    log.info(f"  search '{query}': {pages_done} pages, {new_total} Malay tweets")


async def collect_twitter_search(target=TARGET_TWITTER):
    path     = OUTPUT_DIR / "twitter_data_new.json"
    old_path = OUTPUT_DIR / "twitter_data.json"
    data = load_json(path)
    seen = ({r["id"] for r in load_json(old_path) if r.get("id")} |
            {r["id"] for r in data if r.get("id")})
    log.info(f"Twitter search: {len(data)} new records so far (target {target})")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        ctx = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            viewport={"width": 1280, "height": 720}
        )
        await ctx.add_cookies(load_cookies_for_playwright(TWITTER_COOKIES_FILE))
        page = await ctx.new_page()

        log.info("Warming up browser session...")
        await page.goto("https://x.com/home", wait_until="load", timeout=60000)
        await page.wait_for_timeout(8000)
        title = await page.title()
        log.info(f"Browser ready: {title}")
        if "Home" not in title:
            log.error(f"Not logged in (title: {title}). Exiting.")
            await browser.close()
            return

        pbar = tqdm(total=target, initial=len(data), desc="Search", unit="tweet")
        since_save_ref = [0]

        for query in MALAY_QUERIES:
            if len(data) >= target:
                break
            log.info(f"Search: '{query}'")
            await scrape_search(page, query, data, seen, pbar, since_save_ref, target)
            await asyncio.sleep(3)

        pbar.close()
        save_json(data, path)
        log.info(f"Twitter search done: {len(data)} total new records")
        await browser.close()


if __name__ == "__main__":
    asyncio.run(collect_twitter_search(TARGET_TWITTER))
