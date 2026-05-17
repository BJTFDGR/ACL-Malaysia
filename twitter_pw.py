#!/usr/bin/env python3
"""
Twitter Playwright scraper — uses page.wait_for_response() for reliable pagination.
Navigates to user profile pages and waits for UserTweets API responses explicitly.
"""
import asyncio, json, re, logging
from datetime import datetime, timezone
from pathlib import Path
from playwright.async_api import async_playwright, TimeoutError as PWTimeout
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

TARGET_TWITTER = 1000
OUTPUT_DIR     = Path(__file__).parent
TWITTER_COOKIES_FILE = OUTPUT_DIR / "x.com_cookies.txt"
SAVE_EVERY     = 10

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

def _parse_user_tweets(js):
    tweets, cursor = [], None
    try:
        instructions = (js.get("data",{}).get("user",{}).get("result",{})
                          .get("timeline",{}).get("timeline",{}).get("instructions",[]))
        for instr in instructions:
            if instr.get("type") != "TimelineAddEntries": continue
            for entry in instr.get("entries",[]):
                eid = entry.get("entryId","")
                content = entry.get("content",{})
                if "tweet-" in eid:
                    result = content.get("itemContent",{}).get("tweet_results",{}).get("result",{})
                    legacy = result.get("legacy",{})
                    user_leg = (result.get("core",{}).get("user_results",{})
                                      .get("result",{}).get("legacy",{}))
                    full_text = legacy.get("full_text") or legacy.get("text","")
                    if full_text:
                        tweets.append({"id": legacy.get("id_str",""), "text": full_text,
                                       "created_at": legacy.get("created_at",""),
                                       "screen_name": user_leg.get("screen_name",""),
                                       "retweet_count": legacy.get("retweet_count",0),
                                       "favorite_count": legacy.get("favorite_count",0)})
                elif "cursor-bottom" in eid:
                    val = content.get("value","")
                    if val: cursor = val
    except Exception as e:
        log.debug(f"parse error: {e}")
    return tweets, cursor

MALAY_ACCOUNTS = [
    # New accounts from annotators
    "localrkyt", "marchfoward", "syedewa", "rasyidinneedyou",
    "jatikhwan", "ketengahketepi", "zhafvlog", "alif_haidar", "azraeimuhamad",
    # Proven high-yield colloquial Malay accounts
    "NajwaLatif", "aizatfared", "zizan_razak", "AstroGempak", "AstroRia",
    "era_com_my", "ohbulandotcom", "CheMat_Official", "izzue_islam",
    "mynewshub", "AstroOOIB", "astroradio",
    "nabilahmajid", "harlizsofian", "faizalfazillah", "wanyamia",
    "nazrulzaman", "syahrulsamad", "didicazli", "bunkface_ariff",
    "mStar_Online", "BHarian_MY",
    "anwaribrahim", "DrWanAzizah", "budiey", "KhaledNordin", "AzminAli",
    "DrMahathir", "AbdulHadiAwang", "Ahmad_Zahid",
    "astroawani", "BHarian", "hmetro_online", "utusan_online",
    "rizalsulaiman", "farahmilaa", "shaheizy_sam",
    "matluthfi90", "sinaronline",
]


async def scrape_account(page, account, data, seen, pbar, since_save_ref, target):
    """Scrape one account using wait_for_response + scrolling."""
    path = OUTPUT_DIR / "twitter_data_new.json"

    def is_usertw(response):
        return "UserTweets" in response.url and "/graphql/" in response.url  # catch 200 and 429

    RATE_LIMIT_SLEEP = 780  # 13 minutes

    async def nav_and_get_first():
        """Navigate to profile, return first UserTweets response. Handles 429 with one retry."""
        for attempt in range(2):
            try:
                async with page.expect_response(is_usertw, timeout=60000) as resp_info:
                    await page.goto(f"https://x.com/{account}", wait_until="domcontentloaded", timeout=30000)
                response = await resp_info.value
                if response.status == 429:
                    log.warning(f"  @{account}: 429 rate limited — sleeping {RATE_LIMIT_SLEEP}s")
                    await asyncio.sleep(RATE_LIMIT_SLEEP)
                    continue
                return await response.body()
            except PWTimeout:
                log.warning(f"  @{account}: timeout waiting for UserTweets")
                return None
            except Exception as e:
                log.warning(f"  @{account}: navigation error: {type(e).__name__}: {e}")
                return None
        log.warning(f"  @{account}: still 429 after retry, skipping")
        return None

    # Navigate to profile
    body = await nav_and_get_first()
    if body is None:
        return
    try:
        first_js = json.loads(body)
    except Exception as e:
        log.warning(f"  @{account}: JSON parse error: {e}")
        return

    title = await page.title()
    if "Page not found" in title or title in ("X", ""):
        log.warning(f"  @{account}: not found")
        return

    pages_done = 0
    new_total = 0

    def process_js(js):
        nonlocal pages_done, new_total
        tweets, cursor = _parse_user_tweets(js)
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

    cursor = process_js(first_js)

    # Scroll loop
    MAX_SCROLLS = 60
    no_response_scrolls = 0

    for scroll_i in range(MAX_SCROLLS):
        if len(data) >= target: break
        if no_response_scrolls >= 6: break

        try:
            async with page.expect_response(is_usertw, timeout=15000) as resp_info:
                await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")

            response = await resp_info.value
            if response.status == 429:
                log.warning(f"  scroll {scroll_i}: 429 rate limited — sleeping {RATE_LIMIT_SLEEP}s")
                await asyncio.sleep(RATE_LIMIT_SLEEP)
                no_response_scrolls += 1
                continue
            body = await response.body()
            js = json.loads(body)
            cursor = process_js(js)
            no_response_scrolls = 0  # reset

        except PWTimeout:
            no_response_scrolls += 1
            await asyncio.sleep(1)

        except Exception as e:
            log.debug(f"  scroll {scroll_i}: {type(e).__name__}: {e}")
            no_response_scrolls += 1

    log.info(f"  @{account}: {pages_done} pages, {new_total} Malay tweets")


async def collect_twitter(target=TARGET_TWITTER):
    path     = OUTPUT_DIR / "twitter_data_new.json"
    old_path = OUTPUT_DIR / "twitter_data.json"
    data = load_json(path)           # resume new file if it exists
    seen = {r["id"] for r in load_json(old_path) if r.get("id")} | {r["id"] for r in data if r.get("id")}
    log.info(f"Twitter: {len(data)} new records so far (target {target})")

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

        pbar = tqdm(total=target, initial=len(data), desc="Twitter", unit="tweet")
        since_save_ref = [0]

        for account in MALAY_ACCOUNTS:
            if len(data) >= target:
                break
            log.info(f"Twitter: @{account}")
            await scrape_account(page, account, data, seen, pbar, since_save_ref, target)
            await asyncio.sleep(2)

        pbar.close()
        save_json(data[:target], path)
        log.info(f"Twitter done: {min(len(data), target)}/{target}")
        await browser.close()


if __name__ == "__main__":
    asyncio.run(collect_twitter(TARGET_TWITTER))
