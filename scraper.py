#!/usr/bin/env python3
"""
Malaysian Language Data Collector — cookie-authenticated
=========================================================
Reddit  : OAuth search API (token_v2 Bearer, 100 req/10-min window)
Twitter : GraphQL UserTweets (curl_cffi chrome impersonation)
           - SearchTimeline blocked; UserTweets confirmed working Feb 2026
           - Uses a curated list of high-volume Malay-language accounts

Anti-scrape libs:
  - curl_cffi      github.com/yifeikong/curl_cffi       (TLS fingerprint impersonation)
  - fake-useragent github.com/fake-useragent/fake-useragent  (UA rotation)
  - tenacity       github.com/jd/tenacity                    (retry + backoff)
"""

import json
import re
import time
import random
import logging
from datetime import datetime, timezone
from pathlib import Path

import requests
from curl_cffi import requests as cf
from fake_useragent import UserAgent
from tenacity import (
    retry, stop_after_attempt, wait_exponential,
    retry_if_exception_type, before_sleep_log,
)
from tqdm import tqdm

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
TARGET_REDDIT  = 3000
TARGET_TWITTER = 3000
OUTPUT_DIR     = Path(__file__).parent
SAVE_EVERY     = 50
REDDIT_COOKIES_FILE  = OUTPUT_DIR / "www.reddit.com_cookies.txt"
TWITTER_COOKIES_FILE = OUTPUT_DIR / "x.com_cookies.txt"

# ── Whitelist patterns ────────────────────────────────────────────────────────
_RAW = [
    r"\bkan+n*[\?\!\.\~,]*\b",
    r"(?m)(?:\bker+[\?\!\.\~,]*\b|\bka+[\?\!\.\~,]*\b|\bke\b(?!-\d)(?=\s*[?.!,😂😅🤣😭]*\s*$))",
    r"\b(eh+h*|ek+k*)[\?\!\.\~,]*\b",
    r"(?<!#)\b(ko|ku)t+t*[\?\!\.\~,]*\b",
    r"\bke\b(?!-\d)(?=\s+(?:lah|la|je|pun)\b)",
]
_PAT   = [re.compile(p, re.IGNORECASE) for p in _RAW]
_NAMES = ["kan", "ke/ka", "eh/ek", "kot/kut", "ke_clausal"]

def matched(text: str) -> list[str]:
    if not text:
        return []
    return [n for p, n in zip(_PAT, _NAMES) if p.search(text)]

def is_malay(text: str) -> bool:
    return bool(matched(text))

def make_record(text: str, source: str, **meta) -> dict:
    return {
        "text": text.strip(),
        "source": source,
        "matched_patterns": matched(text),
        "collected_at": datetime.now(timezone.utc).isoformat(),
        **meta,
    }

# ── Persistence ───────────────────────────────────────────────────────────────
def load_json(path: Path) -> list[dict]:
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return []

def save_json(data: list[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# ── Cookie loading ────────────────────────────────────────────────────────────
def load_netscape_cookies(path: Path, skip: tuple = ()) -> dict[str, str]:
    cookies: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) >= 7 and parts[5] not in skip:
                cookies[parts[5]] = parts[6]
    return cookies

_ua = UserAgent()

# ─────────────────────────────────────────────────────────────────────────────
# REDDIT — search-targeted, OAuth bearer
# ─────────────────────────────────────────────────────────────────────────────

SUBREDDITS = [
    "malaysia", "malaysians", "bolehland", "KualaLumpur", "learnmalay",
    "melayu", "MalaysiaPolitics", "AskMalaysia", "malaysianfood", "mildlyinfuriating_my",
]

REDDIT_SEARCH_TERMS = [
    "kan", "kann", "kannn",
    "ke", "ker", "ka",
    "eh", "ehh", "ek",
    "kot", "kott", "kut",
    "lah kan", "betul kan", "tau kan",
    "eh betul", "eh kan", "kan je",
    "eh kot", "kot la", "boleh ke",
    "nak ke", "macam mana eh",
]

def make_reddit_session() -> requests.Session:
    cookies = load_netscape_cookies(REDDIT_COOKIES_FILE)
    token   = cookies.get("token_v2", "")
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

@retry(
    stop=stop_after_attempt(6),
    wait=wait_exponential(multiplier=1, min=3, max=90),
    retry=retry_if_exception_type((requests.RequestException, ValueError)),
    before_sleep=before_sleep_log(log, logging.WARNING),
)
def _reddit_get(session: requests.Session, url: str, params: dict) -> dict:
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

def collect_reddit(target: int = TARGET_REDDIT) -> list[dict]:
    path  = OUTPUT_DIR / "reddit_data.json"
    data  = load_json(path)
    seen  = {r["id"] for r in data}
    log.info(f"Reddit: resuming from {len(data)} records")

    session    = make_reddit_session()
    pbar       = tqdm(total=target, initial=len(data), desc="Reddit ", unit="rec")
    since_save = 0

    def checkpoint():
        nonlocal since_save
        save_json(data, path)
        since_save = 0
        log.info(f"  ckpt → {len(data)} Reddit records saved")

    for sub in SUBREDDITS:
        if len(data) >= target:
            break

        for term in REDDIT_SEARCH_TERMS:
            if len(data) >= target:
                break

            log.info(f"r/{sub} search: '{term}'")
            after       = None
            empty_pages = 0

            while len(data) < target:
                params = {
                    "q":           term,
                    "sort":        "new",
                    "type":        "link",
                    "limit":       100,
                    "raw_json":    1,
                    "restrict_sr": 1,
                }
                if after:
                    params["after"] = after

                try:
                    js = _reddit_get(
                        session,
                        f"https://oauth.reddit.com/r/{sub}/search.json",
                        params,
                    )
                except Exception as e:
                    log.warning(f"r/{sub} search '{term}': {e}")
                    break

                children = js.get("data", {}).get("children", [])
                after    = js.get("data", {}).get("after")

                if not children:
                    break

                new_posts = 0
                for child in children:
                    if len(data) >= target:
                        break
                    d   = child.get("data", {})
                    pid = d.get("id", "")
                    if not pid or f"p_{pid}" in seen:
                        continue
                    seen.add(f"p_{pid}")

                    title    = d.get("title", "")
                    selftext = d.get("selftext", "")
                    combined = f"{title} {selftext}".strip()

                    if is_malay(combined):
                        data.append(make_record(
                            combined, "reddit",
                            id=f"p_{pid}", type="post",
                            subreddit=d.get("subreddit", sub),
                            title=title,
                            url=f"https://reddit.com{d.get('permalink','')}",
                            author=d.get("author", "[deleted]"),
                            score=d.get("score", 0),
                            num_comments=d.get("num_comments", 0),
                            created_utc=datetime.fromtimestamp(
                                d.get("created_utc", 0), tz=timezone.utc
                            ).isoformat(),
                        ))
                        pbar.update(1)
                        since_save += 1
                        new_posts  += 1

                    if since_save >= SAVE_EVERY:
                        checkpoint()

                    time.sleep(random.uniform(0.05, 0.15))

                empty_pages = 0 if new_posts else empty_pages + 1
                if empty_pages >= 3 or not after:
                    break

                session = make_reddit_session()
                time.sleep(random.uniform(6.0, 8.0))

    pbar.close()
    save_json(data[:target], path)
    log.info(f"Reddit done: {min(len(data), target)}")
    return data[:target]


# ─────────────────────────────────────────────────────────────────────────────
# TWITTER/X — GraphQL UserTweets (cookie-authenticated, curl_cffi)
# ─────────────────────────────────────────────────────────────────────────────
# SearchTimeline is blocked (404) for this credential level as of Feb 2026.
# UserTweets works and allows paginating any public account's timeline.
# Strategy: paginate through many high-volume Malay-language accounts.
#
# "ke" (preposition "to") appears in virtually every Malay sentence,
# so hit rate for Malay-language accounts is ~90%+.

_TW_BEARER = (
    "AAAAAAAAAAAAAAAAAAAAANRILgAAAAAAnNwIzUejRCOuH5E6I8xnZz4puTs"
    "%3D1Zv7ttfk8LF81IUq16cHjhLTvJu4FA33AGWWjCpTnA"
)

# GraphQL query IDs from main.a69f9b5a.js (extracted Feb 2026)
_GQL = {
    "UserByScreenName": "pLsOiyHJ1eFwPJlNmLp4Bg",
    "UserTweets":        "_9v58axugmURcAmrOi7nxw",
}

# Tweet-level features required by UserTweets
_TWEET_FEAT = {
    "profile_label_improvements_pcf_label_in_post_enabled": False,
    "rweb_tipjar_consumption_enabled": True,
    "responsive_web_graphql_exclude_directive_enabled": True,
    "verified_phone_label_enabled": False,
    "creator_subscriptions_tweet_preview_api_enabled": True,
    "responsive_web_graphql_timeline_navigation_enabled": True,
    "responsive_web_graphql_skip_user_profile_image_extensions_enabled": False,
    "premium_content_api_read_enabled": False,
    "communities_web_enable_tweet_community_results_fetch": True,
    "c9s_tweet_anatomy_moderator_badge_enabled": True,
    "responsive_web_grok_analyze_button_fetch_trends_enabled": False,
    "responsive_web_grok_analyze_post_followups_enabled": True,
    "responsive_web_jetfuel_frame": False,
    "responsive_web_grok_share_attachment_enabled": True,
    "articles_preview_enabled": True,
    "responsive_web_edit_tweet_api_enabled": True,
    "graphql_is_translatable_rweb_tweet_is_translatable_enabled": True,
    "view_counts_everywhere_api_enabled": True,
    "longform_notetweets_consumption_enabled": True,
    "responsive_web_twitter_article_tweet_consumption_enabled": True,
    "tweet_awards_web_tipping_enabled": False,
    "creator_subscriptions_quote_tweet_preview_enabled": False,
    "freedom_of_speech_not_reach_fetch_enabled": True,
    "standardized_nudges_misinfo": True,
    "tweet_with_visibility_results_prefer_gql_limited_actions_policy_enabled": True,
    "rweb_video_screen_enabled": False,
    "longform_notetweets_rich_text_read_enabled": True,
    "longform_notetweets_inline_media_enabled": True,
    "responsive_web_media_download_video_enabled": False,
    "responsive_web_enhance_cards_enabled": False,
}

_USER_FEAT = {
    "hidden_profile_likes_enabled": False,
    "hidden_profile_subscriptions_enabled": False,
    "responsive_web_graphql_exclude_directive_enabled": True,
    "verified_phone_label_enabled": False,
    "subscriptions_verification_info_is_identity_verified_enabled": False,
    "subscriptions_verification_info_verified_since_enabled": True,
    "highlights_tweets_tab_ui_enabled": True,
    "creator_subscriptions_tweet_preview_api_enabled": True,
    "responsive_web_graphql_skip_user_profile_image_extensions_enabled": False,
    "responsive_web_graphql_timeline_navigation_enabled": True,
}

# Curated high-volume Malay-language Twitter accounts (verified working, Feb 2026).
# All UIDs confirmed via UserByScreenName lookup.
# Hit rates measured empirically: sinaronline 10%, RTM 25%, TV3 19%, AstroArena 5%.
# "ke" (preposition "to") + "kan" appear frequently in Malay sentences.
MALAY_ACCOUNTS = [
    # === Batch 6 — colloquial youth/humor/radio/football ===
    "ohbulandotcom",    # Oh! Bulan — Malaysian lifestyle/entertainment (resolved ✓)
    "harith_iskander",  # Harith Iskander — Malaysian comedian (resolved ✓)
    "JinnyBoyTV",       # Jinny Boy — Malaysian YouTuber/comedian (resolved ✓)
    "RakyatPost",       # Rakyat Post — casual writing style (resolved ✓)
    "DagangNews",       # Dagang News — Malay business (resolved ✓)
    "says_my",          # Says.com MY — casual writing (resolved ✓)
    "lowyat_net",       # Lowyat.net — Malaysian tech/general (resolved ✓)
    "EraFM",            # Era FM radio — colloquial Malay ("Korang rasa?")
    "GempakDotCom",     # Gempak entertainment
    "HarianMetro",      # Harian Metro — Malay tabloid, casual tone
    "AimanAzlan",       # Aiman Azlan — motivational speaker, casual Malay
    "UltrasmalayaFC",   # Ultras Malaya — football fans, very colloquial
    "JDT_Official",     # Johor Darul Ta'zim FC — football, Malay fans
    "FAMalaysia",       # FA Malaysia — football, mixed Malay/English
    "klfa_my",          # KL FA football
    "MysuperleagueMY",  # Super League — Malaysian football
    "AstroSuper",       # Astro Super sports, tweets in Malay
    "NajibRazak",       # Najib Razak — former PM, tweets casually in Malay
    "DrMahathir",       # Dr Mahathir — tweets in Malay
    "anwaribrahim",     # Anwar Ibrahim — current PM, Malay tweets
    "KhaledNordin",     # Khaled Nordin — minister, Malay tweets
    "matluthfi",        # Mat Luthfi — alternate handle
    "zizan",            # Zizan Razak comedian alternate handle
    "NajwaLatif",       # Najwa Latif singer — tweets casually in Malay
    "AstroGempak",      # Astro Gempak youth channel
    # === Confirmed high-volume Malay accounts (already scraped, will skip seen IDs) ===
    "sinaronline",      # Sinar Harian — 931K tweets, ~10% hit rate
    "RTM_Malaysia",     # RTM national broadcaster — 332K tweets, ~25% hit rate
    "TV3Malaysia",      # TV3 — 99K tweets, ~19% hit rate
    "KKMPutrajaya",     # Ministry of Health — 63K tweets, Malay
    "AstroArena",       # Sports broadcaster — 169K tweets, ~5% hit rate
    "kosmo_online",     # Kosmo! — 322K tweets
    "MalaysiaGazette",  # Malaysia Gazette — 484K tweets
    "TVAlhijrah",       # Al-Hijrah TV — 26K tweets, Malay Islamic
    "remaja_my",        # Remaja magazine — 21K tweets, Malay youth ★ 65 records
    "MediaPermata",     # Media Permata — 28K tweets
    "tourismmalaysia",  # Tourism Malaysia — 13K tweets, Malay/English
    "malaymail",        # Malay Mail — 395K tweets, bilingual
    "TheSunDaily",      # The Sun — 319K tweets, bilingual
    "unifi",            # Unifi/TM — 51K tweets, Malay/English mix
    "Maybank2u",        # Maybank — 4K tweets
    "matluthfi90",      # Mat Luthfi — popular Malaysian creator, 796 tweets
    "malaysiakini",     # Malaysiakini — 293K tweets, bilingual
]


def make_tw_session() -> cf.Session:
    """Build curl_cffi session with Twitter cookies (skip expired __cf_bm)."""
    cookies = load_netscape_cookies(TWITTER_COOKIES_FILE, skip=("__cf_bm",))
    ct0 = cookies.get("ct0", "")
    s = cf.Session(impersonate="chrome120")
    s.cookies.update(cookies)
    s.headers.update({
        "Authorization":             f"Bearer {_TW_BEARER}",
        "x-csrf-token":              ct0,
        "x-twitter-active-user":     "yes",
        "x-twitter-auth-type":       "OAuth2Session",
        "x-twitter-client-language": "en",
        "Referer":                   "https://x.com/",
        "Origin":                    "https://x.com",
        "Accept-Language":           "en-US,en;q=0.9",
        "Accept":                    "application/json, text/plain, */*",
    })
    return s


def _tw_gql(session: cf.Session, op: str, variables: dict) -> tuple[int, dict]:
    """Make a GraphQL request; return (status_code, json_body)."""
    feat = _TWEET_FEAT if op == "UserTweets" else _USER_FEAT
    try:
        r = session.get(
            f"https://x.com/i/api/graphql/{_GQL[op]}/{op}",
            params={
                "variables": json.dumps(variables, separators=(",", ":")),
                "features":  json.dumps(feat,      separators=(",", ":")),
            },
            timeout=30,
        )
        if r.status_code == 429:
            reset = int(r.headers.get("x-rate-limit-reset", time.time() + 60))
            wait  = max(reset - int(time.time()), 10)
            log.warning(f"Twitter 429 — sleeping {wait}s")
            time.sleep(wait)
            return 429, {}
        if r.content:
            return r.status_code, r.json()
        return r.status_code, {}
    except Exception as e:
        log.debug(f"_tw_gql {op}: {e}")
        return 0, {}


def _get_twitter_user_id(session: cf.Session, screen_name: str) -> str:
    """Resolve @screen_name → REST user ID string."""
    code, js = _tw_gql(session, "UserByScreenName",
                        {"screen_name": screen_name, "withSafetyModeUserFields": True})
    if code == 200:
        return (js.get("data", {})
                  .get("user", {})
                  .get("result", {})
                  .get("rest_id", ""))
    return ""


def _parse_user_tweets(js: dict) -> tuple[list[dict], str | None]:
    """
    Walk UserTweets GraphQL response.
    Path: data.user.result.timeline.timeline.instructions
    Returns (tweet_list, next_cursor).
    """
    tweets: list[dict] = []
    cursor: str | None = None

    try:
        instructions = (
            js.get("data", {})
              .get("user", {})
              .get("result", {})
              .get("timeline", {})
              .get("timeline", {})
              .get("instructions", [])
        )
        for instr in instructions:
            if instr.get("type") != "TimelineAddEntries":
                continue
            for entry in instr.get("entries", []):
                eid     = entry.get("entryId", "")
                content = entry.get("content", {})

                if "tweet-" in eid:
                    result  = (content.get("itemContent", {})
                                      .get("tweet_results", {})
                                      .get("result", {}))
                    legacy  = result.get("legacy", {})
                    user_leg = (result.get("core", {})
                                      .get("user_results", {})
                                      .get("result", {})
                                      .get("legacy", {}))
                    full_text = legacy.get("full_text") or legacy.get("text", "")
                    if full_text:
                        tweets.append({
                            "id":            result.get("rest_id", ""),
                            "text":          full_text,
                            "author":        user_leg.get("screen_name", ""),
                            "display_name":  user_leg.get("name", ""),
                            "created_at":    legacy.get("created_at", ""),
                            "lang":          legacy.get("lang", ""),
                            "retweet_count": legacy.get("retweet_count", 0),
                            "favorite_count": legacy.get("favorite_count", 0),
                        })

                elif "cursor-bottom" in eid:
                    val = content.get("value", "")
                    if val:
                        cursor = val
    except Exception as e:
        log.debug(f"_parse_user_tweets: {e}")

    return tweets, cursor


def collect_twitter(target: int = TARGET_TWITTER) -> list[dict]:
    path  = OUTPUT_DIR / "twitter_data.json"
    data  = load_json(path)
    seen  = {r["id"] for r in data}
    log.info(f"Twitter: resuming from {len(data)} records")

    session    = make_tw_session()
    pbar       = tqdm(total=target, initial=len(data), desc="Twitter", unit="tweet")
    since_save = 0

    def checkpoint():
        nonlocal since_save
        save_json(data, path)
        since_save = 0
        log.info(f"  ckpt → {len(data)} Twitter records saved")

    for screen_name in MALAY_ACCOUNTS:
        if len(data) >= target:
            break

        log.info(f"Twitter: @{screen_name}")

        # Resolve user ID
        user_id = _get_twitter_user_id(session, screen_name)
        if not user_id:
            log.warning(f"  @{screen_name}: could not resolve user ID — skipping")
            time.sleep(random.uniform(1.0, 2.0))
            continue

        log.info(f"  @{screen_name} → {user_id}")
        cursor     = None
        empty_pages = 0
        page_count  = 0

        while len(data) < target:
            variables: dict = {
                "userId":              user_id,
                "count":               100,
                "includePromotedContent": False,
                "withQuotedTweets":    True,
                "withVoice":           False,
            }
            if cursor:
                variables["cursor"] = cursor

            code, js = _tw_gql(session, "UserTweets", variables)

            if code == 429:
                time.sleep(60)
                continue
            if code != 200 or not js:
                log.warning(f"  @{screen_name} page {page_count}: {code}")
                break

            tweets, next_cursor = _parse_user_tweets(js)
            page_count += 1

            if not tweets:
                empty_pages += 1
                if empty_pages >= 3:
                    break
                cursor = next_cursor
                time.sleep(random.uniform(1.0, 2.0))
                continue

            new_this = 0
            for tw in tweets:
                tid  = tw.get("id", "")
                text = tw.get("text", "")
                if not tid or tid in seen or not text:
                    continue
                seen.add(tid)
                if not is_malay(text):
                    continue
                data.append(make_record(
                    text, "twitter",
                    id=tid,
                    author=tw.get("author", screen_name),
                    display_name=tw.get("display_name", ""),
                    created_at=tw.get("created_at", ""),
                    lang=tw.get("lang", ""),
                    retweet_count=tw.get("retweet_count", 0),
                    favorite_count=tw.get("favorite_count", 0),
                    url=f"https://x.com/{tw.get('author', screen_name)}/status/{tid}",
                    account_scraped=screen_name,
                ))
                pbar.update(1)
                since_save += 1
                new_this   += 1
                if len(data) >= target:
                    break

            if since_save >= SAVE_EVERY:
                checkpoint()

            cursor = next_cursor
            empty_pages = 0 if new_this else empty_pages + 1

            # Stop this account if no more pages or no new content for 3 pages
            if not cursor or empty_pages >= 3:
                break

            time.sleep(random.uniform(1.0, 2.5))

        log.info(f"  @{screen_name}: {page_count} pages scraped")
        time.sleep(random.uniform(2.0, 4.0))

    pbar.close()
    save_json(data[:target], path)
    log.info(f"Twitter done: {min(len(data), target)}")
    return data[:target]


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    log.info("=" * 60)
    log.info("  Malaysian Language Collector (UserTweets GraphQL + Reddit OAuth)")
    log.info("=" * 60)

    reddit_data  = collect_reddit(TARGET_REDDIT)
    twitter_data = collect_twitter(TARGET_TWITTER)

    combined      = reddit_data + twitter_data
    combined_path = OUTPUT_DIR / "combined_data.json"
    save_json(combined, combined_path)

    log.info("=" * 60)
    log.info(f"Reddit   : {len(reddit_data):>5} → reddit_data.json")
    log.info(f"Twitter  : {len(twitter_data):>5} → twitter_data.json")
    log.info(f"Combined : {len(combined):>5} → combined_data.json")
    log.info("Pattern breakdown (combined):")
    for name in _NAMES:
        cnt = sum(1 for r in combined if name in r.get("matched_patterns", []))
        log.info(f"  {name:<8}: {cnt:>5}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
