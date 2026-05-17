#!/usr/bin/env python3
"""Try to find working search: JS bundle query IDs, list timeline, user timeline."""
import json, re, time
from pathlib import Path
from curl_cffi import requests as cf

COOKIES_FILE = Path(__file__).parent / "x.com_cookies.txt"
_TW_BEARER = (
    "AAAAAAAAAAAAAAAAAAAAANRILgAAAAAAnNwIzUejRCOuH5E6I8xnZz4puTs"
    "%3D1Zv7ttfk8LF81IUq16cHjhLTvJu4FA33AGWWjCpTnA"
)

def load_cookies():
    cookies = {}
    with open(COOKIES_FILE) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) >= 7:
                name, value = parts[5], parts[6]
                if name == "__cf_bm":
                    continue
                cookies[name] = value
    return cookies

cookies = load_cookies()
ct0 = cookies.get("ct0", "")
hdrs = {
    "Authorization":             f"Bearer {_TW_BEARER}",
    "x-csrf-token":              ct0,
    "x-twitter-active-user":     "yes",
    "x-twitter-auth-type":       "OAuth2Session",
    "x-twitter-client-language": "en",
    "Referer":                   "https://x.com/search",
    "Origin":                    "https://x.com",
    "Accept-Language":           "en-US,en;q=0.9",
    "Accept":                    "application/json, text/plain, */*",
}

def req(label, url, params=None, extra_hdrs=None):
    h = {**hdrs, **(extra_hdrs or {})}
    try:
        r = cf.get(url, cookies=cookies, headers=h, params=params,
                   impersonate="chrome120", timeout=30)
        print(f"[{r.status_code}] {label}: {len(r.content)} bytes")
        if r.content and r.status_code == 200:
            print(f"  preview: {r.text[:300]}")
        return r.status_code, r.text
    except Exception as e:
        print(f"[ERR] {label}: {e}")
        return 0, ""

# 1. Scrape JS directly from abs.twimg.com to get current query IDs
print("=== 1. Fetch JS bundle from abs.twimg.com ===")
# Known JS bundle URL patterns
try:
    # Get x.com search page HTML (different path)
    r = cf.get("https://x.com/search?q=kan&src=typed_query&f=live",
               cookies=cookies, headers={**hdrs, "Accept": "text/html,*/*"},
               impersonate="chrome120", timeout=30)
    print(f"Search page HTML: {r.status_code}, {len(r.content)} bytes")
    # Look for JS bundle URL
    m = re.findall(r'https://abs\.twimg\.com/responsive-web/client-web/[^"\']+\.js', r.text)
    if m:
        print(f"  Found JS URLs: {m[:3]}")
        for js_url in m[:2]:
            time.sleep(1)
            r2 = cf.get(js_url, impersonate="chrome120", timeout=30)
            print(f"  JS {js_url[-30:]}: {r2.status_code}, {len(r2.content)} bytes")
            if r2.status_code == 200 and r2.content:
                # Search for query IDs
                text = r2.text
                patterns = [
                    r'queryId:"([A-Za-z0-9_-]{20,})"[^}]{0,100}operationName:"SearchTimeline"',
                    r'"([A-Za-z0-9_-]{20,})"\s*,\s*operationName\s*:\s*"SearchTimeline"',
                    r'SearchTimeline[^}]{0,100}queryId:"([A-Za-z0-9_-]{20,})"',
                    r'"([A-Za-z0-9_-]{20,})",\s*"SearchTimeline"',
                ]
                for pat in patterns:
                    m2 = re.search(pat, text)
                    if m2:
                        print(f"  *** SearchTimeline queryId: {m2.group(1)} ***")
                        break
    else:
        print(f"  No JS URLs found. HTML: {r.text[:500]}")
except Exception as e:
    print(f"Error: {e}")

time.sleep(2)

# 2. Try GraphQL with completely different features (minimal)
print("\n=== 2. GraphQL with minimal features ===")
for qid in ["CxeY8KoExnnnPXQdqG5Aaw", "nK1dw4oV3k4w5TdtcAdSww", "HMRC3Oz1X8_v5zvhWjpWIA"]:
    variables = json.dumps({"rawQuery": "kan", "count": 20, "querySource": "typed_query", "product": "Latest"}, separators=(",", ":"))
    features = json.dumps({"rweb_lists_timeline_redesign_enabled": True, "responsive_web_graphql_exclude_directive_enabled": True, "verified_phone_label_enabled": False, "creator_subscriptions_tweet_preview_api_enabled": True, "responsive_web_graphql_timeline_navigation_enabled": True, "responsive_web_graphql_skip_user_profile_image_extensions_enabled": False, "tweetypie_unmention_optimization_enabled": True, "responsive_web_edit_tweet_api_enabled": True, "graphql_is_translatable_rweb_tweet_is_translatable_enabled": True, "view_counts_everywhere_api_enabled": True, "longform_notetweets_consumption_enabled": True, "tweet_awards_web_tipping_enabled": False, "freedom_of_speech_not_reach_fetch_enabled": True, "standardized_nudges_misinfo": True, "tweet_with_visibility_results_prefer_gql_limited_actions_policy_enabled": True, "longform_notetweets_rich_text_read_enabled": True, "longform_notetweets_inline_media_enabled": True, "responsive_web_media_download_video_enabled": False, "responsive_web_enhance_cards_enabled": False}, separators=(",", ":"))
    req(f"GQL/{qid[:16]}", f"https://x.com/i/api/graphql/{qid}/SearchTimeline",
        {"variables": variables, "features": features})
    time.sleep(1)

# 3. User timeline — find a Malay tweeter and paginate their timeline
print("\n=== 3. User timeline (known Malay accounts) ===")
# Some Malaysian Twitter accounts with lots of Malay-language tweets
for screen_name in ["MalaysiaNow", "malaysiakini", "astroawani"]:
    req(f"user timeline @{screen_name}",
        "https://x.com/i/api/1.1/statuses/user_timeline.json",
        {"screen_name": screen_name, "count": 10, "tweet_mode": "extended"})
    time.sleep(1)

# 4. Try home timeline cursor pagination (get many tweets)
print("\n=== 4. Home timeline with cursor ===")
code, body = req("home timeline", "https://x.com/i/api/2/timeline/home.json",
                 {"count": 200})
if body:
    try:
        js = json.loads(body)
        tweets = js.get("globalObjects", {}).get("tweets", {})
        print(f"  Got {len(tweets)} tweets")
        # Look for next_cursor
        timeline = js.get("timeline", {})
        instructions = timeline.get("instructions", [])
        for instr in instructions:
            entries = instr.get("addEntries", {}).get("entries", [])
            for e in entries:
                if "cursor-top" in e.get("entryId", "") or "cursor-bottom" in e.get("entryId", ""):
                    print(f"  Cursor: {e['entryId']} = {e.get('content',{}).get('operation',{}).get('cursor',{}).get('value','')[:40]}")
    except Exception as e:
        print(f"  Parse error: {e}")
