#!/usr/bin/env python3
"""Test search endpoints that might work."""
import json, time
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
headers = {
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

def test(label, url, params=None):
    try:
        r = cf.get(url, cookies=cookies, headers=headers, params=params,
                   impersonate="chrome120", timeout=20)
        body_len = len(r.content)
        print(f"[{r.status_code}] {label}: {body_len} bytes | {r.text[:200]}")
        return r.status_code, r.text
    except Exception as e:
        print(f"[ERR] {label}: {e}")
        return 0, ""

print("=== Search variants ===")

# Variant 1: minimal params
test("adaptive v2 minimal", "https://x.com/i/api/2/search/adaptive.json",
     {"q": "kan", "count": 10})
time.sleep(1)

# Variant 2: full params
test("adaptive v2 full", "https://x.com/i/api/2/search/adaptive.json",
     {"q": "kan", "count": 10, "result_type": "recent",
      "tweet_mode": "extended", "include_entities": 1,
      "include_user_entities": 1, "include_cards": 1, "cards_platform": "Web-13",
      "simple_quoted_tweet": True})
time.sleep(1)

# Variant 3: v1.1 on x.com
test("v1.1 search on x.com", "https://x.com/i/api/1.1/search/tweets.json",
     {"q": "kan", "count": 10, "result_type": "recent", "tweet_mode": "extended"})
time.sleep(1)

# Variant 4: try GraphQL with x.com domain (not twitter.com)
features = json.dumps({
    "rweb_lists_timeline_redesign_enabled": True,
    "responsive_web_graphql_exclude_directive_enabled": True,
    "verified_phone_label_enabled": False,
    "creator_subscriptions_tweet_preview_api_enabled": True,
    "responsive_web_graphql_timeline_navigation_enabled": True,
    "responsive_web_graphql_skip_user_profile_image_extensions_enabled": False,
    "tweetypie_unmention_optimization_enabled": True,
    "responsive_web_edit_tweet_api_enabled": True,
    "graphql_is_translatable_rweb_tweet_is_translatable_enabled": True,
    "view_counts_everywhere_api_enabled": True,
    "longform_notetweets_consumption_enabled": True,
    "tweet_awards_web_tipping_enabled": False,
    "freedom_of_speech_not_reach_fetch_enabled": True,
    "standardized_nudges_misinfo": True,
    "tweet_with_visibility_results_prefer_gql_limited_actions_policy_enabled": True,
    "longform_notetweets_rich_text_read_enabled": True,
    "longform_notetweets_inline_media_enabled": True,
    "responsive_web_media_download_video_enabled": False,
    "responsive_web_enhance_cards_enabled": False,
}, separators=(",", ":"))
variables = json.dumps({
    "rawQuery": "kan lang:ms",
    "count": 20,
    "querySource": "typed_query",
    "product": "Latest",
}, separators=(",", ":"))

for qid in ["CxeY8KoExnnnPXQdqG5Aaw", "nK1dw4oV3k4w5TdtcAdSww"]:
    test(f"GraphQL x.com/{qid[:12]}", f"https://x.com/i/api/graphql/{qid}/SearchTimeline",
         {"variables": variables, "features": features})
    time.sleep(1)

print("\n=== Working: home timeline (for reference) ===")
code, body = test("home timeline", "https://x.com/i/api/2/timeline/home.json",
                  {"count": 5})
if body:
    js = json.loads(body)
    tweets = js.get("globalObjects", {}).get("tweets", {})
    print(f"  → {len(tweets)} tweets in response")
    for tid, tw in list(tweets.items())[:2]:
        print(f"  tweet: {tw.get('full_text', tw.get('text',''))[:100]}")

time.sleep(1)
print("\n=== v2/timeline/search.json ===")
test("timeline/search.json", "https://x.com/i/api/2/timeline/search.json",
     {"q": "kan", "count": 10, "result_type": "recent",
      "tweet_mode": "extended", "include_entities": 1})
