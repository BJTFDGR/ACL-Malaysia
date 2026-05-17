#!/usr/bin/env python3
"""Test: GraphQL UserTweets for known Malay accounts + JS bundle fetch."""
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

def req(label, url, params=None):
    try:
        r = cf.get(url, cookies=cookies, headers=hdrs, params=params,
                   impersonate="chrome120", timeout=30)
        sz = len(r.content)
        print(f"[{r.status_code}] {label}: {sz} bytes")
        if sz > 0:
            print(f"  {r.text[:300]}")
        return r.status_code, r.text
    except Exception as e:
        print(f"[ERR] {label}: {e}")
        return 0, ""

# 1. Fetch abs.twimg.com JS without cookies (just bearer)
print("=== 1. JS bundle fetch (no cookies needed for static assets) ===")
# Try direct CDN URLs
for js_url in [
    "https://abs.twimg.com/responsive-web/client-web/main.a69f9b5a.js",
    "https://abs.twimg.com/responsive-web/client-web-legacy/main.a69f9b5a.js",
]:
    try:
        r = cf.get(js_url, impersonate="chrome120", timeout=30)
        print(f"  {js_url[-30:]}: {r.status_code}, {len(r.content)} bytes")
        if r.status_code == 200 and r.content:
            text = r.text
            for pat in [
                r'queryId:"([A-Za-z0-9_-]{20,})"[^}]{0,100}operationName:"SearchTimeline"',
                r'SearchTimeline[^}]{0,100}"([A-Za-z0-9_-]{20,})"',
                r'"([A-Za-z0-9_-]{20,})"\s*,\s*"SearchTimeline"',
            ]:
                m = re.search(pat, text)
                if m:
                    print(f"  *** SearchTimeline queryId: {m.group(1)} ***")
    except Exception as e:
        print(f"  Error: {e}")
    time.sleep(1)

# 2. Try fetching current JS bundle via x.com API
print("\n=== 2. x.com client-event (may reveal internal structure) ===")
# Actually let's fetch abs.twimg.com index to find current JS
try:
    r = cf.get("https://abs.twimg.com/responsive-web/client-web/",
               impersonate="chrome120", timeout=20)
    print(f"  abs.twimg.com listing: {r.status_code}")
except Exception as e:
    print(f"  Error: {e}")

# 3. GraphQL UserTweets with known query IDs
print("\n=== 3. GraphQL UserTweets for Malay accounts ===")
# User IDs for known Malay news accounts:
# MalaysiaNow: need to get user ID first
# Let's try the GraphQL UserByScreenName to get user IDs
user_features = json.dumps({
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
}, separators=(",", ":"))

# Try multiple known UserByScreenName query IDs
for qid in ["G3KGOASz96M-Ob0U9wnUZQ", "qRednkZG-rn1P6b48NINmQ", "CD7oBP3UHkEdWQbFDhLLhg"]:
    variables = json.dumps({"screen_name": "malaysiakini", "withSafetyModeUserFields": True}, separators=(",", ":"))
    code, body = req(f"UserByScreenName/{qid[:16]}", f"https://x.com/i/api/graphql/{qid}/UserByScreenName",
                    {"variables": variables, "features": user_features})
    if code == 200 and body:
        print(f"  *** Found working UserByScreenName queryId: {qid} ***")
        break
    time.sleep(1)

# 4. Try home timeline with extended/all params to get more content
print("\n=== 4. Home timeline variants ===")
req("home extended", "https://x.com/i/api/2/timeline/home.json",
    {"count": 100, "include_tweet_replies": 1, "include_want_retweets": 1})
time.sleep(1)

# 5. Try "For You" timeline (separate endpoint)
req("ForYou timeline", "https://x.com/i/api/2/timeline/home_latest.json",
    {"count": 100})
time.sleep(1)

# 6. Notifications timeline
req("notifications", "https://x.com/i/api/2/notifications/all.json", {"count": 20})
time.sleep(1)

# 7. Try v2 explore/trending endpoint
req("guide", "https://x.com/i/api/2/guide.json", {"count": 20})
