#!/usr/bin/env python3
"""Diagnostic: find working Twitter/X search endpoint with current cookies."""
import json, re, time
from pathlib import Path
from curl_cffi import requests as cf

COOKIES_FILE = Path(__file__).parent / "x.com_cookies.txt"
_TW_BEARER = (
    "AAAAAAAAAAAAAAAAAAAAANRILgAAAAAAnNwIzUejRCOuH5E6I8xnZz4puTs"
    "%3D1Zv7ttfk8LF81IUq16cHjhLTvJu4FA33AGWWjCpTnA"
)

def load_cookies():
    """Load cookies, skip __cf_bm (short-lived Cloudflare cookie)."""
    cookies = {}
    with open(COOKIES_FILE) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) >= 7:
                name, value = parts[5], parts[6]
                if name in ("__cf_bm",):   # skip expired short-lived cookies
                    continue
                cookies[name] = value
    return cookies

def make_session(cookies):
    s = cf.Session(impersonate="chrome120")
    s.cookies.update(cookies)
    ct0 = cookies.get("ct0", "")
    s.headers.update({
        "Authorization":             f"Bearer {_TW_BEARER}",
        "x-csrf-token":              ct0,
        "x-twitter-active-user":     "yes",
        "x-twitter-auth-type":       "OAuth2Session",
        "x-twitter-client-language": "en",
        "Referer":                   "https://x.com/search",
        "Origin":                    "https://x.com",
        "Accept-Language":           "en-US,en;q=0.9",
        "Accept":                    "*/*",
    })
    return s

def get_query_id(s):
    """Try to fetch current SearchTimeline query ID from main JS bundle."""
    try:
        r = s.get("https://x.com/", timeout=15)
        # find main JS file reference
        m = re.search(r'"(https://abs.twimg.com/responsive-web/client-web/main\.[^"]+\.js)"', r.text)
        if not m:
            m = re.search(r'src="(/[^"]*main\.[^"]+\.js)"', r.text)
        if m:
            js_url = m.group(1)
            if js_url.startswith("/"):
                js_url = "https://x.com" + js_url
            print(f"  Found JS: {js_url}")
            r2 = s.get(js_url, timeout=30)
            # search for SearchTimeline query ID
            m2 = re.search(r'"([A-Za-z0-9_-]{22,})",operationName:"SearchTimeline"', r2.text)
            if m2:
                qid = m2.group(1)
                print(f"  SearchTimeline queryId: {qid}")
                return qid
            # Try other patterns
            m3 = re.search(r'queryId:"([A-Za-z0-9_-]{20,})"[^}]*operationName:"SearchTimeline"', r2.text)
            if m3:
                qid = m3.group(1)
                print(f"  SearchTimeline queryId (alt): {qid}")
                return qid
    except Exception as e:
        print(f"  JS fetch error: {e}")
    return None

def test_endpoint(s, url, params=None, label=""):
    try:
        r = s.get(url, params=params, timeout=15)
        body = r.text[:300].replace("\n", " ")
        print(f"  [{r.status_code}] {label}: {body}")
        return r.status_code, r.text
    except Exception as e:
        print(f"  [ERR] {label}: {e}")
        return 0, ""

def main():
    cookies = load_cookies()
    print(f"Loaded {len(cookies)} cookies (auth_token={'auth_token' in cookies}, ct0={'ct0' in cookies})")

    s = make_session(cookies)

    print("\n=== Warm-up: homepage ===")
    code, _ = test_endpoint(s, "https://x.com/", label="x.com homepage")
    time.sleep(2)

    print("\n=== Try to get current query ID ===")
    query_id = get_query_id(s)
    if not query_id:
        print("  Falling back to known query IDs")
        query_ids = ["CxeY8KoExnnnPXQdqG5Aaw", "nK1dw4oV3k4w5TdtcAdSww", "gkjsKepM6gl_HmFWoWKfgg"]
    else:
        query_ids = [query_id, "CxeY8KoExnnnPXQdqG5Aaw"]

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
        "rawQuery": "kan ke lang:ms",
        "count": 20,
        "querySource": "typed_query",
        "product": "Latest",
    }, separators=(",", ":"))

    print("\n=== GraphQL SearchTimeline (x.com/i/api) ===")
    for qid in query_ids:
        time.sleep(1)
        code, body = test_endpoint(
            s,
            f"https://x.com/i/api/graphql/{qid}/SearchTimeline",
            params={"variables": variables, "features": features},
            label=f"GraphQL/{qid[:12]}",
        )
        if code == 200 and len(body) > 50:
            print(f"  *** WORKING query ID: {qid} ***")
            print(f"  Body preview: {body[:500]}")
            break

    print("\n=== Legacy adaptive search ===")
    test_endpoint(s, "https://x.com/i/api/2/search/adaptive.json",
                  params={"q": "kan ke", "count": 20, "result_type": "recent",
                          "tweet_mode": "extended", "include_entities": 1},
                  label="adaptive.json")

    print("\n=== Verify auth (account/verify_credentials) ===")
    test_endpoint(s, "https://x.com/i/api/1.1/account/verify_credentials.json",
                  label="verify_credentials")

    print("\n=== Bearer-only guest (no cookies) ===")
    guest_s = cf.Session(impersonate="chrome120")
    guest_s.headers.update({
        "Authorization": f"Bearer {_TW_BEARER}",
        "x-twitter-client-language": "en",
    })
    test_endpoint(guest_s, "https://x.com/i/api/2/search/adaptive.json",
                  params={"q": "kan ke", "count": 5}, label="guest adaptive")

if __name__ == "__main__":
    main()
