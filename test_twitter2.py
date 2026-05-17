#!/usr/bin/env python3
"""Debug: check cookies are sent, inspect response headers."""
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
                if name in ("__cf_bm",):
                    continue
                cookies[name] = value
    return cookies

cookies = load_cookies()
ct0 = cookies.get("ct0", "")
auth_token = cookies.get("auth_token", "")
print(f"auth_token: {auth_token[:20]}...")
print(f"ct0: {ct0[:40]}...")

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

# Pass cookies directly (not via session)
print("\n=== Test 1: cookies as dict in request ===")
try:
    r = cf.get(
        "https://x.com/i/api/1.1/account/settings.json",
        cookies=cookies,
        headers=headers,
        impersonate="chrome120",
        timeout=20,
    )
    print(f"Status: {r.status_code}")
    print(f"Headers: {dict(list(r.headers.items())[:10])}")
    print(f"Body: {r.text[:300]}")
except Exception as e:
    print(f"Error: {e}")

time.sleep(2)

print("\n=== Test 2: adaptive.json response body / headers ===")
try:
    r = cf.get(
        "https://x.com/i/api/2/search/adaptive.json",
        cookies=cookies,
        headers=headers,
        params={"q": "kan", "count": 10, "result_type": "recent", "tweet_mode": "extended"},
        impersonate="chrome120",
        timeout=20,
    )
    print(f"Status: {r.status_code}")
    print(f"Content-Length: {r.headers.get('content-length', 'n/a')}")
    print(f"Content-Type: {r.headers.get('content-type', 'n/a')}")
    print(f"Body len: {len(r.content)} bytes")
    print(f"Body: {r.text[:500]}")
except Exception as e:
    print(f"Error: {e}")

time.sleep(2)

print("\n=== Test 3: v2/timeline/home.json ===")
try:
    r = cf.get(
        "https://x.com/i/api/2/timeline/home.json",
        cookies=cookies,
        headers=headers,
        params={"count": 5},
        impersonate="chrome120",
        timeout=20,
    )
    print(f"Status: {r.status_code}")
    print(f"Body: {r.text[:300]}")
except Exception as e:
    print(f"Error: {e}")

time.sleep(2)

print("\n=== Test 4: twid-based user lookup ===")
# twid = u%3D1508448552917381123 → user ID 1508448552917381123
try:
    r = cf.get(
        "https://x.com/i/api/1.1/users/show.json",
        cookies=cookies,
        headers=headers,
        params={"user_id": "1508448552917381123"},
        impersonate="chrome120",
        timeout=20,
    )
    print(f"Status: {r.status_code}")
    print(f"Body: {r.text[:300]}")
except Exception as e:
    print(f"Error: {e}")
