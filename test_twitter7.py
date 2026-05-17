#!/usr/bin/env python3
"""Test UserTweets with withVoice fix + parse Malay content."""
import json, re, time
from pathlib import Path
from curl_cffi import requests as cf

COOKIES_FILE = Path(__file__).parent / "x.com_cookies.txt"
_TW_BEARER = (
    "AAAAAAAAAAAAAAAAAAAAANRILgAAAAAAnNwIzUejRCOuH5E6I8xnZz4puTs"
    "%3D1Zv7ttfk8LF81IUq16cHjhLTvJu4FA33AGWWjCpTnA"
)
QUERY_IDS = {
    "UserTweets":          "_9v58axugmURcAmrOi7nxw",
    "UserTweetsAndReplies":"vIr6Sg_hjC3OOvIaPjiFZA",
    "UserByScreenName":    "pLsOiyHJ1eFwPJlNmLp4Bg",
}

_RAW = [
    r"\bkan+n*[\?\!\.\~,]*\b",
    r"(?m)(?:\bker+[\?\!\.\~,]*\b|\bka+[\?\!\.\~,]*\b|\bke\b(?!-\d)(?=\s*[?.!,😂😅🤣😭]*\s*$))",
    r"\b(eh+h*|ek+k*)[\?\!\.\~,]*\b",
    r"(?<!#)\b(ko|ku)t+t*[\?\!\.\~,]*\b",
    r"\bke\b(?!-\d)(?=\s+(?:lah|la|je|pun)\b)",
]
_PAT = [re.compile(p, re.IGNORECASE) for p in _RAW]
def is_malay(t): return any(p.search(t) for p in _PAT)

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
    "Referer":                   "https://x.com/",
    "Origin":                    "https://x.com",
    "Accept-Language":           "en-US,en;q=0.9",
    "Accept":                    "application/json, text/plain, */*",
}

tweet_feat = {
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

def get_user_id(screen_name):
    user_feat = {
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
    r = cf.get(
        f"https://x.com/i/api/graphql/{QUERY_IDS['UserByScreenName']}/UserByScreenName",
        cookies=cookies, headers=hdrs,
        params={"variables": json.dumps({"screen_name": screen_name, "withSafetyModeUserFields": True}, separators=(",", ":")),
                "features": json.dumps(user_feat, separators=(",", ":"))},
        impersonate="chrome120", timeout=20)
    if r.status_code == 200:
        js = r.json()
        return js.get("data", {}).get("user", {}).get("result", {}).get("rest_id", "")
    return ""

def parse_urt_tweets(js):
    """Parse TimelineUrt format (used by GraphQL UserTweets)."""
    tweets = []
    cursor = None
    try:
        instr = (js.get("data", {}).get("user", {}).get("result", {})
                   .get("timeline_v2", {}).get("timeline", {})
                   .get("instructions", []))
        for i in instr:
            if i.get("type") == "TimelineAddEntries":
                for entry in i.get("entries", []):
                    eid = entry.get("entryId", "")
                    content = entry.get("content", {})
                    if "tweet-" in eid:
                        item = content.get("itemContent", {})
                        result = item.get("tweet_results", {}).get("result", {})
                        legacy = result.get("legacy", {})
                        user_res = result.get("core", {}).get("user_results", {}).get("result", {})
                        user_leg = user_res.get("legacy", {})
                        full_text = legacy.get("full_text", legacy.get("text", ""))
                        if full_text:
                            tweets.append({
                                "id": result.get("rest_id", ""),
                                "text": full_text,
                                "author": user_leg.get("screen_name", ""),
                                "lang": legacy.get("lang", ""),
                                "created_at": legacy.get("created_at", ""),
                            })
                    elif "cursor-bottom" in eid:
                        val = content.get("value", "")
                        if val:
                            cursor = val
    except Exception as e:
        print(f"  Parse error: {e}")
    return tweets, cursor

print("=== Test UserTweets with withVoice fix ===")
user_id = "18040230"  # malaysiakini

variables = {
    "userId": user_id,
    "count": 20,
    "includePromotedContent": False,
    "withQuotedTweets": True,
    "withVoice": False,  # Required field
}
r = cf.get(
    f"https://x.com/i/api/graphql/{QUERY_IDS['UserTweets']}/UserTweets",
    cookies=cookies, headers=hdrs,
    params={"variables": json.dumps(variables, separators=(",", ":")),
            "features": json.dumps(tweet_feat, separators=(",", ":"))},
    impersonate="chrome120", timeout=30
)
print(f"Status: {r.status_code}, Size: {len(r.content)} bytes")
if r.status_code == 200 and r.content:
    js = r.json()
    tweets, cursor = parse_urt_tweets(js)
    print(f"Parsed {len(tweets)} tweets, cursor: {str(cursor)[:40] if cursor else 'None'}")
    malay_count = 0
    for tw in tweets:
        if is_malay(tw["text"]):
            malay_count += 1
            print(f"  [MALAY] {tw['text'][:100]}")
        else:
            print(f"  [other] {tw['text'][:80]}")
    print(f"\nMalay tweets: {malay_count}/{len(tweets)}")
elif r.content:
    print(f"Error: {r.text[:300]}")

time.sleep(2)
print("\n=== Test with cursor pagination (1 more page) ===")
if cursor:
    variables["cursor"] = cursor
    r2 = cf.get(
        f"https://x.com/i/api/graphql/{QUERY_IDS['UserTweets']}/UserTweets",
        cookies=cookies, headers=hdrs,
        params={"variables": json.dumps(variables, separators=(",", ":")),
                "features": json.dumps(tweet_feat, separators=(",", ":"))},
        impersonate="chrome120", timeout=30
    )
    print(f"Page 2 status: {r2.status_code}, Size: {len(r2.content)} bytes")
    if r2.status_code == 200 and r2.content:
        tweets2, c2 = parse_urt_tweets(r2.json())
        print(f"Parsed {len(tweets2)} more tweets, next cursor: {str(c2)[:40] if c2 else 'None'}")
        malay2 = sum(1 for t in tweets2 if is_malay(t["text"]))
        print(f"Malay: {malay2}/{len(tweets2)}")
