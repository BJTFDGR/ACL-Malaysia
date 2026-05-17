#!/usr/bin/env python3
"""Test: UserTweets GraphQL + HomeTimeline GraphQL + SearchTimeline features."""
import json, re, time
from pathlib import Path
from curl_cffi import requests as cf

COOKIES_FILE = Path(__file__).parent / "x.com_cookies.txt"
_TW_BEARER = (
    "AAAAAAAAAAAAAAAAAAAAANRILgAAAAAAnNwIzUejRCOuH5E6I8xnZz4puTs"
    "%3D1Zv7ttfk8LF81IUq16cHjhLTvJu4FA33AGWWjCpTnA"
)
# Confirmed from main.a69f9b5a.js (Feb 2026)
QUERY_IDS = {
    "HomeTimeline":        "TJxUI58LjqXUqrtpwRSK_A",
    "SearchTimeline":      "CxeY8KoExnnnPXQdqG5Aaw",
    "UserTweets":          "_9v58axugmURcAmrOi7nxw",
    "UserTweetsAndReplies":"vIr6Sg_hjC3OOvIaPjiFZA",
    "UserByScreenName":    "pLsOiyHJ1eFwPJlNmLp4Bg",
    "TweetDetail":         "7U1X7-LeNUX-OYmIndrSiw",
}

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

def gql(op, variables, features):
    qid = QUERY_IDS[op]
    params = {"variables": json.dumps(variables, separators=(",", ":")),
              "features": json.dumps(features, separators=(",", ":"))}
    try:
        r = cf.get(f"https://x.com/i/api/graphql/{qid}/{op}",
                   cookies=cookies, headers=hdrs, params=params,
                   impersonate="chrome120", timeout=30)
        sz = len(r.content)
        print(f"[{r.status_code}] {op}: {sz} bytes")
        if sz > 0 and r.status_code == 200:
            print(f"  preview: {r.text[:400]}")
        elif sz > 0:
            print(f"  error: {r.text[:200]}")
        return r.status_code, r.text
    except Exception as e:
        print(f"[ERR] {op}: {e}")
        return 0, ""

# User features (used for UserByScreenName)
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

# Tweet features
tweet_feat = {
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
}

print("=== 1. UserByScreenName (current query ID from JS) ===")
code, body = gql("UserByScreenName",
    {"screen_name": "malaysiakini", "withSafetyModeUserFields": True},
    user_feat)
user_id = None
if code == 200 and body:
    try:
        js = json.loads(body)
        user_id = js["data"]["user"]["result"]["rest_id"]
        print(f"  *** malaysiakini user ID: {user_id} ***")
    except Exception as e:
        print(f"  parse error: {e}")
time.sleep(2)

print("\n=== 2. UserTweets ===")
if user_id:
    gql("UserTweets",
        {"userId": user_id, "count": 20, "includePromotedContent": False,
         "withQuotedTweets": True},
        tweet_feat)
else:
    # Try with a hardcoded user ID (Malaysiakini: ~18040230)
    gql("UserTweets",
        {"userId": "18040230", "count": 20, "includePromotedContent": False,
         "withQuotedTweets": True},
        tweet_feat)
time.sleep(2)

print("\n=== 3. SearchTimeline with user_feat (not tweet_feat) ===")
gql("SearchTimeline",
    {"rawQuery": "kan lang:ms", "count": 20, "querySource": "typed_query", "product": "Latest"},
    user_feat)
time.sleep(1)

print("\n=== 4. SearchTimeline with empty features ===")
gql("SearchTimeline",
    {"rawQuery": "kan lang:ms", "count": 20, "querySource": "typed_query", "product": "Latest"},
    {})
time.sleep(1)

print("\n=== 5. HomeTimeline GraphQL ===")
gql("HomeTimeline",
    {"count": 20, "includePromotedContent": True, "latestControlAvailable": True,
     "requestContext": "launch", "withCommunity": False, "seenTweetIds": []},
    tweet_feat)
time.sleep(1)

print("\n=== 6. SearchTimeline with ALL features from JS ===")
# Try with a comprehensive feature set
full_feat = {
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
    "responsive_web_graphql_exclude_directive_enabled": True,
    "verified_phone_label_enabled": False,
    "responsive_web_media_download_video_enabled": False,
    "responsive_web_enhance_cards_enabled": False,
}
gql("SearchTimeline",
    {"rawQuery": "kan lang:ms", "count": 20, "querySource": "typed_query", "product": "Latest"},
    full_feat)
