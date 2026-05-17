#!/usr/bin/env python3
import json
from curl_cffi import requests as cf

cookies = {}
with open('/home/xitongzhang/Maylie/x.com_cookies.txt') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) >= 7 and parts[5] not in ('__cf_bm',):
            cookies[parts[5]] = parts[6]

ct0 = cookies.get('ct0', '')
_TW_BEARER = 'AAAAAAAAAAAAAAAAAAAAANRILgAAAAAAnNwIzUejRCOuH5E6I8xnZz4puTs%3D1Zv7ttfk8LF81IUq16cHjhLTvJu4FA33AGWWjCpTnA'
hdrs = {
    'Authorization': f'Bearer {_TW_BEARER}', 'x-csrf-token': ct0,
    'x-twitter-active-user': 'yes', 'x-twitter-auth-type': 'OAuth2Session',
    'Referer': 'https://x.com/', 'Origin': 'https://x.com',
}
tweet_feat = {
    'profile_label_improvements_pcf_label_in_post_enabled': False, 'rweb_tipjar_consumption_enabled': True,
    'responsive_web_graphql_exclude_directive_enabled': True, 'verified_phone_label_enabled': False,
    'creator_subscriptions_tweet_preview_api_enabled': True, 'responsive_web_graphql_timeline_navigation_enabled': True,
    'responsive_web_graphql_skip_user_profile_image_extensions_enabled': False, 'premium_content_api_read_enabled': False,
    'communities_web_enable_tweet_community_results_fetch': True, 'c9s_tweet_anatomy_moderator_badge_enabled': True,
    'responsive_web_grok_analyze_button_fetch_trends_enabled': False, 'responsive_web_grok_analyze_post_followups_enabled': True,
    'responsive_web_jetfuel_frame': False, 'responsive_web_grok_share_attachment_enabled': True,
    'articles_preview_enabled': True, 'responsive_web_edit_tweet_api_enabled': True,
    'graphql_is_translatable_rweb_tweet_is_translatable_enabled': True, 'view_counts_everywhere_api_enabled': True,
    'longform_notetweets_consumption_enabled': True, 'responsive_web_twitter_article_tweet_consumption_enabled': True,
    'tweet_awards_web_tipping_enabled': False, 'creator_subscriptions_quote_tweet_preview_enabled': False,
    'freedom_of_speech_not_reach_fetch_enabled': True, 'standardized_nudges_misinfo': True,
    'tweet_with_visibility_results_prefer_gql_limited_actions_policy_enabled': True, 'rweb_video_screen_enabled': False,
    'longform_notetweets_rich_text_read_enabled': True, 'longform_notetweets_inline_media_enabled': True,
    'responsive_web_media_download_video_enabled': False, 'responsive_web_enhance_cards_enabled': False,
}
variables = {'userId': '18040230', 'count': 20, 'includePromotedContent': False, 'withQuotedTweets': True, 'withVoice': False}

r = cf.get('https://x.com/i/api/graphql/_9v58axugmURcAmrOi7nxw/UserTweets',
           cookies=cookies, headers=hdrs,
           params={'variables': json.dumps(variables, separators=(',', ':')),
                   'features': json.dumps(tweet_feat, separators=(',', ':'))},
           impersonate='chrome120', timeout=30)
js = r.json()
instr = js['data']['user']['result']['timeline']['timeline']['instructions']
for i in instr:
    if i.get('type') == 'TimelineAddEntries':
        for e in i['entries']:
            eid = e['entryId']
            c = e.get('content', {})
            if 'cursor' in eid.lower():
                val = c.get('value', c.get('itemContent', {}).get('value', ''))
                print(f'CURSOR: {eid} = {val[:60]}')
            elif 'tweet-' in eid:
                leg = (c.get('itemContent', {})
                        .get('tweet_results', {})
                        .get('result', {})
                        .get('legacy', {}))
                print(f'TWEET: lang={leg.get("lang","?")} | {leg.get("full_text","")[:80]}')
