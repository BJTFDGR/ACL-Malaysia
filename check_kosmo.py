#!/usr/bin/env python3
"""Check kosmo_online and MalaysiaGazette tweet content."""
import sys, time
sys.path.insert(0, '/home/xitongzhang/Maylie')
import scraper

sess = scraper.make_tw_session()

for handle in ['kosmo_online', 'MalaysiaGazette']:
    uid = scraper._get_twitter_user_id(sess, handle)
    if not uid:
        print(f'@{handle}: NOT FOUND')
        continue
    code, js = scraper._tw_gql(sess, 'UserTweets', {
        'userId': uid, 'count': 20, 'includePromotedContent': False,
        'withQuotedTweets': True, 'withVoice': False
    })
    tweets, _ = scraper._parse_user_tweets(js)
    print(f'@{handle}: {len(tweets)} tweets')
    for t in tweets[:10]:
        match = scraper.matched(t['text'])
        print(f'  [{t["lang"]}] {"|".join(match) if match else "---"} | {t["text"][:80]}')
    time.sleep(1)
