#!/usr/bin/env python3
import sys, time
sys.path.insert(0, '/home/xitongzhang/Maylie')
import scraper

sess = scraper.make_tw_session()

for handle in ['KKMPutrajaya', 'RTM_Malaysia', 'sinaronline', 'kosmo_online', 'TV3Malaysia', 'MalaysiaGazette']:
    uid = scraper._get_twitter_user_id(sess, handle)
    if not uid:
        print(f'@{handle}: NOT FOUND')
        time.sleep(1)
        continue
    code, js = scraper._tw_gql(sess, 'UserTweets', {
        'userId': uid, 'count': 50, 'includePromotedContent': False,
        'withQuotedTweets': True, 'withVoice': False
    })
    if code != 200:
        print(f'@{handle}: status {code}')
        time.sleep(1)
        continue
    tweets, _ = scraper._parse_user_tweets(js)
    malay = [t for t in tweets if scraper.is_malay(t['text'])]
    pct = 100 * len(malay) // max(len(tweets), 1)
    sample = malay[0]["text"][:80] if malay else "N/A"
    print(f'@{handle:<20}: {len(tweets)} tweets, {len(malay)} Malay ({pct}%)')
    print(f'  sample: {sample}')
    time.sleep(1)
