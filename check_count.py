#!/usr/bin/env python3
"""Check how many tweets per page with count=100."""
import sys, time
sys.path.insert(0, '/home/xitongzhang/Maylie')
import scraper

sess = scraper.make_tw_session()

uid = scraper._get_twitter_user_id(sess, 'sinaronline')  # 931K tweets
print(f'sinaronline uid: {uid}')
for count in [20, 50, 100]:
    code, js = scraper._tw_gql(sess, 'UserTweets', {
        'userId': uid, 'count': count, 'includePromotedContent': False,
        'withQuotedTweets': True, 'withVoice': False
    })
    tweets, cursor = scraper._parse_user_tweets(js)
    malay = sum(1 for t in tweets if scraper.is_malay(t['text']))
    print(f'  count={count}: {len(tweets)} tweets returned, {malay} Malay ({100*malay//max(len(tweets),1)}%)')
    time.sleep(1)
