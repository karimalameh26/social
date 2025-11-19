"""
Two-Phase Data Collection and Analysis
Phase 1: Collect ALL raw data from X API endpoints and save to JSON
Phase 2: Load data and run AI sentiment analysis, topic modeling, etc.
"""
import os
import json
from datetime import datetime
from dotenv import load_dotenv
import requests
import time

# Load credentials
load_dotenv()

class DataCollector:
    """Phase 1: Collect raw data from all endpoints"""
    
    def __init__(self, bearer_token):
        self.bearer_token = bearer_token
        self.headers = {
            'Authorization': f'Bearer {bearer_token}',
            'Content-Type': 'application/json'
        }
        self.politician_id = None
        self.politician_username = None
        
    def _make_request(self, url, params=None):
        """Make API request with rate limit handling"""
        try:
            response = requests.get(url, headers=self.headers, params=params)
            
            if response.status_code == 429:
                reset_time = int(response.headers.get('x-rate-limit-reset', time.time() + 900))
                wait_time = reset_time - int(time.time())
                print(f"[WAIT] Rate limit. Waiting {wait_time}s...")
                time.sleep(wait_time + 1)
                return self._make_request(url, params)
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"[ERROR] {response.status_code}: {response.text[:100]}")
                return None
        except Exception as e:
            print(f"[ERROR] Request failed: {e}")
            return None
    
    def collect_all_data(self, username):
        """Collect ALL data from X API endpoints"""
        print("\n" + "="*70)
        print("  PHASE 1: DATA COLLECTION FROM X API")
        print("  Collecting raw data from ALL endpoints...")
        print("="*70 + "\n")
        
        data = {
            'collection_timestamp': datetime.now().isoformat(),
            'politician_username': username,
            'raw_data': {}
        }
        
        # 1. User Profile
        print("[1/10] Fetching user profile...")
        url = f"https://api.twitter.com/2/users/by/username/{username}"
        params = {
            'user.fields': 'created_at,description,entities,id,location,name,pinned_tweet_id,profile_image_url,protected,public_metrics,url,username,verified,verified_type,withheld'
        }
        user_data = self._make_request(url, params)
        if user_data and 'data' in user_data:
            self.politician_id = user_data['data']['id']
            self.politician_username = user_data['data']['username']
            data['raw_data']['user_profile'] = user_data
            print(f"[OK] Profile: @{self.politician_username}, ID: {self.politician_id}")
        else:
            print("[ERROR] Could not find user")
            return None
        
        # 2. User Tweets
        print("[2/10] Fetching user tweets (max 100)...")
        url = f"https://api.twitter.com/2/users/{self.politician_id}/tweets"
        params = {
            'max_results': 100,
            'tweet.fields': 'attachments,author_id,context_annotations,conversation_id,created_at,edit_controls,entities,geo,id,in_reply_to_user_id,lang,possibly_sensitive,public_metrics,referenced_tweets,reply_settings,source,text,withheld',
            'expansions': 'author_id,referenced_tweets.id,in_reply_to_user_id,attachments.media_keys,geo.place_id,entities.mentions.username',
            'media.fields': 'alt_text,duration_ms,height,media_key,preview_image_url,type,url,width,public_metrics',
            'place.fields': 'contained_within,country,country_code,full_name,geo,id,name,place_type'
        }
        tweets = self._make_request(url, params)
        if tweets:
            count = len(tweets.get('data', []))
            data['raw_data']['user_tweets'] = tweets
            print(f"[OK] Collected {count} tweets")
        
        # 3. Mentions
        print("[3/10] Fetching mentions (max 100)...")
        url = f"https://api.twitter.com/2/users/{self.politician_id}/mentions"
        params = {
            'max_results': 100,
            'tweet.fields': 'attachments,author_id,context_annotations,conversation_id,created_at,entities,geo,id,in_reply_to_user_id,lang,possibly_sensitive,public_metrics,referenced_tweets,reply_settings,source,text',
            'expansions': 'author_id,referenced_tweets.id,in_reply_to_user_id,geo.place_id',
            'user.fields': 'created_at,description,location,name,public_metrics,username,verified',
            'place.fields': 'country,country_code,full_name,geo,name,place_type'
        }
        mentions = self._make_request(url, params)
        if mentions:
            count = len(mentions.get('data', []))
            data['raw_data']['mentions'] = mentions
            print(f"[OK] Collected {count} mentions")
        
        # 4. Search Results
        print("[4/10] Searching recent tweets...")
        url = "https://api.twitter.com/2/tweets/search/recent"
        search_query = f'("{username}" OR @{username}) (Lebanon)'
        params = {
            'query': search_query,
            'max_results': 100,
            'tweet.fields': 'attachments,author_id,context_annotations,conversation_id,created_at,entities,geo,id,lang,possibly_sensitive,public_metrics,referenced_tweets,text',
            'expansions': 'author_id,geo.place_id',
            'user.fields': 'created_at,description,location,name,public_metrics,username,verified',
            'place.fields': 'country,country_code,full_name,name,place_type'
        }
        search = self._make_request(url, params)
        if search:
            count = len(search.get('data', []))
            data['raw_data']['search_results'] = search
            print(f"[OK] Collected {count} search results")
        
        # 5. Followers
        print("[5/10] Fetching followers (max 1000)...")
        url = f"https://api.twitter.com/2/users/{self.politician_id}/followers"
        params = {
            'max_results': 1000,
            'user.fields': 'created_at,description,entities,location,name,profile_image_url,public_metrics,url,username,verified,verified_type'
        }
        followers = self._make_request(url, params)
        if followers:
            count = len(followers.get('data', []))
            data['raw_data']['followers'] = followers
            print(f"[OK] Collected {count} followers")
        
        # 6. Following
        print("[6/10] Fetching following (max 1000)...")
        url = f"https://api.twitter.com/2/users/{self.politician_id}/following"
        params = {
            'max_results': 1000,
            'user.fields': 'created_at,description,location,name,public_metrics,username,verified,verified_type'
        }
        following = self._make_request(url, params)
        if following:
            count = len(following.get('data', []))
            data['raw_data']['following'] = following
            print(f"[OK] Collected {count} following")
        
        # 7-10. Get engagement on top 3 tweets
        if tweets and 'data' in tweets:
            top_tweets = sorted(
                tweets['data'],
                key=lambda x: x.get('public_metrics', {}).get('like_count', 0),
                reverse=True
            )[:3]
            
            data['raw_data']['top_tweets_engagement'] = {}
            
            for i, tweet in enumerate(top_tweets, 1):
                tweet_id = tweet['id']
                print(f"[{6+i}/10] Analyzing engagement for top tweet {i}...")
                
                engagement = {'tweet': tweet}
                
                # Liking users
                url = f"https://api.twitter.com/2/tweets/{tweet_id}/liking_users"
                params = {'max_results': 100, 'user.fields': 'created_at,location,public_metrics,verified'}
                liking = self._make_request(url, params)
                if liking:
                    engagement['liking_users'] = liking
                    print(f"  - {len(liking.get('data', []))} liking users")
                
                time.sleep(1)
                
                # Retweeting users
                url = f"https://api.twitter.com/2/tweets/{tweet_id}/retweeted_by"
                params = {'max_results': 100, 'user.fields': 'created_at,location,public_metrics,verified'}
                retweeting = self._make_request(url, params)
                if retweeting:
                    engagement['retweeting_users'] = retweeting
                    print(f"  - {len(retweeting.get('data', []))} retweeting users")
                
                time.sleep(1)
                
                # Quote tweets
                url = f"https://api.twitter.com/2/tweets/{tweet_id}/quote_tweets"
                params = {
                    'max_results': 100,
                    'tweet.fields': 'author_id,created_at,public_metrics,text,context_annotations',
                    'expansions': 'author_id',
                    'user.fields': 'name,username,verified,public_metrics'
                }
                quotes = self._make_request(url, params)
                if quotes:
                    engagement['quote_tweets'] = quotes
                    print(f"  - {len(quotes.get('data', []))} quote tweets")
                
                data['raw_data']['top_tweets_engagement'][tweet_id] = engagement
                time.sleep(1)
        
        # Save raw data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        raw_filename = f'mark_daou_raw_data_{timestamp}.json'
        
        print("\n[SAVE] Saving raw data to JSON...")
        with open(raw_filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print("\n" + "="*70)
        print(f"[OK] RAW DATA COLLECTION COMPLETE!")
        print(f"[FILE] Saved to: {raw_filename}")
        print("="*70 + "\n")
        
        return raw_filename

if __name__ == "__main__":
    bearer_token = os.getenv('TWITTER_BEARER_TOKEN')
    username = os.getenv('POLITICIAN_USERNAME', 'DaouMark')
    
    if not bearer_token:
        print("[ERROR] TWITTER_BEARER_TOKEN not found in .env")
        exit(1)
    
    print("[AUTH] Using credentials from .env")
    print(f"[TARGET] @{username}\n")
    
    # Phase 1: Collect data
    collector = DataCollector(bearer_token)
    raw_file = collector.collect_all_data(username)
    
    if raw_file:
        print(f"\n[SUCCESS] Raw data saved to: {raw_file}")
        print("\n[NEXT] Run analysis:")
        print(f"   python analyze_data.py {raw_file}")
    else:
        print("\n[FAILED] Data collection failed")

