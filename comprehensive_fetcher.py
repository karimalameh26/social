"""
Comprehensive Political Data Fetcher for Mark Daou
Uses multiple X API v2 endpoints for sentiment, campaign tracking, and public opinion analysis
"""
import requests
import json
from datetime import datetime, timedelta
import time
from collections import Counter
import re

class ComprehensivePoliticalDataFetcher:
    def __init__(self, bearer_token):
        self.bearer_token = bearer_token
        self.headers = {
            'Authorization': f'Bearer {bearer_token}',
            'Content-Type': 'application/json'
        }
        self.politician_name = "Mark Daou"
        self.politician_username = None
        self.politician_id = None
        
    def _make_request(self, url, params=None, method='GET'):
        """Make API request with error handling and rate limit management"""
        try:
            if method == 'GET':
                response = requests.get(url, headers=self.headers, params=params)
            else:
                response = requests.post(url, headers=self.headers, json=params)
            
            if response.status_code == 429:
                # Rate limit hit
                reset_time = int(response.headers.get('x-rate-limit-reset', time.time() + 900))
                wait_time = reset_time - int(time.time())
                print(f"[RATE LIMIT] Waiting {wait_time} seconds...")
                time.sleep(wait_time + 1)
                return self._make_request(url, params, method)
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"[ERROR] {response.status_code}: {response.text}")
                return None
                
        except Exception as e:
            print(f"[ERROR] Request failed: {e}")
            return None
    
    def find_user(self, username=None):
        """Find user by username or name"""
        if username:
            print(f"[SEARCH] Looking up user: @{username}")
            url = f"https://api.twitter.com/2/users/by/username/{username}"
            params = {
                'user.fields': 'created_at,description,entities,id,location,name,pinned_tweet_id,profile_image_url,protected,public_metrics,url,username,verified,verified_type'
            }
            result = self._make_request(url, params)
            if result and 'data' in result:
                self.politician_username = result['data']['username']
                self.politician_id = result['data']['id']
                return result
        return None
    
    def get_user_tweets(self, max_results=100):
        """Get recent tweets from the politician"""
        if not self.politician_id:
            return None
        
        print(f"[TWEETS] Fetching tweets from @{self.politician_username}...")
        url = f"https://api.twitter.com/2/users/{self.politician_id}/tweets"
        params = {
            'max_results': min(max_results, 100),
            'tweet.fields': 'created_at,public_metrics,entities,referenced_tweets,reply_settings,lang,possibly_sensitive,context_annotations',
            'expansions': 'author_id,referenced_tweets.id,attachments.media_keys',
            'media.fields': 'type,url,preview_image_url'
        }
        return self._make_request(url, params)
    
    def get_mentions(self, max_results=100):
        """Get tweets mentioning the politician"""
        if not self.politician_id:
            return None
        
        print(f"[MENTIONS] Fetching mentions of @{self.politician_username}...")
        url = f"https://api.twitter.com/2/users/{self.politician_id}/mentions"
        params = {
            'max_results': min(max_results, 100),
            'tweet.fields': 'created_at,public_metrics,entities,author_id,lang,possibly_sensitive,context_annotations',
            'expansions': 'author_id',
            'user.fields': 'name,username,verified,public_metrics'
        }
        return self._make_request(url, params)
    
    def search_recent_tweets(self, query=None, max_results=100):
        """Search recent tweets (last 7 days) about the politician"""
        search_query = query or f'"{self.politician_name}" OR @{self.politician_username} Lebanon'
        print(f"[SEARCH] Searching recent tweets: {search_query}")
        
        url = "https://api.twitter.com/2/tweets/search/recent"
        params = {
            'query': search_query,
            'max_results': min(max_results, 100),
            'tweet.fields': 'created_at,public_metrics,entities,author_id,lang,possibly_sensitive,context_annotations',
            'expansions': 'author_id',
            'user.fields': 'name,username,verified,public_metrics,location'
        }
        return self._make_request(url, params)
    
    def get_followers(self, max_results=1000):
        """Get follower information"""
        if not self.politician_id:
            return None
        
        print(f"[FOLLOWERS] Fetching followers of @{self.politician_username}...")
        url = f"https://api.twitter.com/2/users/{self.politician_id}/followers"
        params = {
            'max_results': min(max_results, 1000),
            'user.fields': 'created_at,description,location,public_metrics,verified'
        }
        return self._make_request(url, params)
    
    def get_following(self, max_results=1000):
        """Get accounts the politician follows"""
        if not self.politician_id:
            return None
        
        print(f"[FOLLOWING] Fetching accounts @{self.politician_username} follows...")
        url = f"https://api.twitter.com/2/users/{self.politician_id}/following"
        params = {
            'max_results': min(max_results, 1000),
            'user.fields': 'created_at,description,location,public_metrics,verified'
        }
        return self._make_request(url, params)
    
    def analyze_sentiment_keywords(self, tweets):
        """Basic sentiment analysis based on keywords"""
        positive_keywords = ['great', 'excellent', 'good', 'support', 'progress', 'success', 'hope', 'proud', 
                            'better', 'improvement', 'positive', 'forward', 'reform', 'change', 'future']
        negative_keywords = ['bad', 'terrible', 'corrupt', 'fail', 'crisis', 'problem', 'worse', 'against',
                            'disaster', 'wrong', 'poor', 'decline', 'issue', 'concern', 'worry']
        neutral_keywords = ['discuss', 'meet', 'announce', 'statement', 'comment', 'update', 'news', 'report']
        
        sentiments = {'positive': 0, 'negative': 0, 'neutral': 0}
        
        if not tweets or 'data' not in tweets:
            return sentiments
        
        for tweet in tweets['data']:
            text = tweet.get('text', '').lower()
            pos_count = sum(1 for word in positive_keywords if word in text)
            neg_count = sum(1 for word in negative_keywords if word in text)
            
            if pos_count > neg_count:
                sentiments['positive'] += 1
            elif neg_count > pos_count:
                sentiments['negative'] += 1
            else:
                sentiments['neutral'] += 1
        
        return sentiments
    
    def extract_hashtags(self, tweets):
        """Extract and count hashtags from tweets"""
        hashtags = []
        if not tweets or 'data' not in tweets:
            return {}
        
        for tweet in tweets['data']:
            if 'entities' in tweet and 'hashtags' in tweet['entities']:
                for hashtag in tweet['entities']['hashtags']:
                    hashtags.append(hashtag['tag'].lower())
        
        return dict(Counter(hashtags).most_common(20))
    
    def extract_topics(self, tweets):
        """Extract topics from context annotations"""
        topics = []
        if not tweets or 'data' not in tweets:
            return {}
        
        for tweet in tweets['data']:
            if 'context_annotations' in tweet:
                for annotation in tweet['context_annotations']:
                    if 'domain' in annotation:
                        topics.append(annotation['domain'].get('name', ''))
                    if 'entity' in annotation:
                        topics.append(annotation['entity'].get('name', ''))
        
        return dict(Counter(topics).most_common(20))
    
    def calculate_engagement_metrics(self, tweets):
        """Calculate comprehensive engagement metrics"""
        if not tweets or 'data' not in tweets:
            return {}
        
        metrics = {
            'total_tweets': len(tweets['data']),
            'total_likes': 0,
            'total_retweets': 0,
            'total_replies': 0,
            'total_quotes': 0,
            'avg_likes': 0,
            'avg_retweets': 0,
            'avg_replies': 0,
            'engagement_rate': 0,
            'top_tweet': None
        }
        
        max_engagement = 0
        
        for tweet in tweets['data']:
            if 'public_metrics' in tweet:
                pm = tweet['public_metrics']
                likes = pm.get('like_count', 0)
                retweets = pm.get('retweet_count', 0)
                replies = pm.get('reply_count', 0)
                quotes = pm.get('quote_count', 0)
                
                metrics['total_likes'] += likes
                metrics['total_retweets'] += retweets
                metrics['total_replies'] += replies
                metrics['total_quotes'] += quotes
                
                engagement = likes + retweets + replies + quotes
                if engagement > max_engagement:
                    max_engagement = engagement
                    metrics['top_tweet'] = {
                        'text': tweet.get('text', ''),
                        'id': tweet.get('id', ''),
                        'engagement': engagement,
                        'metrics': pm
                    }
        
        if metrics['total_tweets'] > 0:
            metrics['avg_likes'] = metrics['total_likes'] / metrics['total_tweets']
            metrics['avg_retweets'] = metrics['total_retweets'] / metrics['total_tweets']
            metrics['avg_replies'] = metrics['total_replies'] / metrics['total_tweets']
        
        return metrics
    
    def analyze_posting_patterns(self, tweets):
        """Analyze when the politician posts"""
        if not tweets or 'data' not in tweets:
            return {}
        
        hours = []
        days = []
        
        for tweet in tweets['data']:
            if 'created_at' in tweet:
                dt = datetime.fromisoformat(tweet['created_at'].replace('Z', '+00:00'))
                hours.append(dt.hour)
                days.append(dt.strftime('%A'))
        
        return {
            'most_active_hours': dict(Counter(hours).most_common(5)),
            'most_active_days': dict(Counter(days).most_common(7))
        }
    
    def fetch_comprehensive_data(self, username):
        """Fetch all available data using multiple endpoints"""
        print("\n" + "="*60)
        print(f"  Comprehensive Data Fetcher for {self.politician_name}")
        print("="*60 + "\n")
        
        data = {
            'timestamp': datetime.now().isoformat(),
            'politician': self.politician_name,
            'username': username,
            'user_profile': None,
            'user_tweets': None,
            'mentions': None,
            'search_results': None,
            'followers_sample': None,
            'following_sample': None,
            'sentiment_analysis': {},
            'hashtag_analysis': {},
            'topic_analysis': {},
            'engagement_metrics': {},
            'posting_patterns': {},
            'campaign_metrics': {}
        }
        
        # 1. Find and get user profile
        user_data = self.find_user(username)
        if not user_data:
            print("[ERROR] Could not find user. Please check the username.")
            return None
        
        data['user_profile'] = user_data
        print(f"[SUCCESS] Found user: @{self.politician_username}")
        print(f"   Followers: {user_data['data']['public_metrics']['followers_count']:,}")
        print(f"   Following: {user_data['data']['public_metrics']['following_count']:,}")
        print()
        
        # 2. Get user's tweets
        data['user_tweets'] = self.get_user_tweets(max_results=100)
        if data['user_tweets']:
            print(f"[SUCCESS] Fetched {len(data['user_tweets'].get('data', []))} tweets\n")
        
        # 3. Get mentions
        data['mentions'] = self.get_mentions(max_results=100)
        if data['mentions']:
            print(f"[SUCCESS] Fetched {len(data['mentions'].get('data', []))} mentions\n")
        
        # 4. Search recent tweets
        data['search_results'] = self.search_recent_tweets(max_results=100)
        if data['search_results']:
            print(f"[SUCCESS] Fetched {len(data['search_results'].get('data', []))} search results\n")
        
        # 5. Get sample of followers
        data['followers_sample'] = self.get_followers(max_results=100)
        if data['followers_sample']:
            print(f"[SUCCESS] Fetched {len(data['followers_sample'].get('data', []))} followers sample\n")
        
        # 6. Get following
        data['following_sample'] = self.get_following(max_results=100)
        if data['following_sample']:
            print(f"[SUCCESS] Fetched {len(data['following_sample'].get('data', []))} following accounts\n")
        
        # 7. Analyze sentiment
        print("[ANALYSIS] Analyzing sentiment...")
        data['sentiment_analysis'] = {
            'user_tweets': self.analyze_sentiment_keywords(data['user_tweets']),
            'mentions': self.analyze_sentiment_keywords(data['mentions']),
            'search_results': self.analyze_sentiment_keywords(data['search_results'])
        }
        print("[SUCCESS] Sentiment analysis complete\n")
        
        # 8. Extract hashtags
        print("[ANALYSIS] Analyzing hashtags...")
        data['hashtag_analysis'] = {
            'user_tweets': self.extract_hashtags(data['user_tweets']),
            'mentions': self.extract_hashtags(data['mentions']),
            'search_results': self.extract_hashtags(data['search_results'])
        }
        print("[SUCCESS] Hashtag analysis complete\n")
        
        # 9. Extract topics
        print("[ANALYSIS] Analyzing topics...")
        data['topic_analysis'] = {
            'user_tweets': self.extract_topics(data['user_tweets']),
            'mentions': self.extract_topics(data['mentions'])
        }
        print("[SUCCESS] Topic analysis complete\n")
        
        # 10. Calculate engagement
        print("[ANALYSIS] Calculating engagement metrics...")
        data['engagement_metrics'] = {
            'user_tweets': self.calculate_engagement_metrics(data['user_tweets']),
            'mentions': self.calculate_engagement_metrics(data['mentions'])
        }
        print("[SUCCESS] Engagement metrics complete\n")
        
        # 11. Analyze posting patterns
        print("[ANALYSIS] Analyzing posting patterns...")
        data['posting_patterns'] = self.analyze_posting_patterns(data['user_tweets'])
        print("[SUCCESS] Posting pattern analysis complete\n")
        
        # 12. Campaign metrics
        data['campaign_metrics'] = self._calculate_campaign_metrics(data)
        
        # Save to file
        filename = f'mark_daou_comprehensive_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print("\n" + "="*60)
        print(f"[SUCCESS] Data collection complete!")
        print(f"[FILE] Saved to: {filename}")
        print("="*60 + "\n")
        
        self._print_summary(data)
        
        return data
    
    def _calculate_campaign_metrics(self, data):
        """Calculate campaign-specific metrics"""
        metrics = {
            'reach': 0,
            'influence_score': 0,
            'visibility': 0,
            'engagement_rate': 0,
            'follower_growth_indicator': 'N/A'
        }
        
        if data['user_profile'] and 'data' in data['user_profile']:
            profile = data['user_profile']['data']['public_metrics']
            metrics['reach'] = profile.get('followers_count', 0)
            
            # Simple influence score
            followers = profile.get('followers_count', 0)
            tweets = profile.get('tweet_count', 1)
            metrics['influence_score'] = (followers * 0.7) + (tweets * 0.3)
        
        if data['engagement_metrics'].get('user_tweets'):
            eng = data['engagement_metrics']['user_tweets']
            total_eng = eng.get('total_likes', 0) + eng.get('total_retweets', 0)
            total_tweets = eng.get('total_tweets', 1)
            metrics['engagement_rate'] = (total_eng / total_tweets) if total_tweets > 0 else 0
        
        return metrics
    
    def _print_summary(self, data):
        """Print summary of collected data"""
        print("\n[DATA SUMMARY]")
        print("-" * 60)
        
        if data['user_profile']:
            profile = data['user_profile']['data']
            pm = profile['public_metrics']
            print(f"[PROFILE] @{profile['username']}")
            print(f"   Followers: {pm['followers_count']:,}")
            print(f"   Following: {pm['following_count']:,}")
            print(f"   Total Tweets: {pm['tweet_count']:,}")
            print()
        
        if data['sentiment_analysis'].get('mentions'):
            sent = data['sentiment_analysis']['mentions']
            total = sum(sent.values())
            if total > 0:
                print(f"[SENTIMENT] Public Sentiment (from mentions):")
                print(f"   Positive: {sent['positive']} ({sent['positive']/total*100:.1f}%)")
                print(f"   Negative: {sent['negative']} ({sent['negative']/total*100:.1f}%)")
                print(f"   Neutral:  {sent['neutral']} ({sent['neutral']/total*100:.1f}%)")
                print()
        
        if data['engagement_metrics'].get('user_tweets'):
            eng = data['engagement_metrics']['user_tweets']
            print(f"[ENGAGEMENT] Engagement (own tweets):")
            print(f"   Avg Likes: {eng['avg_likes']:.1f}")
            print(f"   Avg Retweets: {eng['avg_retweets']:.1f}")
            print(f"   Avg Replies: {eng['avg_replies']:.1f}")
            print()
        
        if data['hashtag_analysis'].get('user_tweets'):
            hashtags = data['hashtag_analysis']['user_tweets']
            if hashtags:
                top_5 = list(hashtags.items())[:5]
                print(f"[HASHTAGS] Top Hashtags:")
                for tag, count in top_5:
                    print(f"   #{tag}: {count}")
                print()
        
        print("-" * 60)

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  Mark Daou Comprehensive Political Data Fetcher")
    print("  Using Multiple X API v2 Endpoints")
    print("="*60 + "\n")
    
    bearer_token = input("Enter your X API Bearer Token: ").strip()
    
    if not bearer_token:
        print("\n[ERROR] Bearer Token is required")
        print("\n[INFO] See GET_API_CREDENTIALS.md for instructions on getting your token")
        exit(1)
    
    username = input(f"Enter Mark Daou's X username (e.g., markdaou_lb): ").strip()
    
    if not username:
        print("\n[ERROR] Username is required")
        exit(1)
    
    # Remove @ if user included it
    username = username.lstrip('@')
    
    fetcher = ComprehensivePoliticalDataFetcher(bearer_token)
    data = fetcher.fetch_comprehensive_data(username)
    
    if data:
        print("\n[SUCCESS] Success! You can now run the dashboard:")
        print("   streamlit run enhanced_dashboard.py")
    else:
        print("\n[ERROR] Data collection failed. Please check your credentials and username.")

