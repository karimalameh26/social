"""
Comprehensive Political Data Fetcher for Mark Daou
Uses multiple X API v2 endpoints for sentiment, campaign tracking, and public opinion analysis
"""
import requests
import json
from datetime import datetime, timedelta, timezone
import time
from collections import Counter
import re
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

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
            'tweet.fields': 'created_at,public_metrics,entities,referenced_tweets,reply_settings,lang,possibly_sensitive,context_annotations,geo',
            'expansions': 'author_id,referenced_tweets.id,attachments.media_keys,geo.place_id',
            'media.fields': 'type,url,preview_image_url',
            'place.fields': 'full_name,country,country_code,geo,name,place_type'
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
            'tweet.fields': 'created_at,public_metrics,entities,author_id,lang,possibly_sensitive,context_annotations,geo',
            'expansions': 'author_id,geo.place_id',
            'user.fields': 'name,username,verified,public_metrics,location',
            'place.fields': 'full_name,country,country_code,geo,name,place_type'
        }
        return self._make_request(url, params)
    
    def search_recent_tweets(self, query=None, max_results=100):
        """Search recent tweets (last 7 days) about the politician
        
        Geographic data extraction:
        - user.location: Self-reported location from user profiles (e.g., "Beirut, Lebanon")
        - geo.place_id: Precise location if tweet is geo-tagged (rare, <1% of tweets)
        - place objects: Structured city/country data when available
        """
        search_query = query or f'"{self.politician_name}" OR @{self.politician_username}'
        print(f"[SEARCH] Searching recent tweets: {search_query}")
        
        url = "https://api.twitter.com/2/tweets/search/recent"
        params = {
            'query': search_query,
            'max_results': min(max_results, 100),
            'tweet.fields': 'created_at,public_metrics,entities,author_id,lang,possibly_sensitive,context_annotations,geo',
            'expansions': 'author_id,geo.place_id',
            'user.fields': 'name,username,verified,public_metrics,location,description,created_at',
            'place.fields': 'full_name,country,country_code,geo,name,place_type'
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
    
    def get_liking_users(self, tweet_id, max_results=100):
        """Get users who liked a specific tweet
        
        Geographic data: Extracts user.location from profiles of users who liked the tweet
        This shows geographic distribution of people engaging with content
        """
        print(f"[ENGAGEMENT] Fetching users who liked tweet {tweet_id}...")
        url = f"https://api.twitter.com/2/tweets/{tweet_id}/liking_users"
        params = {
            'max_results': min(max_results, 100),
            'user.fields': 'created_at,description,location,public_metrics,verified,name,username'
        }
        return self._make_request(url, params)
    
    def get_retweeting_users(self, tweet_id, max_results=100):
        """Get users who retweeted a specific tweet
        
        Geographic data: Extracts user.location from profiles of users who retweeted
        This shows where the content is being amplified geographically
        """
        print(f"[ENGAGEMENT] Fetching users who retweeted tweet {tweet_id}...")
        url = f"https://api.twitter.com/2/tweets/{tweet_id}/retweeted_by"
        params = {
            'max_results': min(max_results, 100),
            'user.fields': 'created_at,description,location,public_metrics,verified,name,username'
        }
        return self._make_request(url, params)
    
    def get_quote_tweets(self, tweet_id, max_results=100):
        """Get quote tweets of a specific tweet
        
        Geographic data: Extracts author location from quote tweet authors
        Shows where people are discussing/commenting on the content
        """
        print(f"[ENGAGEMENT] Fetching quote tweets of {tweet_id}...")
        url = f"https://api.twitter.com/2/tweets/{tweet_id}/quote_tweets"
        params = {
            'max_results': min(max_results, 100),
            'tweet.fields': 'created_at,public_metrics,author_id,text',
            'expansions': 'author_id',
            'user.fields': 'location,name,username,verified,public_metrics'
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
    
    def analyze_location_data(self, tweets):
        """Analyze geographic data from tweets and user profiles
        
        Extracts location from 3 sources:
        1. user.location - Self-reported text from user profiles (PRIMARY SOURCE)
        2. geo.place_id - GPS location if tweet is geo-tagged (rare)
        3. place objects - Structured location data when available
        """
        location_data = {
            'geotagged_tweets': [],
            'user_locations': [],
            'countries': [],
            'cities': [],
            'has_geo_data': False
        }
        
        if not tweets or 'data' not in tweets:
            return location_data
        
        # Extract geo data from tweets (GPS location - rare)
        for tweet in tweets['data']:
            if 'geo' in tweet and tweet['geo']:
                location_data['has_geo_data'] = True
                location_data['geotagged_tweets'].append({
                    'tweet_id': tweet['id'],
                    'text': tweet.get('text', '')[:100] + '...',
                    'geo': tweet['geo']
                })
        
        # Extract location from places (if included in response)
        if 'includes' in tweets and 'places' in tweets['includes']:
            for place in tweets['includes']['places']:
                location_data['cities'].append(place.get('full_name', ''))
                location_data['countries'].append(place.get('country', ''))
        
        # Extract user profile locations (MAIN SOURCE - self-reported text)
        if 'includes' in tweets and 'users' in tweets['includes']:
            for user in tweets['includes']['users']:
                if 'location' in user and user['location']:
                    location_data['user_locations'].append(user['location'])
        
        # Count occurrences
        location_data['top_cities'] = dict(Counter(location_data['cities']).most_common(10))
        location_data['top_countries'] = dict(Counter(location_data['countries']).most_common(10))
        location_data['top_user_locations'] = dict(Counter(location_data['user_locations']).most_common(10))
        
        return location_data
    
    def analyze_engagement_geography(self, engagement_data):
        """Analyze geographic distribution of users who engaged with content
        
        INPUT: Data from liking_users, retweeted_by, or quote_tweets endpoints
        EXTRACTS: user.location field from each user's profile
        
        Location is self-reported text like:
        - "Beirut, Lebanon"
        - "Paris, France" 
        - "بيروت"
        - "Lebanese in USA"
        
        Returns geographic breakdown including Lebanon vs Diaspora classification
        """
        if not engagement_data or 'data' not in engagement_data:
            return {
                'total_users': 0,
                'with_location': 0,
                'locations': [],
                'lebanon_based': 0,
                'diaspora': 0,
                'top_locations': {},
                'top_countries': {},
                'top_cities': {}
            }
        
        users = engagement_data['data']
        locations = []
        countries = []
        cities = []
        lebanon_count = 0
        diaspora_count = 0
        no_location = 0
        
        # Lebanese location keywords
        lebanese_keywords = [
            'beirut', 'بيروت', 'lebanon', 'لبنان', 'tripoli', 'طرابلس',
            'sidon', 'صيدا', 'tyre', 'صور', 'baalbek', 'بعلبك',
            'zahle', 'زحلة', 'jounieh', 'جونية', 'byblos', 'جبيل',
            'choueifat', 'الشويفات', 'bekaa', 'البقاع', 'mount lebanon',
            'north lebanon', 'south lebanon', 'nabatieh', 'النبطية',
            'saida', 'tyr', 'jbeil', 'liban'
        ]
        
        for user in users:
            location = user.get('location', '').strip()
            
            if not location:
                no_location += 1
                continue
            
            locations.append(location)
            location_lower = location.lower()
            
            # Classify as Lebanon-based or Diaspora
            is_lebanon = any(keyword in location_lower for keyword in lebanese_keywords)
            
            if is_lebanon:
                lebanon_count += 1
            else:
                diaspora_count += 1
            
            # Parse location (simple approach)
            location_parts = [part.strip() for part in location.split(',')]
            if len(location_parts) >= 2:
                cities.append(location_parts[0])
                countries.append(location_parts[-1])
            elif len(location_parts) == 1:
                countries.append(location_parts[0])
        
        total_with_location = lebanon_count + diaspora_count
        
        return {
            'total_users': len(users),
            'with_location': total_with_location,
            'without_location': no_location,
            'lebanon_based': lebanon_count,
            'diaspora': diaspora_count,
            'lebanon_percentage': (lebanon_count / total_with_location * 100) if total_with_location > 0 else 0,
            'diaspora_percentage': (diaspora_count / total_with_location * 100) if total_with_location > 0 else 0,
            'top_locations': dict(Counter(locations).most_common(20)),
            'top_countries': dict(Counter(countries).most_common(15)),
            'top_cities': dict(Counter(cities).most_common(15)),
            'location_coverage': (total_with_location / len(users) * 100) if len(users) > 0 else 0
        }
    
    def analyze_follower_geography(self, followers_data):
        """Advanced geographic analysis of followers"""
        if not followers_data or 'data' not in followers_data:
            return {}
        
        followers = followers_data['data']
        
        # Initialize containers
        locations = []
        countries = []
        cities = []
        lebanon_followers = 0
        diaspora_followers = 0
        unknown_location = 0
        
        # Lebanese city/region keywords
        lebanese_keywords = [
            'beirut', 'بيروت', 'lebanon', 'لبنان', 'tripoli', 'طرابلس', 
            'sidon', 'صيدا', 'tyre', 'صور', 'baalbek', 'بعلبك',
            'zahle', 'زحلة', 'jounieh', 'جونية', 'byblos', 'جبيل',
            'choueifat', 'الشويفات', 'bekaa', 'البقاع', 'mount lebanon',
            'north lebanon', 'south lebanon', 'nabatieh', 'النبطية'
        ]
        
        for follower in followers:
            location = follower.get('location', '').strip()
            
            if not location:
                unknown_location += 1
                continue
            
            locations.append(location)
            location_lower = location.lower()
            
            # Check if Lebanon-based or diaspora
            is_lebanon = any(keyword in location_lower for keyword in lebanese_keywords)
            
            if is_lebanon:
                lebanon_followers += 1
            else:
                diaspora_followers += 1
            
            # Extract country/city (simple parsing)
            location_parts = [part.strip() for part in location.split(',')]
            if len(location_parts) >= 2:
                cities.append(location_parts[0])
                countries.append(location_parts[-1])
            elif len(location_parts) == 1:
                countries.append(location_parts[0])
        
        total_with_location = lebanon_followers + diaspora_followers
        
        return {
            'total_followers': len(followers),
            'with_location': total_with_location,
            'without_location': unknown_location,
            'lebanon_based': lebanon_followers,
            'diaspora': diaspora_followers,
            'lebanon_percentage': (lebanon_followers / total_with_location * 100) if total_with_location > 0 else 0,
            'diaspora_percentage': (diaspora_followers / total_with_location * 100) if total_with_location > 0 else 0,
            'top_locations': dict(Counter(locations).most_common(20)),
            'top_countries': dict(Counter(countries).most_common(15)),
            'top_cities': dict(Counter(cities).most_common(15)),
            'location_diversity': len(set(locations))
        }
    
    def analyze_follower_demographics(self, followers_data):
        """Analyze follower demographics and characteristics"""
        if not followers_data or 'data' not in followers_data:
            return {}
        
        followers = followers_data['data']
        
        verified_count = 0
        total_followers = 0
        total_following = 0
        total_tweets = 0
        account_ages = []
        follower_ratios = []
        influential_followers = []
        
        now = datetime.now(timezone.utc)
        
        for follower in followers:
            # Verified status
            if follower.get('verified', False):
                verified_count += 1
            
            # Public metrics
            metrics = follower.get('public_metrics', {})
            followers_count = metrics.get('followers_count', 0)
            following_count = metrics.get('following_count', 0)
            tweet_count = metrics.get('tweet_count', 0)
            
            total_followers += followers_count
            total_following += following_count
            total_tweets += tweet_count
            
            # Calculate follower ratio (followers/following)
            if following_count > 0:
                ratio = followers_count / following_count
                follower_ratios.append(ratio)
            
            # Account age
            if 'created_at' in follower:
                created = datetime.fromisoformat(follower['created_at'].replace('Z', '+00:00'))
                age_days = (now - created).days
                age_years = age_days / 365.25
                account_ages.append(age_years)
            
            # Identify influential followers (>10k followers)
            if followers_count >= 10000:
                influential_followers.append({
                    'username': follower.get('username', 'unknown'),
                    'name': follower.get('name', 'unknown'),
                    'followers': followers_count,
                    'verified': follower.get('verified', False),
                    'location': follower.get('location', 'Unknown')
                })
        
        total = len(followers)
        
        # Calculate averages
        avg_followers = total_followers / total if total > 0 else 0
        avg_following = total_following / total if total > 0 else 0
        avg_tweets = total_tweets / total if total > 0 else 0
        avg_age = sum(account_ages) / len(account_ages) if account_ages else 0
        avg_ratio = sum(follower_ratios) / len(follower_ratios) if follower_ratios else 0
        
        # Sort influential followers
        influential_followers.sort(key=lambda x: x['followers'], reverse=True)
        
        return {
            'total_analyzed': total,
            'verified_count': verified_count,
            'verified_percentage': (verified_count / total * 100) if total > 0 else 0,
            'avg_followers_per_follower': round(avg_followers, 1),
            'avg_following_per_follower': round(avg_following, 1),
            'avg_tweets_per_follower': round(avg_tweets, 1),
            'avg_account_age_years': round(avg_age, 1),
            'avg_follower_ratio': round(avg_ratio, 2),
            'influential_followers': influential_followers[:20],
            'influential_count': len(influential_followers),
            'account_age_distribution': {
                'under_1_year': sum(1 for age in account_ages if age < 1),
                '1_3_years': sum(1 for age in account_ages if 1 <= age < 3),
                '3_5_years': sum(1 for age in account_ages if 3 <= age < 5),
                '5_10_years': sum(1 for age in account_ages if 5 <= age < 10),
                'over_10_years': sum(1 for age in account_ages if age >= 10)
            }
        }
    
    def analyze_follower_segmentation(self, followers_data):
        """Segment followers by activity level and type"""
        if not followers_data or 'data' not in followers_data:
            return {}
        
        followers = followers_data['data']
        
        # Activity segmentation
        very_active = []  # >10k tweets
        active = []       # 1k-10k tweets
        moderate = []     # 100-1k tweets
        low_activity = [] # <100 tweets
        
        # Influence segmentation
        mega_influencers = []    # >100k followers
        macro_influencers = []   # 10k-100k followers
        micro_influencers = []   # 1k-10k followers
        regular_users = []       # <1k followers
        
        # Engagement type
        broadcasters = []  # High followers, low following
        connectors = []    # Balanced ratio
        followers_type = [] # High following, low followers
        
        # Lebanese keywords for bio analysis
        political_keywords = [
            'politics', 'political', 'parliament', 'mp', 'minister', 'government',
            'سياسة', 'سياسي', 'نائب', 'وزير', 'حكومة', 'برلمان',
            'lebanon', 'lebanese', 'لبنان', 'لبناني'
        ]
        
        political_followers = []
        
        for follower in followers:
            username = follower.get('username', 'unknown')
            metrics = follower.get('public_metrics', {})
            
            followers_count = metrics.get('followers_count', 0)
            following_count = metrics.get('following_count', 0)
            tweet_count = metrics.get('tweet_count', 0)
            
            # Activity segmentation
            if tweet_count >= 10000:
                very_active.append(username)
            elif tweet_count >= 1000:
                active.append(username)
            elif tweet_count >= 100:
                moderate.append(username)
            else:
                low_activity.append(username)
            
            # Influence segmentation
            if followers_count >= 100000:
                mega_influencers.append(username)
            elif followers_count >= 10000:
                macro_influencers.append(username)
            elif followers_count >= 1000:
                micro_influencers.append(username)
            else:
                regular_users.append(username)
            
            # Engagement type
            if following_count > 0:
                ratio = followers_count / following_count
                if ratio > 2:
                    broadcasters.append(username)
                elif 0.5 <= ratio <= 2:
                    connectors.append(username)
                else:
                    followers_type.append(username)
            
            # Political interest (from bio)
            bio = follower.get('description', '').lower()
            if any(keyword in bio for keyword in political_keywords):
                political_followers.append({
                    'username': username,
                    'name': follower.get('name', 'unknown'),
                    'bio': follower.get('description', '')[:100]
                })
        
        total = len(followers)
        
        return {
            'activity_levels': {
                'very_active': {
                    'count': len(very_active),
                    'percentage': (len(very_active) / total * 100) if total > 0 else 0,
                    'description': '>10k tweets'
                },
                'active': {
                    'count': len(active),
                    'percentage': (len(active) / total * 100) if total > 0 else 0,
                    'description': '1k-10k tweets'
                },
                'moderate': {
                    'count': len(moderate),
                    'percentage': (len(moderate) / total * 100) if total > 0 else 0,
                    'description': '100-1k tweets'
                },
                'low_activity': {
                    'count': len(low_activity),
                    'percentage': (len(low_activity) / total * 100) if total > 0 else 0,
                    'description': '<100 tweets'
                }
            },
            'influence_levels': {
                'mega_influencers': {
                    'count': len(mega_influencers),
                    'percentage': (len(mega_influencers) / total * 100) if total > 0 else 0,
                    'description': '>100k followers'
                },
                'macro_influencers': {
                    'count': len(macro_influencers),
                    'percentage': (len(macro_influencers) / total * 100) if total > 0 else 0,
                    'description': '10k-100k followers'
                },
                'micro_influencers': {
                    'count': len(micro_influencers),
                    'percentage': (len(micro_influencers) / total * 100) if total > 0 else 0,
                    'description': '1k-10k followers'
                },
                'regular_users': {
                    'count': len(regular_users),
                    'percentage': (len(regular_users) / total * 100) if total > 0 else 0,
                    'description': '<1k followers'
                }
            },
            'engagement_types': {
                'broadcasters': {
                    'count': len(broadcasters),
                    'percentage': (len(broadcasters) / total * 100) if total > 0 else 0,
                    'description': 'High reach (followers > 2x following)'
                },
                'connectors': {
                    'count': len(connectors),
                    'percentage': (len(connectors) / total * 100) if total > 0 else 0,
                    'description': 'Balanced engagement'
                },
                'followers': {
                    'count': len(followers_type),
                    'percentage': (len(followers_type) / total * 100) if total > 0 else 0,
                    'description': 'Follow many (following > 2x followers)'
                }
            },
            'political_interest': {
                'count': len(political_followers),
                'percentage': (len(political_followers) / total * 100) if total > 0 else 0,
                'sample': political_followers[:10]
            }
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
        
        # 2. Get user's tweets (50 to stay under API limits)
        data['user_tweets'] = self.get_user_tweets(max_results=50)
        if data['user_tweets']:
            print(f"[SUCCESS] Fetched {len(data['user_tweets'].get('data', []))} tweets\n")
        
        # 3. Get mentions (50 to stay under API limits)
        data['mentions'] = self.get_mentions(max_results=50)
        if data['mentions']:
            print(f"[SUCCESS] Fetched {len(data['mentions'].get('data', []))} mentions\n")
        
        # 4. Search recent tweets - NOW ENABLED for geographic analysis
        print("[SEARCH] Searching for recent discussions about the politician...")
        data['search_results'] = self.search_recent_tweets(max_results=100)
        if data['search_results']:
            print(f"[SUCCESS] Fetched {len(data['search_results'].get('data', []))} search results\n")
        else:
            print("[INFO] No search results found\n")
        
        # 5. Get engagement users for top tweets (NEW - for geographic analysis)
        print("[ENGAGEMENT] Analyzing who engages with top-performing content...")
        data['engagement_users'] = {}
        
        if data['user_tweets'] and 'data' in data['user_tweets']:
            # Find top 3 tweets by engagement
            tweets_with_engagement = []
            for tweet in data['user_tweets']['data']:
                if 'public_metrics' in tweet:
                    pm = tweet['public_metrics']
                    engagement = pm.get('like_count', 0) + pm.get('retweet_count', 0)
                    tweets_with_engagement.append({
                        'id': tweet['id'],
                        'text': tweet.get('text', '')[:50],
                        'engagement': engagement
                    })
            
            # Sort and get top 3
            top_tweets = sorted(tweets_with_engagement, key=lambda x: x['engagement'], reverse=True)[:3]
            
            print(f"[ENGAGEMENT] Found {len(top_tweets)} top-performing tweets to analyze\n")
            
            # For each top tweet, get retweeting users
            # Note: liking_users endpoint requires OAuth User Context (not available with Bearer Token)
            for idx, tweet_info in enumerate(top_tweets, 1):
                tweet_id = tweet_info['id']
                print(f"[ENGAGEMENT] Analyzing tweet #{idx}: {tweet_info['text']}... (engagement: {tweet_info['engagement']})")
                
                # Get users who retweeted this tweet (liking_users not available with Bearer Token)
                retweeting = self.get_retweeting_users(tweet_id, max_results=100)
                
                data['engagement_users'][tweet_id] = {
                    'tweet_info': tweet_info,
                    'liking_users': None,  # Requires OAuth User Context authentication
                    'retweeting_users': retweeting
                }
                
                if retweeting and 'data' in retweeting:
                    print(f"   └─ Retweets: {len(retweeting['data'])} users with locations")
                print()
        
        print("[SUCCESS] Engagement user analysis complete\n")
        
        # 6. Followers - NOT AVAILABLE in Basic plan
        # The GET /2/users/:id/followers endpoint is not included in Basic tier
        data['followers'] = None
        print("[INFO] Followers endpoint not available in Basic plan - skipped\n")
        
        # 7. Following - NOT AVAILABLE in Basic plan
        # The GET /2/users/:id/following endpoint is not included in Basic tier
        data['following'] = None
        print("[INFO] Following endpoint not available in Basic plan - skipped\n")
        print(f"[QUOTA] Total API usage: ~300-350 items (50 tweets + 50 mentions + 100 search + ~145 retweet users)\n")
        
        # 8. Analyze sentiment
        print("[ANALYSIS] Analyzing sentiment...")
        data['sentiment_analysis'] = {
            'user_tweets': self.analyze_sentiment_keywords(data['user_tweets']),
            'mentions': self.analyze_sentiment_keywords(data['mentions']),
            'search_results': self.analyze_sentiment_keywords(data['search_results'])
        }
        print("[SUCCESS] Sentiment analysis complete\n")
        
        # 9. Extract hashtags
        print("[ANALYSIS] Analyzing hashtags...")
        data['hashtag_analysis'] = {
            'user_tweets': self.extract_hashtags(data['user_tweets']),
            'mentions': self.extract_hashtags(data['mentions']),
            'search_results': self.extract_hashtags(data['search_results'])
        }
        print("[SUCCESS] Hashtag analysis complete\n")
        
        # 10. Extract topics
        print("[ANALYSIS] Analyzing topics...")
        data['topic_analysis'] = {
            'user_tweets': self.extract_topics(data['user_tweets']),
            'mentions': self.extract_topics(data['mentions']),
            'search_results': self.extract_topics(data['search_results'])
        }
        print("[SUCCESS] Topic analysis complete\n")
        
        # 11. Calculate engagement
        print("[ANALYSIS] Calculating engagement metrics...")
        data['engagement_metrics'] = {
            'user_tweets': self.calculate_engagement_metrics(data['user_tweets']),
            'mentions': self.calculate_engagement_metrics(data['mentions']),
            'search_results': self.calculate_engagement_metrics(data['search_results'])
        }
        print("[SUCCESS] Engagement metrics complete\n")
        
        # 12. Analyze posting patterns
        print("[ANALYSIS] Analyzing posting patterns...")
        data['posting_patterns'] = self.analyze_posting_patterns(data['user_tweets'])
        print("[SUCCESS] Posting pattern analysis complete\n")
        
        # 13. Analyze location data from tweets and mentions
        print("[ANALYSIS] Analyzing geographic/location data from content...")
        data['location_analysis'] = {
            'user_tweets': self.analyze_location_data(data['user_tweets']),
            'mentions': self.analyze_location_data(data['mentions']),
            'search_results': self.analyze_location_data(data['search_results'])
        }
        print("[SUCCESS] Content location analysis complete\n")
        
        # 14. Analyze geographic distribution of engagement users (NEW!)
        print("[ANALYSIS] Analyzing WHERE people who engage with content are located...")
        data['engagement_geography'] = {}
        
        for tweet_id, engagement_data in data.get('engagement_users', {}).items():
            # Note: liking_users requires OAuth User Context (not available)
            retweet_geo = self.analyze_engagement_geography(engagement_data.get('retweeting_users'))
            
            data['engagement_geography'][tweet_id] = {
                'tweet_info': engagement_data['tweet_info'],
                'liking_users_geography': {},  # Not available with Bearer Token
                'retweeting_users_geography': retweet_geo
            }
            
            # Print summary for retweets
            if retweet_geo['total_users'] > 0:
                print(f"   Tweet: {engagement_data['tweet_info']['text']}...")
                print(f"   └─ Retweets: {retweet_geo['lebanon_based']} Lebanon ({retweet_geo['lebanon_percentage']:.1f}%), "
                      f"{retweet_geo['diaspora']} Diaspora ({retweet_geo['diaspora_percentage']:.1f}%)")
                if retweet_geo['top_locations']:
                    top_loc = list(retweet_geo['top_locations'].items())[0]
                    print(f"   └─ Top location: {top_loc[0]} ({top_loc[1]} users)")
        
        print("\n[SUCCESS] Engagement geography analysis complete\n")
        
        # 15. Advanced follower analytics - SKIPPED (not available in Basic plan)
        print("[INFO] Follower analytics skipped (endpoint not available in Basic plan)")
        data['follower_geography'] = {}
        data['follower_demographics'] = {}
        data['follower_segmentation'] = {}
        print()
        
        # 16. Campaign metrics
        data['campaign_metrics'] = self._calculate_campaign_metrics(data)
        
        # 17. Extract insights
        print("[INSIGHTS] Extracting actionable insights...")
        insights = self.extract_insights(data)
        print("[SUCCESS] Insights extraction complete\n")
        
        # Save comprehensive data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'mark_daou_comprehensive_{timestamp}.json'
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        # Save insights to separate file
        insights_filename = f'mark_daou_insights_{timestamp}.json'
        with open(insights_filename, 'w', encoding='utf-8') as f:
            json.dump(insights, f, indent=2, ensure_ascii=False)
        
        print("\n" + "="*60)
        print(f"[SUCCESS] Data collection complete!")
        print(f"[FILE] Data saved to: {filename}")
        print(f"[FILE] Insights saved to: {insights_filename}")
        print("="*60 + "\n")
        
        self._print_summary(data)
        self._print_insights(insights)
        
        # Auto-trigger advanced analysis
        print("\n" + "="*60)
        print("  TRIGGERING ADVANCED ANALYSIS...")
        print("="*60 + "\n")
        
        try:
            from advanced_insights_analyzer import AdvancedInsightsAnalyzer
            analyzer = AdvancedInsightsAnalyzer()
            analyzed_file, analyzed_results = analyzer.analyze_comprehensive_data(filename)
            
            print(f"\n[SUCCESS] Advanced analysis complete: {analyzed_file}\n")
        except Exception as e:
            print(f"\n[WARNING] Advanced analysis failed: {e}")
            print("[INFO] You can run it manually: python advanced_insights_analyzer.py")
            print()
        
        return data, insights
    
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
    
    def extract_insights(self, data):
        """Extract actionable insights from collected data"""
        insights = {
            'timestamp': datetime.now().isoformat(),
            'summary': {},
            'sentiment_insights': {},
            'engagement_insights': {},
            'follower_insights': {},
            'content_insights': {},
            'recommendations': []
        }
        
        # Summary metrics
        insights['summary'] = {
            'total_reach': data.get('user_profile', {}).get('data', {}).get('public_metrics', {}).get('followers_count', 0),
            'tweets_analyzed': len(data.get('user_tweets', {}).get('data', [])) if data.get('user_tweets') else 0,
            'mentions_analyzed': len(data.get('mentions', {}).get('data', [])) if data.get('mentions') else 0,
            'followers_analyzed': len(data.get('followers', {}).get('data', [])) if data.get('followers') else 0
        }
        
        # Sentiment insights
        sentiment = data.get('sentiment_analysis', {})
        if sentiment:
            own_tweets_sent = sentiment.get('user_tweets', {})
            mentions_sent = sentiment.get('mentions', {})
            
            own_total = sum(own_tweets_sent.values())
            mentions_total = sum(mentions_sent.values())
            
            if own_total > 0:
                insights['sentiment_insights']['own_tweets'] = {
                    'dominant_sentiment': max(own_tweets_sent, key=own_tweets_sent.get),
                    'positive_rate': own_tweets_sent.get('positive', 0) / own_total * 100,
                    'negative_rate': own_tweets_sent.get('negative', 0) / own_total * 100,
                    'sentiment_score': (own_tweets_sent.get('positive', 0) - own_tweets_sent.get('negative', 0)) / own_total * 100
                }
            
            if mentions_total > 0:
                insights['sentiment_insights']['public_perception'] = {
                    'dominant_sentiment': max(mentions_sent, key=mentions_sent.get),
                    'positive_rate': mentions_sent.get('positive', 0) / mentions_total * 100,
                    'negative_rate': mentions_sent.get('negative', 0) / mentions_total * 100,
                    'sentiment_score': (mentions_sent.get('positive', 0) - mentions_sent.get('negative', 0)) / mentions_total * 100
                }
        
        # Engagement insights
        eng = data.get('engagement_metrics', {}).get('user_tweets', {})
        if eng:
            insights['engagement_insights'] = {
                'avg_engagement_per_tweet': eng.get('avg_likes', 0) + eng.get('avg_retweets', 0) + eng.get('avg_replies', 0),
                'best_performing_metric': max(
                    [('likes', eng.get('avg_likes', 0)), 
                     ('retweets', eng.get('avg_retweets', 0)), 
                     ('replies', eng.get('avg_replies', 0))],
                    key=lambda x: x[1]
                )[0],
                'engagement_rate': data.get('campaign_metrics', {}).get('engagement_rate', 0),
                'top_tweet_engagement': eng.get('top_tweet', {}).get('engagement', 0)
            }
        
        # Follower insights
        follower_geo = data.get('follower_geography', {})
        follower_demo = data.get('follower_demographics', {})
        follower_seg = data.get('follower_segmentation', {})
        
        if follower_geo:
            insights['follower_insights']['geography'] = {
                'lebanon_percentage': follower_geo.get('lebanon_percentage', 0),
                'diaspora_percentage': follower_geo.get('diaspora_percentage', 0),
                'top_location': list(follower_geo.get('top_locations', {}).items())[0] if follower_geo.get('top_locations') else None,
                'location_diversity': follower_geo.get('location_diversity', 0)
            }
        
        if follower_demo:
            insights['follower_insights']['demographics'] = {
                'verified_percentage': follower_demo.get('verified_percentage', 0),
                'avg_account_age': follower_demo.get('avg_account_age_years', 0),
                'influential_count': follower_demo.get('influential_count', 0),
                'avg_follower_quality': follower_demo.get('avg_followers_per_follower', 0)
            }
        
        if follower_seg:
            activity = follower_seg.get('activity_levels', {})
            influence = follower_seg.get('influence_levels', {})
            political = follower_seg.get('political_interest', {})
            
            insights['follower_insights']['segmentation'] = {
                'active_percentage': (activity.get('very_active', {}).get('percentage', 0) + 
                                     activity.get('active', {}).get('percentage', 0)),
                'influencer_percentage': (influence.get('mega_influencers', {}).get('percentage', 0) +
                                         influence.get('macro_influencers', {}).get('percentage', 0)),
                'politically_engaged_percentage': political.get('percentage', 0)
            }
        
        # Content insights
        hashtags = data.get('hashtag_analysis', {}).get('user_tweets', {})
        topics = data.get('topic_analysis', {}).get('user_tweets', {})
        
        if hashtags:
            top_hashtags = list(hashtags.items())[:5]
            insights['content_insights']['top_hashtags'] = [{'tag': tag, 'count': count} for tag, count in top_hashtags]
        
        if topics:
            top_topics = list(topics.items())[:5]
            insights['content_insights']['top_topics'] = [{'topic': topic, 'count': count} for topic, count in top_topics]
        
        # Generate recommendations
        recommendations = []
        
        # Sentiment recommendations
        if insights['sentiment_insights'].get('public_perception'):
            score = insights['sentiment_insights']['public_perception']['sentiment_score']
            if score < -20:
                recommendations.append({
                    'priority': 'HIGH',
                    'category': 'Sentiment',
                    'finding': f"Public sentiment is negative ({score:.1f}%)",
                    'action': "Increase positive messaging, address concerns, engage with critics constructively"
                })
            elif score > 20:
                recommendations.append({
                    'priority': 'POSITIVE',
                    'category': 'Sentiment',
                    'finding': f"Public sentiment is positive ({score:.1f}%)",
                    'action': "Maintain current messaging strategy, amplify successful content"
                })
        
        # Engagement recommendations
        if insights['engagement_insights']:
            avg_eng = insights['engagement_insights']['avg_engagement_per_tweet']
            if avg_eng < 10:
                recommendations.append({
                    'priority': 'MEDIUM',
                    'category': 'Engagement',
                    'finding': f"Low average engagement ({avg_eng:.1f} per tweet)",
                    'action': "Post during peak hours, use more multimedia, increase direct interactions"
                })
        
        # Follower recommendations
        if insights['follower_insights'].get('geography'):
            diaspora = insights['follower_insights']['geography']['diaspora_percentage']
            if diaspora > 40:
                recommendations.append({
                    'priority': 'OPPORTUNITY',
                    'category': 'Audience',
                    'finding': f"Large diaspora following ({diaspora:.1f}%)",
                    'action': "Create diaspora-specific content, highlight Lebanese abroad issues"
                })
        
        if insights['follower_insights'].get('segmentation'):
            influencer_pct = insights['follower_insights']['segmentation']['influencer_percentage']
            if influencer_pct > 2:
                recommendations.append({
                    'priority': 'OPPORTUNITY',
                    'category': 'Influence',
                    'finding': f"{influencer_pct:.1f}% of followers are influencers",
                    'action': "Engage with influential followers, encourage retweets, build amplification network"
                })
        
        insights['recommendations'] = recommendations
        
        return insights
    
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
        
        if data.get('location_analysis'):
            print(f"[LOCATION] Geographic Data:")
            
            # User profile location
            if data['user_profile'] and 'location' in data['user_profile']['data']:
                print(f"   Profile Location: {data['user_profile']['data']['location']}")
            
            # Top user locations from mentions/interactions
            mentions_loc = data['location_analysis'].get('mentions', {})
            if mentions_loc.get('top_user_locations'):
                top_locs = list(mentions_loc['top_user_locations'].items())[:5]
                print(f"   Top Mention Locations:")
                for loc, count in top_locs:
                    print(f"      {loc}: {count}")
            
            # Geotagged tweets
            search_loc = data['location_analysis'].get('search_results', {})
            if search_loc.get('has_geo_data'):
                print(f"   [OK] Geographic data found in search results")
            
            print()
        
        print("-" * 60)
    
    def _print_insights(self, insights):
        """Print extracted insights"""
        print("\n[INSIGHTS SUMMARY]")
        print("=" * 60)
        
        # Summary
        summary = insights.get('summary', {})
        print(f"\n[DATA] Analyzed:")
        print(f"   Reach: {summary.get('total_reach', 0):,} followers")
        print(f"   Tweets: {summary.get('tweets_analyzed', 0)}")
        print(f"   Mentions: {summary.get('mentions_analyzed', 0)}")
        print(f"   Follower Sample: {summary.get('followers_analyzed', 0)}")
        
        # Sentiment insights
        sent_insights = insights.get('sentiment_insights', {})
        if sent_insights.get('public_perception'):
            perc = sent_insights['public_perception']
            print(f"\n[SENTIMENT] Public Perception:")
            print(f"   Dominant: {perc['dominant_sentiment'].upper()}")
            print(f"   Sentiment Score: {perc['sentiment_score']:.1f}%")
            print(f"   Positive: {perc['positive_rate']:.1f}% | Negative: {perc['negative_rate']:.1f}%")
        
        # Engagement insights
        eng_insights = insights.get('engagement_insights', {})
        if eng_insights:
            print(f"\n[ENGAGEMENT] Performance:")
            print(f"   Avg per tweet: {eng_insights.get('avg_engagement_per_tweet', 0):.1f}")
            print(f"   Best metric: {eng_insights.get('best_performing_metric', 'N/A')}")
            print(f"   Top tweet: {eng_insights.get('top_tweet_engagement', 0):,} engagements")
        
        # Follower insights
        follower_insights = insights.get('follower_insights', {})
        if follower_insights.get('geography'):
            geo = follower_insights['geography']
            print(f"\n[GEOGRAPHY] Follower Distribution:")
            print(f"   Lebanon: {geo['lebanon_percentage']:.1f}% | Diaspora: {geo['diaspora_percentage']:.1f}%")
            if geo['top_location']:
                print(f"   Top Location: {geo['top_location'][0]} ({geo['top_location'][1]} followers)")
        
        if follower_insights.get('demographics'):
            demo = follower_insights['demographics']
            print(f"\n[DEMOGRAPHICS] Follower Quality:")
            print(f"   Verified: {demo['verified_percentage']:.1f}%")
            print(f"   Influential followers: {demo['influential_count']}")
            print(f"   Avg account age: {demo['avg_account_age']:.1f} years")
        
        # Recommendations
        recommendations = insights.get('recommendations', [])
        if recommendations:
            print(f"\n[RECOMMENDATIONS] Actionable Insights:")
            for i, rec in enumerate(recommendations[:5], 1):
                print(f"\n   {i}. [{rec['priority']}] {rec['category']}")
                print(f"      Finding: {rec['finding']}")
                print(f"      Action: {rec['action']}")
        
        print("\n" + "=" * 60)

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  Mark Daou Comprehensive Political Data Fetcher")
    print("  Using Multiple X API v2 Endpoints")
    print("="*60 + "\n")
    
    # Try to get credentials from environment variables
    # Support multiple variable name formats
    # Note: Check specific variables first to avoid conflicts with system variables
    bearer_token = (os.getenv('BEARER_TOKEN') or 
                    os.getenv('X_BEARER_TOKEN') or 
                    os.getenv('TWITTER_BEARER_TOKEN'))
    username = (os.getenv('POLITICIAN_USERNAME') or 
                os.getenv('X_USERNAME') or 
                os.getenv('TARGET_USERNAME'))
    
    # If not in .env, prompt for input
    if not bearer_token:
        bearer_token = input("Enter your X API Bearer Token: ").strip()
    else:
        print(f"[INFO] Using Bearer Token from .env file")
    
    if not bearer_token:
        print("\n[ERROR] Bearer Token is required")
        print("\n[INFO] See GET_API_CREDENTIALS.md for instructions on getting your token")
        print("\n[TIP] You can add TWITTER_BEARER_TOKEN=your_token to your .env file to skip this prompt")
        exit(1)
    
    if not username:
        username = input(f"Enter Mark Daou's X username (e.g., markdaou_lb): ").strip()
    else:
        print(f"[INFO] Using username from .env file: @{username}")
    
    if not username:
        print("\n[ERROR] Username is required")
        print("\n[TIP] You can add POLITICIAN_USERNAME=DaouMark to your .env file to skip this prompt")
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

