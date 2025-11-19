"""
ADVANCED Political Data Fetcher - Mark Daou
- Uses ALL available X API v2 endpoints
- Real AI sentiment analysis with transformers
- Deep topic modeling and analysis
- Comprehensive hashtag and audience analysis
- NO SAMPLE DATA - 100% REAL API DATA
"""
import os
import requests
import json
from datetime import datetime, timedelta, timezone
import time
from collections import Counter
import re
from dotenv import load_dotenv

# AI/ML Libraries for sentiment analysis
from transformers import pipeline
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import warnings
warnings.filterwarnings('ignore')

class AdvancedPoliticalDataFetcher:
    def __init__(self, bearer_token):
        self.bearer_token = bearer_token
        self.headers = {
            'Authorization': f'Bearer {bearer_token}',
            'Content-Type': 'application/json'
        }
        self.politician_name = "Mark Daou"
        self.politician_username = None
        self.politician_id = None
        
        # Initialize AI sentiment analyzers
        print("[AI] Initializing sentiment analysis models...")
        try:
            # Use multilingual model for Arabic/English
            self.sentiment_model = pipeline(
                "sentiment-analysis",
                model="nlptown/bert-base-multilingual-uncased-sentiment",
                top_k=None
            )
            print("[OK] Multilingual BERT sentiment model loaded")
        except Exception as e:
            print(f"[WARN] Could not load transformer model: {e}")
            self.sentiment_model = None
        
        self.vader_analyzer = SentimentIntensityAnalyzer()
        print("[OK] VADER sentiment analyzer loaded")
        print()
        
    def _make_request(self, url, params=None, method='GET'):
        """Make API request with error handling"""
        try:
            if method == 'GET':
                response = requests.get(url, headers=self.headers, params=params)
            else:
                response = requests.post(url, headers=self.headers, json=params)
            
            if response.status_code == 429:
                reset_time = int(response.headers.get('x-rate-limit-reset', time.time() + 900))
                wait_time = reset_time - int(time.time())
                print(f"[WAIT] Rate limit hit. Waiting {wait_time} seconds...")
                time.sleep(wait_time + 1)
                return self._make_request(url, params, method)
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"[ERROR] Error {response.status_code}: {response.text[:200]}")
                return None
                
        except Exception as e:
            print(f"[ERROR] Request failed: {e}")
            return None
    
    def analyze_sentiment_ai(self, text):
        """Advanced AI-powered sentiment analysis"""
        if not text or len(text.strip()) < 3:
            return {'method': 'none', 'sentiment': 'neutral', 'score': 0.0}
        
        results = {}
        
        # 1. VADER (good for social media)
        vader_scores = self.vader_analyzer.polarity_scores(text)
        vader_compound = vader_scores['compound']
        
        if vader_compound >= 0.05:
            vader_sentiment = 'positive'
        elif vader_compound <= -0.05:
            vader_sentiment = 'negative'
        else:
            vader_sentiment = 'neutral'
        
        results['vader'] = {
            'sentiment': vader_sentiment,
            'score': vader_compound,
            'scores': vader_scores
        }
        
        # 2. TextBlob
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            
            if polarity > 0.1:
                textblob_sentiment = 'positive'
            elif polarity < -0.1:
                textblob_sentiment = 'negative'
            else:
                textblob_sentiment = 'neutral'
            
            results['textblob'] = {
                'sentiment': textblob_sentiment,
                'score': polarity,
                'subjectivity': blob.sentiment.subjectivity
            }
        except:
            results['textblob'] = {'sentiment': 'neutral', 'score': 0.0}
        
        # 3. Transformer model (multilingual BERT)
        if self.sentiment_model:
            try:
                # Truncate text for model
                text_truncated = text[:512]
                bert_result = self.sentiment_model(text_truncated)
                
                if bert_result and len(bert_result) > 0:
                    # Model returns ratings 1-5 stars
                    top_result = max(bert_result[0], key=lambda x: x['score'])
                    stars = int(top_result['label'].split()[0])
                    
                    if stars >= 4:
                        bert_sentiment = 'positive'
                    elif stars <= 2:
                        bert_sentiment = 'negative'
                    else:
                        bert_sentiment = 'neutral'
                    
                    results['bert'] = {
                        'sentiment': bert_sentiment,
                        'stars': stars,
                        'confidence': top_result['score'],
                        'all_scores': bert_result[0]
                    }
            except Exception as e:
                results['bert'] = {'sentiment': 'neutral', 'error': str(e)}
        
        # 4. Aggregate sentiment (weighted voting)
        sentiments = []
        if 'vader' in results:
            sentiments.append(results['vader']['sentiment'])
        if 'textblob' in results:
            sentiments.append(results['textblob']['sentiment'])
        if 'bert' in results:
            sentiments.append(results['bert']['sentiment'])
            sentiments.append(results['bert']['sentiment'])  # Double weight for BERT
        
        if sentiments:
            sentiment_counts = Counter(sentiments)
            final_sentiment = sentiment_counts.most_common(1)[0][0]
        else:
            final_sentiment = 'neutral'
        
        results['final'] = {
            'sentiment': final_sentiment,
            'methods_used': list(results.keys()),
            'confidence': 'high' if len(set(sentiments)) == 1 else 'medium'
        }
        
        return results
    
    def find_user(self, username):
        """Find user with ALL available fields"""
        print(f"[SEARCH] Looking up user: @{username}")
        url = f"https://api.twitter.com/2/users/by/username/{username}"
        params = {
            'user.fields': 'created_at,description,entities,id,location,name,pinned_tweet_id,profile_image_url,protected,public_metrics,url,username,verified,verified_type,withheld'
        }
        result = self._make_request(url, params)
        if result and 'data' in result:
            self.politician_username = result['data']['username']
            self.politician_id = result['data']['id']
            return result
        return None
    
    def get_user_tweets(self, max_results=100):
        """Get tweets with MAXIMUM fields"""
        if not self.politician_id:
            return None
        
        print(f"[TWEETS] Fetching {max_results} tweets with full data...")
        url = f"https://api.twitter.com/2/users/{self.politician_id}/tweets"
        params = {
            'max_results': min(max_results, 100),
            'tweet.fields': 'attachments,author_id,context_annotations,conversation_id,created_at,edit_controls,entities,geo,id,in_reply_to_user_id,lang,possibly_sensitive,public_metrics,referenced_tweets,reply_settings,source,text,withheld',
            'expansions': 'author_id,referenced_tweets.id,in_reply_to_user_id,attachments.media_keys,geo.place_id,entities.mentions.username,referenced_tweets.id.author_id',
            'media.fields': 'alt_text,duration_ms,height,media_key,preview_image_url,type,url,width,public_metrics',
            'place.fields': 'contained_within,country,country_code,full_name,geo,id,name,place_type',
            'user.fields': 'created_at,description,entities,id,location,name,pinned_tweet_id,profile_image_url,protected,public_metrics,url,username,verified'
        }
        return self._make_request(url, params)
    
    def get_mentions(self, max_results=100):
        """Get mentions with full context"""
        if not self.politician_id:
            return None
        
        print(f"[MENTIONS] Fetching {max_results} mentions...")
        url = f"https://api.twitter.com/2/users/{self.politician_id}/mentions"
        params = {
            'max_results': min(max_results, 100),
            'tweet.fields': 'attachments,author_id,context_annotations,conversation_id,created_at,entities,geo,id,in_reply_to_user_id,lang,possibly_sensitive,public_metrics,referenced_tweets,reply_settings,source,text',
            'expansions': 'author_id,referenced_tweets.id,in_reply_to_user_id,geo.place_id',
            'user.fields': 'created_at,description,location,name,public_metrics,username,verified',
            'place.fields': 'country,country_code,full_name,geo,name,place_type'
        }
        return self._make_request(url, params)
    
    def search_recent_tweets(self, query=None, max_results=100):
        """Enhanced search with geo and context"""
        search_query = query or f'("{self.politician_name}" OR @{self.politician_username}) (Lebanon OR لبنان)'
        print(f"[SEARCH] Searching recent tweets...")
        
        url = "https://api.twitter.com/2/tweets/search/recent"
        params = {
            'query': search_query,
            'max_results': min(max_results, 100),
            'tweet.fields': 'attachments,author_id,context_annotations,conversation_id,created_at,entities,geo,id,lang,possibly_sensitive,public_metrics,referenced_tweets,reply_settings,text',
            'expansions': 'author_id,geo.place_id,entities.mentions.username',
            'user.fields': 'created_at,description,location,name,public_metrics,username,verified',
            'place.fields': 'country,country_code,full_name,name,place_type'
        }
        return self._make_request(url, params)
    
    def get_followers(self, max_results=1000):
        """Get detailed follower data"""
        if not self.politician_id:
            return None
        
        print(f"[USERS] Fetching {max_results} followers...")
        url = f"https://api.twitter.com/2/users/{self.politician_id}/followers"
        params = {
            'max_results': min(max_results, 1000),
            'user.fields': 'created_at,description,entities,location,name,pinned_tweet_id,profile_image_url,public_metrics,url,username,verified,verified_type'
        }
        return self._make_request(url, params)
    
    def get_following(self, max_results=1000):
        """Get detailed following data"""
        if not self.politician_id:
            return None
        
        print(f"[FOLLOWING] Fetching {max_results} following accounts...")
        url = f"https://api.twitter.com/2/users/{self.politician_id}/following"
        params = {
            'max_results': min(max_results, 1000),
            'user.fields': 'created_at,description,location,name,public_metrics,username,verified,verified_type'
        }
        return self._make_request(url, params)
    
    def get_tweet_liking_users(self, tweet_id, max_results=100):
        """Get users who liked a specific tweet"""
        print(f"[LIKES] Fetching liking users for tweet {tweet_id}...")
        url = f"https://api.twitter.com/2/tweets/{tweet_id}/liking_users"
        params = {
            'max_results': min(max_results, 100),
            'user.fields': 'created_at,description,location,public_metrics,verified'
        }
        return self._make_request(url, params)
    
    def get_tweet_retweeted_by(self, tweet_id, max_results=100):
        """Get users who retweeted a specific tweet"""
        print(f"[RT] Fetching retweet users for tweet {tweet_id}...")
        url = f"https://api.twitter.com/2/tweets/{tweet_id}/retweeted_by"
        params = {
            'max_results': min(max_results, 100),
            'user.fields': 'created_at,description,location,public_metrics,verified'
        }
        return self._make_request(url, params)
    
    def get_quote_tweets(self, tweet_id, max_results=100):
        """Get quote tweets of a specific tweet"""
        print(f"[QUOTES] Fetching quote tweets for tweet {tweet_id}...")
        url = f"https://api.twitter.com/2/tweets/{tweet_id}/quote_tweets"
        params = {
            'max_results': min(max_results, 100),
            'tweet.fields': 'author_id,created_at,public_metrics,text,context_annotations',
            'expansions': 'author_id',
            'user.fields': 'name,username,verified,public_metrics'
        }
        return self._make_request(url, params)
    
    def analyze_tweets_with_ai(self, tweets_data):
        """Analyze all tweets with AI sentiment analysis"""
        if not tweets_data or 'data' not in tweets_data:
            return []
        
        print(f"[AI] Running AI sentiment analysis on {len(tweets_data['data'])} tweets...")
        
        analyzed_tweets = []
        for i, tweet in enumerate(tweets_data['data']):
            text = tweet.get('text', '')
            
            # Run AI sentiment analysis
            sentiment_analysis = self.analyze_sentiment_ai(text)
            
            analyzed_tweet = {
                **tweet,
                'ai_sentiment': sentiment_analysis
            }
            analyzed_tweets.append(analyzed_tweet)
            
            if (i + 1) % 10 == 0:
                print(f"   Analyzed {i + 1}/{len(tweets_data['data'])} tweets...")
        
        print(f"[OK] AI sentiment analysis complete\n")
        return analyzed_tweets
    
    def extract_advanced_topics(self, tweets_data):
        """Advanced topic modeling using LDA"""
        if not tweets_data or 'data' not in tweets_data:
            return {}
        
        print("[ANALYSIS] Performing advanced topic modeling...")
        
        texts = [tweet.get('text', '') for tweet in tweets_data['data'] if tweet.get('text')]
        
        if len(texts) < 5:
            return {}
        
        try:
            # TF-IDF Vectorization
            vectorizer = TfidfVectorizer(
                max_features=100,
                stop_words='english',
                ngram_range=(1, 2)
            )
            tfidf = vectorizer.fit_transform(texts)
            
            # LDA Topic Modeling
            n_topics = min(5, len(texts) // 3)
            lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
            lda.fit(tfidf)
            
            # Extract topics
            feature_names = vectorizer.get_feature_names_out()
            topics = {}
            
            for topic_idx, topic in enumerate(lda.components_):
                top_indices = topic.argsort()[-10:][::-1]
                top_words = [feature_names[i] for i in top_indices]
                topics[f"Topic_{topic_idx + 1}"] = {
                    'keywords': top_words[:5],
                    'all_words': top_words
                }
            
            print(f"[OK] Identified {len(topics)} topics\n")
            return topics
            
        except Exception as e:
            print(f"[WARN] Topic modeling failed: {e}\n")
            return {}
    
    def analyze_hashtag_sentiment(self, tweets_data):
        """Analyze sentiment per hashtag"""
        if not tweets_data or 'data' not in tweets_data:
            return {}
        
        print("[TAGS] Analyzing hashtag sentiment...")
        
        hashtag_sentiments = {}
        
        for tweet in tweets_data['data']:
            if 'entities' in tweet and 'hashtags' in tweet['entities']:
                text = tweet.get('text', '')
                sentiment = self.analyze_sentiment_ai(text)
                
                for hashtag in tweet['entities']['hashtags']:
                    tag = hashtag['tag'].lower()
                    
                    if tag not in hashtag_sentiments:
                        hashtag_sentiments[tag] = {
                            'count': 0,
                            'positive': 0,
                            'negative': 0,
                            'neutral': 0,
                            'avg_engagement': 0,
                            'total_engagement': 0
                        }
                    
                    hashtag_sentiments[tag]['count'] += 1
                    
                    final_sent = sentiment.get('final', {}).get('sentiment', 'neutral')
                    hashtag_sentiments[tag][final_sent] += 1
                    
                    if 'public_metrics' in tweet:
                        pm = tweet['public_metrics']
                        engagement = pm.get('like_count', 0) + pm.get('retweet_count', 0)
                        hashtag_sentiments[tag]['total_engagement'] += engagement
        
        # Calculate averages
        for tag in hashtag_sentiments:
            count = hashtag_sentiments[tag]['count']
            if count > 0:
                hashtag_sentiments[tag]['avg_engagement'] = (
                    hashtag_sentiments[tag]['total_engagement'] / count
                )
        
        print(f"[OK] Analyzed {len(hashtag_sentiments)} unique hashtags\n")
        return hashtag_sentiments
    
    def analyze_audience_engagement(self, tweets_data, mentions_data):
        """Analyze how audience engages with topics"""
        print("[USERS] Analyzing audience engagement patterns...")
        
        analysis = {
            'most_engaging_topics': {},
            'most_replied_to': [],
            'most_controversial': [],
            'viral_potential': []
        }
        
        if tweets_data and 'data' in tweets_data:
            tweets = tweets_data['data']
            
            # Sort by engagement
            for tweet in tweets:
                if 'public_metrics' in tweet and 'text' in tweet:
                    pm = tweet['public_metrics']
                    engagement = (
                        pm.get('like_count', 0) +
                        pm.get('retweet_count', 0) * 2 +
                        pm.get('reply_count', 0) * 3
                    )
                    
                    tweet_info = {
                        'text': tweet['text'][:100],
                        'engagement': engagement,
                        'likes': pm.get('like_count', 0),
                        'retweets': pm.get('retweet_count', 0),
                        'replies': pm.get('reply_count', 0)
                    }
                    
                    # Most replied (discussion drivers)
                    if pm.get('reply_count', 0) > 10:
                        analysis['most_replied_to'].append(tweet_info)
                    
                    # Controversial (high replies but low likes)
                    reply_ratio = pm.get('reply_count', 0) / max(pm.get('like_count', 1), 1)
                    if reply_ratio > 0.5:
                        analysis['most_controversial'].append({
                            **tweet_info,
                            'controversy_score': reply_ratio
                        })
                    
                    # Viral potential (high retweets)
                    if pm.get('retweet_count', 0) > 20:
                        analysis['viral_potential'].append(tweet_info)
            
            # Sort lists
            analysis['most_replied_to'] = sorted(
                analysis['most_replied_to'],
                key=lambda x: x['replies'],
                reverse=True
            )[:10]
            
            analysis['most_controversial'] = sorted(
                analysis['most_controversial'],
                key=lambda x: x['controversy_score'],
                reverse=True
            )[:10]
            
            analysis['viral_potential'] = sorted(
                analysis['viral_potential'],
                key=lambda x: x['retweets'],
                reverse=True
            )[:10]
        
        print(f"[OK] Audience engagement analysis complete\n")
        return analysis
    
    def fetch_comprehensive_data(self, username):
        """Fetch ALL data with AI analysis"""
        print("\n" + "="*70)
        print(f"  ADVANCED Political Intelligence System - {self.politician_name}")
        print(f"  AI-Powered Sentiment Analysis & Deep Topic Modeling")
        print("="*70 + "\n")
        
        data = {
            'timestamp': datetime.now().isoformat(),
            'politician': self.politician_name,
            'username': username,
            'user_profile': None,
            'user_tweets': None,
            'user_tweets_analyzed': None,
            'mentions': None,
            'mentions_analyzed': None,
            'search_results': None,
            'search_results_analyzed': None,
            'followers_sample': None,
            'following_sample': None,
            'top_tweet_engagement': {},
            'ai_sentiment_summary': {},
            'advanced_topics': {},
            'hashtag_sentiment_analysis': {},
            'audience_engagement_analysis': {},
            'geographic_distribution': {},
            'campaign_metrics': {}
        }
        
        # 1. Get user profile
        user_data = self.find_user(username)
        if not user_data:
            print("[ERROR] Could not find user")
            return None
        
        data['user_profile'] = user_data
        print(f"[OK] Profile loaded\n")
        
        # 2. Get tweets with full data
        data['user_tweets'] = self.get_user_tweets(max_results=100)
        
        # 3. AI analyze tweets
        if data['user_tweets']:
            data['user_tweets_analyzed'] = self.analyze_tweets_with_ai(data['user_tweets'])
        
        # 4. Get mentions
        data['mentions'] = self.get_mentions(max_results=100)
        
        # 5. AI analyze mentions
        if data['mentions']:
            data['mentions_analyzed'] = self.analyze_tweets_with_ai(data['mentions'])
        
        # 6. Search tweets
        data['search_results'] = self.search_recent_tweets(max_results=100)
        
        # 7. AI analyze search results
        if data['search_results']:
            data['search_results_analyzed'] = self.analyze_tweets_with_ai(data['search_results'])
        
        # 8. Get followers
        data['followers_sample'] = self.get_followers(max_results=1000)
        
        # 9. Get following
        data['following_sample'] = self.get_following(max_results=1000)
        
        # 10. Get engagement on top tweets
        if data['user_tweets'] and 'data' in data['user_tweets']:
            top_tweets = sorted(
                data['user_tweets']['data'],
                key=lambda x: x.get('public_metrics', {}).get('like_count', 0),
                reverse=True
            )[:3]
            
            for tweet in top_tweets:
                tweet_id = tweet['id']
                print(f"\n[STATS] Analyzing engagement for top tweet {tweet_id}...")
                
                data['top_tweet_engagement'][tweet_id] = {
                    'tweet_text': tweet.get('text', '')[:100],
                    'metrics': tweet.get('public_metrics', {}),
                    'liking_users': self.get_tweet_liking_users(tweet_id, 100),
                    'retweeting_users': self.get_tweet_retweeted_by(tweet_id, 100),
                    'quote_tweets': self.get_quote_tweets(tweet_id, 100)
                }
                time.sleep(1)  # Rate limit spacing
        
        # 11. Advanced topic modeling
        if data['user_tweets']:
            data['advanced_topics'] = self.extract_advanced_topics(data['user_tweets'])
        
        # 12. Hashtag sentiment analysis
        combined_tweets = {'data': []}
        if data['user_tweets'] and 'data' in data['user_tweets']:
            combined_tweets['data'].extend(data['user_tweets']['data'])
        if data['mentions'] and 'data' in data['mentions']:
            combined_tweets['data'].extend(data['mentions']['data'])
        
        data['hashtag_sentiment_analysis'] = self.analyze_hashtag_sentiment(combined_tweets)
        
        # 13. Audience engagement analysis
        data['audience_engagement_analysis'] = self.analyze_audience_engagement(
            data['user_tweets'],
            data['mentions']
        )
        
        # 14. AI sentiment summary
        print("[STATS] Generating AI sentiment summary...")
        sentiment_summary = {
            'own_tweets': {'positive': 0, 'negative': 0, 'neutral': 0},
            'mentions': {'positive': 0, 'negative': 0, 'neutral': 0},
            'search_results': {'positive': 0, 'negative': 0, 'neutral': 0}
        }
        
        if data['user_tweets_analyzed']:
            for tweet in data['user_tweets_analyzed']:
                sent = tweet.get('ai_sentiment', {}).get('final', {}).get('sentiment', 'neutral')
                sentiment_summary['own_tweets'][sent] += 1
        
        if data['mentions_analyzed']:
            for tweet in data['mentions_analyzed']:
                sent = tweet.get('ai_sentiment', {}).get('final', {}).get('sentiment', 'neutral')
                sentiment_summary['mentions'][sent] += 1
        
        if data['search_results_analyzed']:
            for tweet in data['search_results_analyzed']:
                sent = tweet.get('ai_sentiment', {}).get('final', {}).get('sentiment', 'neutral')
                sentiment_summary['search_results'][sent] += 1
        
        data['ai_sentiment_summary'] = sentiment_summary
        print("[OK] AI sentiment summary complete\n")
        
        # Save to file
        filename = f'mark_daou_advanced_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print("\n" + "="*70)
        print(f"[OK] ADVANCED DATA COLLECTION COMPLETE!")
        print(f"[FILE] Saved to: {filename}")
        print("="*70 + "\n")
        
        self._print_advanced_summary(data)
        
        return data
    
    def _print_advanced_summary(self, data):
        """Print comprehensive summary"""
        print("\n[STATS] ADVANCED ANALYSIS SUMMARY")
        print("-" * 70)
        
        if data['ai_sentiment_summary']:
            print("\n[AI] AI SENTIMENT ANALYSIS:")
            for category, sentiments in data['ai_sentiment_summary'].items():
                total = sum(sentiments.values())
                if total > 0:
                    print(f"\n{category.replace('_', ' ').title()}:")
                    print(f"  Positive: {sentiments['positive']} ({sentiments['positive']/total*100:.1f}%)")
                    print(f"  Negative: {sentiments['negative']} ({sentiments['negative']/total*100:.1f}%)")
                    print(f"  Neutral:  {sentiments['neutral']} ({sentiments['neutral']/total*100:.1f}%)")
        
        if data['advanced_topics']:
            print(f"\n[TOPICS] TOPICS IDENTIFIED: {len(data['advanced_topics'])}")
            for topic_name, topic_data in list(data['advanced_topics'].items())[:3]:
                print(f"  {topic_name}: {', '.join(topic_data['keywords'])}")
        
        if data['hashtag_sentiment_analysis']:
            top_hashtags = sorted(
                data['hashtag_sentiment_analysis'].items(),
                key=lambda x: x[1]['count'],
                reverse=True
            )[:5]
            
            print(f"\n[TAGS] TOP HASHTAGS WITH SENTIMENT:")
            for tag, data_dict in top_hashtags:
                print(f"  #{tag}: {data_dict['count']} uses, "
                      f"Pos:{data_dict['positive']} Neg:{data_dict['negative']}")
        
        if data['audience_engagement_analysis']:
            engagement = data['audience_engagement_analysis']
            
            if engagement.get('most_replied_to'):
                print(f"\n[DISCUSS] MOST DISCUSSED: {len(engagement['most_replied_to'])} tweets")
            
            if engagement.get('most_controversial'):
                print(f"[HOT] CONTROVERSIAL: {len(engagement['most_controversial'])} tweets")
            
            if engagement.get('viral_potential'):
                print(f"[VIRAL] VIRAL POTENTIAL: {len(engagement['viral_potential'])} tweets")
        
        print("\n" + "-" * 70)

if __name__ == "__main__":
    # Load credentials
    load_dotenv()
    
    bearer_token = os.getenv('TWITTER_BEARER_TOKEN')
    username = os.getenv('POLITICIAN_USERNAME', 'DaouMark')
    
    if not bearer_token:
        print("[ERROR] TWITTER_BEARER_TOKEN not found in .env file")
        exit(1)
    
    print("[AUTH] Using credentials from .env")
    print(f"[TARGET] @{username}\n")
    
    # Run advanced fetcher
    fetcher = AdvancedPoliticalDataFetcher(bearer_token)
    data = fetcher.fetch_comprehensive_data(username)
    
    if data:
        print("\n[SUCCESS] Run dashboard:")
        print("   streamlit run enhanced_dashboard.py")
    else:
        print("\n[FAILED]")

