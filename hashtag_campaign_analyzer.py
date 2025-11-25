"""
Hashtag Campaign Analyzer - Mark Daou
Search and analyze tweets by hashtag for campaign tracking
Limited to 100 API calls per run
"""
import requests
import json
from datetime import datetime, timezone
from collections import Counter
import time

class HashtagCampaignAnalyzer:
    def __init__(self, bearer_token):
        self.bearer_token = bearer_token
        self.headers = {
            'Authorization': f'Bearer {bearer_token}',
            'Content-Type': 'application/json'
        }
    
    def _make_request(self, url, params=None):
        """Make API request with error handling"""
        try:
            response = requests.get(url, headers=self.headers, params=params)
            
            if response.status_code == 429:
                reset_time = int(response.headers.get('x-rate-limit-reset', time.time() + 900))
                wait_time = reset_time - int(time.time())
                print(f"[WAIT] Rate limit hit. Waiting {wait_time} seconds...")
                time.sleep(wait_time + 1)
                return self._make_request(url, params)
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"[ERROR] {response.status_code}: {response.text[:200]}")
                return None
                
        except Exception as e:
            print(f"[ERROR] Request failed: {e}")
            return None
    
    def search_hashtag(self, hashtag, max_results=100):
        """Search for tweets containing a specific hashtag"""
        # Remove # if included
        hashtag = hashtag.lstrip('#')
        
        print(f"[SEARCH] Searching for #{hashtag}...")
        
        url = "https://api.twitter.com/2/tweets/search/recent"
        params = {
            'query': f'#{hashtag} -is:retweet',  # Exclude retweets for original content
            'max_results': min(max_results, 100),
            'tweet.fields': 'created_at,public_metrics,entities,author_id,lang,possibly_sensitive,context_annotations',
            'expansions': 'author_id',
            'user.fields': 'name,username,verified,public_metrics,location,description'
        }
        
        result = self._make_request(url, params)
        if result:
            print(f"[SUCCESS] Found {len(result.get('data', []))} tweets\n")
        return result
    
    def analyze_campaign_performance(self, tweets_data, hashtag):
        """Analyze campaign performance metrics"""
        if not tweets_data or 'data' not in tweets_data:
            return {}
        
        tweets = tweets_data['data']
        users = {u['id']: u for u in tweets_data.get('includes', {}).get('users', [])}
        
        # Initialize metrics
        total_likes = 0
        total_retweets = 0
        total_replies = 0
        total_quotes = 0
        total_impressions = 0
        
        languages = []
        locations = []
        verified_users = 0
        unique_authors = set()
        
        tweet_details = []
        
        for tweet in tweets:
            # Engagement metrics
            pm = tweet.get('public_metrics', {})
            total_likes += pm.get('like_count', 0)
            total_retweets += pm.get('retweet_count', 0)
            total_replies += pm.get('reply_count', 0)
            total_quotes += pm.get('quote_count', 0)
            total_impressions += pm.get('impression_count', 0)
            
            # Language
            languages.append(tweet.get('lang', 'unknown'))
            
            # Author info
            author_id = tweet.get('author_id')
            unique_authors.add(author_id)
            
            if author_id in users:
                user = users[author_id]
                if user.get('verified'):
                    verified_users += 1
                if user.get('location'):
                    locations.append(user['location'])
            
            # Store for top tweets
            engagement = pm.get('like_count', 0) + pm.get('retweet_count', 0) * 2 + pm.get('reply_count', 0) * 3
            tweet_details.append({
                'text': tweet.get('text', '')[:100],
                'author': users.get(author_id, {}).get('username', 'unknown'),
                'likes': pm.get('like_count', 0),
                'retweets': pm.get('retweet_count', 0),
                'replies': pm.get('reply_count', 0),
                'total_engagement': engagement,
                'created_at': tweet.get('created_at', '')
            })
        
        # Sort tweets by engagement
        tweet_details.sort(key=lambda x: x['total_engagement'], reverse=True)
        
        total_tweets = len(tweets)
        
        return {
            'hashtag': hashtag,
            'total_tweets': total_tweets,
            'unique_authors': len(unique_authors),
            'verified_users': verified_users,
            'verified_percentage': (verified_users / total_tweets * 100) if total_tweets > 0 else 0,
            'engagement_metrics': {
                'total_likes': total_likes,
                'total_retweets': total_retweets,
                'total_replies': total_replies,
                'total_quotes': total_quotes,
                'total_impressions': total_impressions,
                'avg_likes': total_likes / total_tweets if total_tweets > 0 else 0,
                'avg_retweets': total_retweets / total_tweets if total_tweets > 0 else 0,
                'avg_replies': total_replies / total_tweets if total_tweets > 0 else 0,
                'total_engagement': total_likes + total_retweets + total_replies + total_quotes
            },
            'reach_metrics': {
                'total_impressions': total_impressions,
                'avg_impressions_per_tweet': total_impressions / total_tweets if total_tweets > 0 else 0,
                'potential_reach': sum(users.get(author_id, {}).get('public_metrics', {}).get('followers_count', 0) 
                                      for author_id in unique_authors)
            },
            'top_tweets': tweet_details[:10],
            'language_distribution': dict(Counter(languages).most_common(10)),
            'top_locations': dict(Counter(locations).most_common(10)) if locations else {}
        }
    
    def analyze_campaign_sentiment(self, tweets_data):
        """Basic sentiment analysis for campaign"""
        if not tweets_data or 'data' not in tweets_data:
            return {}
        
        positive_keywords = ['great', 'excellent', 'good', 'support', 'progress', 'success', 'hope', 'proud', 
                            'better', 'improvement', 'positive', 'forward', 'reform', 'change', 'future',
                            'رائع', 'ممتاز', 'جيد', 'دعم', 'تقدم', 'نجاح', 'أمل', 'فخور', 'أفضل']
        negative_keywords = ['bad', 'terrible', 'corrupt', 'fail', 'crisis', 'problem', 'worse', 'against',
                            'disaster', 'wrong', 'poor', 'decline', 'issue', 'concern', 'worry',
                            'سيء', 'فاسد', 'فشل', 'أزمة', 'مشكلة', 'أسوأ', 'ضد', 'كارثة']
        
        sentiments = {'positive': 0, 'negative': 0, 'neutral': 0}
        
        for tweet in tweets_data['data']:
            text = tweet.get('text', '').lower()
            pos_count = sum(1 for word in positive_keywords if word in text)
            neg_count = sum(1 for word in negative_keywords if word in text)
            
            if pos_count > neg_count:
                sentiments['positive'] += 1
            elif neg_count > pos_count:
                sentiments['negative'] += 1
            else:
                sentiments['neutral'] += 1
        
        total = sum(sentiments.values())
        
        return {
            'positive': sentiments['positive'],
            'negative': sentiments['negative'],
            'neutral': sentiments['neutral'],
            'positive_percentage': (sentiments['positive'] / total * 100) if total > 0 else 0,
            'negative_percentage': (sentiments['negative'] / total * 100) if total > 0 else 0,
            'neutral_percentage': (sentiments['neutral'] / total * 100) if total > 0 else 0,
            'sentiment_score': ((sentiments['positive'] - sentiments['negative']) / total * 100) if total > 0 else 0
        }
    
    def extract_campaign_insights(self, performance, sentiment):
        """Extract actionable campaign insights"""
        insights = {
            'campaign_health': 'Unknown',
            'key_metrics': {},
            'strengths': [],
            'weaknesses': [],
            'recommendations': []
        }
        
        # Calculate campaign health
        eng_rate = performance['engagement_metrics']['avg_likes'] + performance['engagement_metrics']['avg_retweets']
        sent_score = sentiment['sentiment_score']
        
        if eng_rate > 50 and sent_score > 20:
            insights['campaign_health'] = 'Excellent'
        elif eng_rate > 20 and sent_score > 0:
            insights['campaign_health'] = 'Good'
        elif eng_rate > 10 or sent_score > -20:
            insights['campaign_health'] = 'Moderate'
        else:
            insights['campaign_health'] = 'Needs Improvement'
        
        # Key metrics
        insights['key_metrics'] = {
            'virality_score': performance['engagement_metrics']['avg_retweets'],
            'conversation_score': performance['engagement_metrics']['avg_replies'],
            'sentiment_score': sent_score,
            'reach_potential': performance['reach_metrics']['potential_reach'],
            'author_diversity': performance['unique_authors']
        }
        
        # Strengths
        if sent_score > 20:
            insights['strengths'].append("Strong positive sentiment")
        if performance['verified_percentage'] > 10:
            insights['strengths'].append(f"High verified user participation ({performance['verified_percentage']:.1f}%)")
        if performance['unique_authors'] > 50:
            insights['strengths'].append("Wide audience participation")
        if performance['engagement_metrics']['avg_retweets'] > 5:
            insights['strengths'].append("High viral potential")
        
        # Weaknesses
        if sent_score < -10:
            insights['weaknesses'].append("Negative sentiment detected")
        if performance['engagement_metrics']['avg_likes'] < 5:
            insights['weaknesses'].append("Low engagement rate")
        if performance['unique_authors'] < 20:
            insights['weaknesses'].append("Limited reach - few unique authors")
        
        # Recommendations
        if sent_score < 0:
            insights['recommendations'].append({
                'priority': 'HIGH',
                'action': 'Address negative sentiment',
                'details': 'Monitor conversations, respond to concerns, shift messaging tone'
            })
        
        if performance['engagement_metrics']['avg_replies'] > performance['engagement_metrics']['avg_likes']:
            insights['recommendations'].append({
                'priority': 'MEDIUM',
                'action': 'High discussion activity',
                'details': 'Engage with conversations, moderate discussions, amplify positive voices'
            })
        
        if performance['reach_metrics']['potential_reach'] > 1000000:
            insights['recommendations'].append({
                'priority': 'OPPORTUNITY',
                'action': 'Large potential reach',
                'details': 'Engage influential participants, encourage amplification, create shareable content'
            })
        
        if performance['verified_percentage'] < 5:
            insights['recommendations'].append({
                'priority': 'MEDIUM',
                'action': 'Increase influencer participation',
                'details': 'Target verified accounts, reach out to media, engage thought leaders'
            })
        
        return insights
    
    def analyze_hashtag_campaign(self, hashtag):
        """Complete hashtag campaign analysis"""
        print("\n" + "="*60)
        print(f"  Hashtag Campaign Analyzer")
        print(f"  Analyzing: #{hashtag}")
        print("="*60 + "\n")
        
        # Search for hashtag tweets
        tweets_data = self.search_hashtag(hashtag, max_results=100)
        
        if not tweets_data or 'data' not in tweets_data:
            print("[ERROR] No tweets found or API error")
            return None
        
        # Analyze performance
        print("[ANALYSIS] Analyzing campaign performance...")
        performance = self.analyze_campaign_performance(tweets_data, hashtag)
        print("[SUCCESS] Performance analysis complete\n")
        
        # Analyze sentiment
        print("[ANALYSIS] Analyzing campaign sentiment...")
        sentiment = self.analyze_campaign_sentiment(tweets_data)
        print("[SUCCESS] Sentiment analysis complete\n")
        
        # Extract insights
        print("[INSIGHTS] Extracting campaign insights...")
        insights = self.extract_campaign_insights(performance, sentiment)
        print("[SUCCESS] Insights extraction complete\n")
        
        # Compile results
        results = {
            'timestamp': datetime.now().isoformat(),
            'hashtag': hashtag,
            'raw_data': tweets_data,
            'performance': performance,
            'sentiment': sentiment,
            'insights': insights
        }
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'campaign_{hashtag}_{timestamp}.json'
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print("="*60)
        print(f"[SUCCESS] Campaign analysis complete!")
        print(f"[FILE] Saved to: {filename}")
        print("="*60 + "\n")
        
        self._print_campaign_summary(results)
        
        return results
    
    def _print_campaign_summary(self, results):
        """Print campaign analysis summary"""
        performance = results['performance']
        sentiment = results['sentiment']
        insights = results['insights']
        
        print("\n[CAMPAIGN SUMMARY]")
        print("-" * 60)
        
        print(f"\n[OVERVIEW] #{results['hashtag']}")
        print(f"   Total Tweets: {performance['total_tweets']}")
        print(f"   Unique Authors: {performance['unique_authors']}")
        print(f"   Verified Users: {performance['verified_users']} ({performance['verified_percentage']:.1f}%)")
        print(f"   Campaign Health: {insights['campaign_health']}")
        
        print(f"\n[ENGAGEMENT]")
        eng = performance['engagement_metrics']
        print(f"   Total: {eng['total_engagement']:,}")
        print(f"   Avg Likes: {eng['avg_likes']:.1f}")
        print(f"   Avg Retweets: {eng['avg_retweets']:.1f}")
        print(f"   Avg Replies: {eng['avg_replies']:.1f}")
        
        print(f"\n[REACH]")
        reach = performance['reach_metrics']
        print(f"   Total Impressions: {reach['total_impressions']:,}")
        print(f"   Potential Reach: {reach['potential_reach']:,} followers")
        
        print(f"\n[SENTIMENT]")
        print(f"   Score: {sentiment['sentiment_score']:.1f}%")
        print(f"   Positive: {sentiment['positive']} ({sentiment['positive_percentage']:.1f}%)")
        print(f"   Negative: {sentiment['negative']} ({sentiment['negative_percentage']:.1f}%)")
        print(f"   Neutral: {sentiment['neutral']} ({sentiment['neutral_percentage']:.1f}%)")
        
        if insights['strengths']:
            print(f"\n[STRENGTHS]")
            for strength in insights['strengths']:
                print(f"   ✓ {strength}")
        
        if insights['weaknesses']:
            print(f"\n[WEAKNESSES]")
            for weakness in insights['weaknesses']:
                print(f"   ✗ {weakness}")
        
        if insights['recommendations']:
            print(f"\n[RECOMMENDATIONS]")
            for rec in insights['recommendations']:
                print(f"   [{rec['priority']}] {rec['action']}")
                print(f"      → {rec['details']}")
        
        print(f"\n[TOP TWEETS]")
        for i, tweet in enumerate(performance['top_tweets'][:3], 1):
            print(f"   {i}. @{tweet['author']}: {tweet['text']}...")
            print(f"      {tweet['total_engagement']:,} engagements")
        
        print("\n" + "-" * 60)

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  Hashtag Campaign Analyzer for Mark Daou")
    print("  Track campaign hashtags and extract insights")
    print("="*60 + "\n")
    
    bearer_token = input("Enter your X API Bearer Token: ").strip()
    
    if not bearer_token:
        print("\n[ERROR] Bearer Token is required")
        exit(1)
    
    hashtag = input("Enter hashtag to analyze (without #): ").strip()
    
    if not hashtag:
        print("\n[ERROR] Hashtag is required")
        exit(1)
    
    analyzer = HashtagCampaignAnalyzer(bearer_token)
    results = analyzer.analyze_hashtag_campaign(hashtag)
    
    if results:
        print("\n[SUCCESS] Analysis complete!")
        print("\nView dashboard: streamlit run enhanced_dashboard.py")

