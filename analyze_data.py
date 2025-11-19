"""
Phase 2: Analyze collected data with AI
Load raw JSON data and run:
- AI sentiment analysis (BERT + VADER + TextBlob)
- Topic modeling (LDA)
- Hashtag sentiment analysis
- Audience engagement analysis
"""
import sys
import json
from datetime import datetime
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# AI/ML imports
from transformers import pipeline
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

class DataAnalyzer:
    """Analyze raw data with AI models"""
    
    def __init__(self):
        print("\n" + "="*70)
        print("  PHASE 2: AI-POWERED DATA ANALYSIS")
        print("  Loading AI models...")
        print("="*70 + "\n")
        
        # Initialize AI models
        print("[AI] Loading multilingual BERT sentiment model...")
        try:
            self.sentiment_model = pipeline(
                "sentiment-analysis",
                model="nlptown/bert-base-multilingual-uncased-sentiment",
                top_k=None
            )
            print("[OK] BERT model loaded")
        except:
            self.sentiment_model = None
            print("[WARN] Could not load BERT model")
        
        print("[AI] Loading VADER sentiment analyzer...")
        self.vader = SentimentIntensityAnalyzer()
        print("[OK] VADER loaded\n")
    
    def analyze_sentiment_ai(self, text):
        """Multi-model AI sentiment analysis"""
        if not text or len(text.strip()) < 3:
            return {'sentiment': 'neutral', 'confidence': 0.0, 'scores': {}}
        
        results = {}
        
        # VADER (best for social media)
        vader_scores = self.vader.polarity_scores(text)
        vader_compound = vader_scores['compound']
        
        if vader_compound >= 0.05:
            vader_sent = 'positive'
        elif vader_compound <= -0.05:
            vader_sent = 'negative'
        else:
            vader_sent = 'neutral'
        
        results['vader'] = {
            'sentiment': vader_sent,
            'score': vader_compound,
            'detailed': vader_scores
        }
        
        # TextBlob
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            
            if polarity > 0.1:
                tb_sent = 'positive'
            elif polarity < -0.1:
                tb_sent = 'negative'
            else:
                tb_sent = 'neutral'
            
            results['textblob'] = {
                'sentiment': tb_sent,
                'polarity': polarity,
                'subjectivity': blob.sentiment.subjectivity
            }
        except:
            results['textblob'] = {'sentiment': 'neutral', 'polarity': 0.0}
        
        # BERT (multilingual - best for Arabic/English)
        if self.sentiment_model:
            try:
                bert_result = self.sentiment_model(text[:512])
                if bert_result and len(bert_result) > 0:
                    top = max(bert_result[0], key=lambda x: x['score'])
                    stars = int(top['label'].split()[0])
                    
                    if stars >= 4:
                        bert_sent = 'positive'
                    elif stars <= 2:
                        bert_sent = 'negative'
                    else:
                        bert_sent = 'neutral'
                    
                    results['bert'] = {
                        'sentiment': bert_sent,
                        'stars': stars,
                        'confidence': top['score']
                    }
            except:
                pass
        
        # Aggregate (weighted voting)
        sentiments = []
        sentiments.append(results['vader']['sentiment'])
        sentiments.append(results['textblob']['sentiment'])
        if 'bert' in results:
            sentiments.append(results['bert']['sentiment'])
            sentiments.append(results['bert']['sentiment'])  # Double weight
        
        sentiment_counts = Counter(sentiments)
        final_sentiment = sentiment_counts.most_common(1)[0][0]
        confidence = sentiment_counts[final_sentiment] / len(sentiments)
        
        return {
            'sentiment': final_sentiment,
            'confidence': confidence,
            'scores': results
        }
    
    def analyze_tweets(self, tweets_list):
        """Run AI sentiment on tweets"""
        if not tweets_list:
            return []
        
        print(f"[AI] Analyzing sentiment for {len(tweets_list)} tweets...")
        analyzed = []
        
        for i, tweet in enumerate(tweets_list):
            text = tweet.get('text', '')
            sentiment = self.analyze_sentiment_ai(text)
            
            analyzed.append({
                **tweet,
                'ai_sentiment': sentiment
            })
            
            if (i + 1) % 20 == 0:
                print(f"  - Analyzed {i + 1}/{len(tweets_list)}")
        
        print(f"[OK] Sentiment analysis complete\n")
        return analyzed
    
    def extract_topics(self, tweets_list):
        """Topic modeling with LDA"""
        if not tweets_list or len(tweets_list) < 5:
            return {}
        
        print("[ANALYSIS] Running topic modeling...")
        
        texts = [t.get('text', '') for t in tweets_list if t.get('text')]
        
        if len(texts) < 5:
            return {}
        
        try:
            vectorizer = TfidfVectorizer(
                max_features=100,
                stop_words='english',
                ngram_range=(1, 2)
            )
            tfidf = vectorizer.fit_transform(texts)
            
            n_topics = min(5, len(texts) // 3)
            lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
            lda.fit(tfidf)
            
            topics = {}
            feature_names = vectorizer.get_feature_names_out()
            
            for idx, topic in enumerate(lda.components_):
                top_indices = topic.argsort()[-10:][::-1]
                top_words = [feature_names[i] for i in top_indices]
                topics[f"Topic_{idx + 1}"] = {
                    'keywords': top_words[:5],
                    'all_keywords': top_words
                }
            
            print(f"[OK] Identified {len(topics)} topics\n")
            return topics
        except Exception as e:
            print(f"[WARN] Topic modeling failed: {e}\n")
            return {}
    
    def analyze_hashtags(self, tweets_list):
        """Analyze hashtag sentiment"""
        if not tweets_list:
            return {}
        
        print("[TAGS] Analyzing hashtag sentiment...")
        
        hashtag_data = {}
        
        for tweet in tweets_list:
            if 'entities' in tweet and 'hashtags' in tweet['entities']:
                sentiment = tweet.get('ai_sentiment', {}).get('sentiment', 'neutral')
                
                for hashtag in tweet['entities']['hashtags']:
                    tag = hashtag['tag'].lower()
                    
                    if tag not in hashtag_data:
                        hashtag_data[tag] = {
                            'count': 0,
                            'positive': 0,
                            'negative': 0,
                            'neutral': 0,
                            'total_engagement': 0
                        }
                    
                    hashtag_data[tag]['count'] += 1
                    hashtag_data[tag][sentiment] += 1
                    
                    if 'public_metrics' in tweet:
                        pm = tweet['public_metrics']
                        engagement = pm.get('like_count', 0) + pm.get('retweet_count', 0)
                        hashtag_data[tag]['total_engagement'] += engagement
        
        for tag in hashtag_data:
            if hashtag_data[tag]['count'] > 0:
                hashtag_data[tag]['avg_engagement'] = (
                    hashtag_data[tag]['total_engagement'] / hashtag_data[tag]['count']
                )
        
        print(f"[OK] Analyzed {len(hashtag_data)} hashtags\n")
        return hashtag_data
    
    def analyze_audience_engagement(self, tweets_list):
        """Analyze engagement patterns"""
        print("[AUDIENCE] Analyzing engagement patterns...")
        
        analysis = {
            'most_discussed': [],
            'most_controversial': [],
            'highest_engagement': []
        }
        
        for tweet in tweets_list:
            if 'public_metrics' in tweet:
                pm = tweet['public_metrics']
                
                tweet_info = {
                    'text': tweet.get('text', '')[:100],
                    'likes': pm.get('like_count', 0),
                    'retweets': pm.get('retweet_count', 0),
                    'replies': pm.get('reply_count', 0),
                    'quotes': pm.get('quote_count', 0),
                    'total_engagement': (
                        pm.get('like_count', 0) +
                        pm.get('retweet_count', 0) * 2 +
                        pm.get('reply_count', 0) * 3
                    )
                }
                
                # Most discussed (high replies)
                if pm.get('reply_count', 0) > 10:
                    analysis['most_discussed'].append(tweet_info)
                
                # Controversial (replies > likes ratio)
                if pm.get('like_count', 0) > 0:
                    ratio = pm.get('reply_count', 0) / pm.get('like_count', 1)
                    if ratio > 0.5:
                        analysis['most_controversial'].append({
                            **tweet_info,
                            'controversy_ratio': ratio
                        })
                
                # High engagement
                analysis['highest_engagement'].append(tweet_info)
        
        # Sort
        analysis['most_discussed'] = sorted(
            analysis['most_discussed'],
            key=lambda x: x['replies'],
            reverse=True
        )[:10]
        
        analysis['most_controversial'] = sorted(
            analysis['most_controversial'],
            key=lambda x: x['controversy_ratio'],
            reverse=True
        )[:10]
        
        analysis['highest_engagement'] = sorted(
            analysis['highest_engagement'],
            key=lambda x: x['total_engagement'],
            reverse=True
        )[:10]
        
        print(f"[OK] Engagement analysis complete\n")
        return analysis
    
    def generate_sentiment_summary(self, analyzed_tweets, analyzed_mentions, analyzed_search):
        """Generate overall sentiment summary"""
        summary = {
            'own_tweets': {'positive': 0, 'negative': 0, 'neutral': 0, 'total': 0},
            'mentions': {'positive': 0, 'negative': 0, 'neutral': 0, 'total': 0},
            'search_results': {'positive': 0, 'negative': 0, 'neutral': 0, 'total': 0}
        }
        
        for tweet in analyzed_tweets:
            sent = tweet.get('ai_sentiment', {}).get('sentiment', 'neutral')
            summary['own_tweets'][sent] += 1
            summary['own_tweets']['total'] += 1
        
        for tweet in analyzed_mentions:
            sent = tweet.get('ai_sentiment', {}).get('sentiment', 'neutral')
            summary['mentions'][sent] += 1
            summary['mentions']['total'] += 1
        
        for tweet in analyzed_search:
            sent = tweet.get('ai_sentiment', {}).get('sentiment', 'neutral')
            summary['search_results'][sent] += 1
            summary['search_results']['total'] += 1
        
        return summary
    
    def analyze_raw_data(self, raw_filename):
        """Main analysis function"""
        print(f"[LOAD] Loading raw data from: {raw_filename}\n")
        
        with open(raw_filename, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        analyzed_data = {
            'analysis_timestamp': datetime.now().isoformat(),
            'source_file': raw_filename,
            'politician': raw_data.get('politician_username'),
            'raw_data': raw_data['raw_data'],
            'analysis': {}
        }
        
        # 1. Analyze own tweets
        tweets = raw_data['raw_data'].get('user_tweets', {}).get('data', [])
        if tweets:
            analyzed_data['analysis']['analyzed_tweets'] = self.analyze_tweets(tweets)
        else:
            analyzed_data['analysis']['analyzed_tweets'] = []
        
        # 2. Analyze mentions
        mentions = raw_data['raw_data'].get('mentions', {}).get('data', [])
        if mentions:
            analyzed_data['analysis']['analyzed_mentions'] = self.analyze_tweets(mentions)
        else:
            analyzed_data['analysis']['analyzed_mentions'] = []
        
        # 3. Analyze search results
        search = raw_data['raw_data'].get('search_results', {}).get('data', [])
        if search:
            analyzed_data['analysis']['analyzed_search'] = self.analyze_tweets(search)
        else:
            analyzed_data['analysis']['analyzed_search'] = []
        
        # 4. Topic modeling
        all_tweets = tweets + mentions
        analyzed_data['analysis']['topics'] = self.extract_topics(all_tweets)
        
        # 5. Hashtag analysis
        analyzed_data['analysis']['hashtag_sentiment'] = self.analyze_hashtags(
            analyzed_data['analysis']['analyzed_tweets'] +
            analyzed_data['analysis']['analyzed_mentions']
        )
        
        # 6. Audience engagement
        analyzed_data['analysis']['audience_engagement'] = self.analyze_audience_engagement(
            analyzed_data['analysis']['analyzed_tweets']
        )
        
        # 7. Sentiment summary
        analyzed_data['analysis']['sentiment_summary'] = self.generate_sentiment_summary(
            analyzed_data['analysis']['analyzed_tweets'],
            analyzed_data['analysis']['analyzed_mentions'],
            analyzed_data['analysis']['analyzed_search']
        )
        
        # Save analyzed data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        analyzed_filename = f'mark_daou_analyzed_{timestamp}.json'
        
        print("[SAVE] Saving analyzed data...")
        with open(analyzed_filename, 'w', encoding='utf-8') as f:
            json.dump(analyzed_data, f, indent=2, ensure_ascii=False)
        
        print("\n" + "="*70)
        print(f"[OK] ANALYSIS COMPLETE!")
        print(f"[FILE] Saved to: {analyzed_filename}")
        print("="*70 + "\n")
        
        self.print_summary(analyzed_data['analysis'])
        
        return analyzed_filename
    
    def print_summary(self, analysis):
        """Print analysis summary"""
        print("[SUMMARY] Analysis Results:")
        print("-" * 70)
        
        # Sentiment
        sentiment = analysis.get('sentiment_summary', {})
        for category, data in sentiment.items():
            total = data.get('total', 0)
            if total > 0:
                print(f"\n{category.replace('_', ' ').title()}:")
                print(f"  Positive: {data['positive']} ({data['positive']/total*100:.1f}%)")
                print(f"  Negative: {data['negative']} ({data['negative']/total*100:.1f}%)")
                print(f"  Neutral:  {data['neutral']} ({data['neutral']/total*100:.1f}%)")
        
        # Topics
        topics = analysis.get('topics', {})
        if topics:
            print(f"\n[TOPICS] Identified {len(topics)} topics:")
            for name, data in list(topics.items())[:3]:
                print(f"  {name}: {', '.join(data['keywords'])}")
        
        # Top hashtags
        hashtags = analysis.get('hashtag_sentiment', {})
        if hashtags:
            top = sorted(hashtags.items(), key=lambda x: x[1]['count'], reverse=True)[:5]
            print(f"\n[HASHTAGS] Top 5 hashtags:")
            for tag, data in top:
                print(f"  #{tag}: {data['count']} uses, Pos:{data['positive']} Neg:{data['negative']}")
        
        print("\n" + "-" * 70)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("[ERROR] Usage: python analyze_data.py <raw_data_file.json>")
        print("   Or just run: python analyze_data.py")
        print("   Will auto-find latest raw data file")
        
        # Auto-find latest file
        import glob
        files = glob.glob("mark_daou_raw_data_*.json")
        if files:
            raw_file = max(files)
            print(f"\n[AUTO] Found: {raw_file}")
        else:
            print("\n[ERROR] No raw data files found")
            print("Run: python collect_and_analyze.py first")
            exit(1)
    else:
        raw_file = sys.argv[1]
    
    analyzer = DataAnalyzer()
    result = analyzer.analyze_raw_data(raw_file)
    
    if result:
        print(f"\n[SUCCESS] Analysis saved to: {result}")
        print("\n[NEXT] View dashboard:")
        print("   streamlit run enhanced_dashboard.py")

