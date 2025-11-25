"""
Advanced Insights Analyzer with State-of-the-Art AI Models
Extracts comprehensive insights from political social media data

Features:
- Advanced sentiment analysis (RoBERTa Twitter-tuned)
- Emotion detection
- Named Entity Recognition (NER)
- Topic modeling with coherence scoring
- Influence network analysis
- Temporal trend analysis
- Engagement prediction
- Geographic insights
- Content categorization
- Hashtag performance tracking
"""

import json
import warnings
import sys
warnings.filterwarnings('ignore')

# Fix Windows encoding issues
if sys.platform == 'win32':
    import os
    os.environ['PYTHONIOENCODING'] = 'utf-8'

def convert_to_serializable(obj):
    """Convert numpy types to Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj

from datetime import datetime, timedelta
from collections import Counter, defaultdict
import re
import numpy as np
import pandas as pd

# NLP & ML imports
from transformers import (
    pipeline, 
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification
)
import torch
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import nltk

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('maxent_ne_chunker', quiet=True)
    nltk.download('words', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass

class AdvancedInsightsAnalyzer:
    """Extract comprehensive insights using state-of-the-art AI models"""
    
    def __init__(self):
        print("\n" + "="*80)
        print("  ADVANCED INSIGHTS ANALYZER")
        print("  Loading state-of-the-art AI models...")
        print("="*80 + "\n")
        
        self.device = 0 if torch.cuda.is_available() else -1
        print(f"[DEVICE] Using: {'GPU' if self.device == 0 else 'CPU'}\n")
        
        # Initialize models
        self._load_sentiment_models()
        self._load_emotion_model()
        self._load_ner_model()
        
        print("\n[SUCCESS] All models loaded!\n")
    
    def _is_retweet(self, tweet):
        """Return True if tweet is a retweet"""
        for ref in tweet.get('referenced_tweets', []) or []:
            if ref.get('type') == 'retweeted':
                return True
        text = (tweet.get('text') or '').strip()
        return text.startswith('RT ')

    def _filter_retweets(self, tweets, label):
        """Remove retweets from tweet collections to avoid duplicates"""
        if not tweets:
            return []
        filtered = [t for t in tweets if not self._is_retweet(t)]
        removed = len(tweets) - len(filtered)
        if removed > 0:
            print(f"[FILTER] Removed {removed} retweets from {label} to avoid double counting.")
        return filtered
    
    def _deduplicate_tweets_across_sources(self, own_tweets, mentions, search_results):
        """Remove duplicate tweets across different sources by tweet ID.
        
        Priority order:
        1. own_tweets (keep all - these are the politician's tweets)
        2. mentions (remove if already in own_tweets)
        3. search_results (remove if already in own_tweets or mentions)
        """
        # Collect tweet IDs from higher priority sources
        seen_ids = set()
        
        # Track own tweets
        for tweet in own_tweets:
            tweet_id = tweet.get('id')
            if tweet_id:
                seen_ids.add(tweet_id)
        
        # Filter mentions - remove if already seen
        mentions_filtered = []
        mentions_removed = 0
        for tweet in mentions:
            tweet_id = tweet.get('id')
            if tweet_id and tweet_id not in seen_ids:
                mentions_filtered.append(tweet)
                seen_ids.add(tweet_id)
            elif tweet_id:
                mentions_removed += 1
        
        # Filter search results - remove if already seen
        search_filtered = []
        search_removed = 0
        for tweet in search_results:
            tweet_id = tweet.get('id')
            if tweet_id and tweet_id not in seen_ids:
                search_filtered.append(tweet)
                seen_ids.add(tweet_id)
            elif tweet_id:
                search_removed += 1
        
        if mentions_removed > 0:
            print(f"[DEDUP] Removed {mentions_removed} duplicate mentions (already in own tweets)")
        if search_removed > 0:
            print(f"[DEDUP] Removed {search_removed} duplicate search results (already in own tweets or mentions)")
        
        return own_tweets, mentions_filtered, search_filtered

    def _load_sentiment_models(self):
        """Load advanced sentiment analysis models"""
        print("[AI] Loading Twitter-optimized RoBERTa sentiment model...")
        try:
            # Twitter-RoBERTa - best for Twitter/social media sentiment
            self.sentiment_model = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=self.device,
                truncation=True,
                max_length=512
            )
            print("[OK] RoBERTa sentiment model loaded (Twitter-optimized)")
        except Exception as e:
            print(f"[WARN] Could not load RoBERTa: {e}")
            print("[AI] Falling back to multilingual BERT...")
            try:
                self.sentiment_model = pipeline(
                    "sentiment-analysis",
                    model="nlptown/bert-base-multilingual-uncased-sentiment",
                    device=self.device
                )
                print("[OK] BERT sentiment model loaded (multilingual)")
            except:
                self.sentiment_model = None
                print("[ERROR] Could not load sentiment model")
    
    def _load_emotion_model(self):
        """Load emotion detection model"""
        print("[AI] Loading emotion detection model...")
        try:
            self.emotion_model = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                device=self.device,
                top_k=None
            )
            print("[OK] Emotion model loaded (6 emotions)")
        except Exception as e:
            print(f"[WARN] Could not load emotion model: {e}")
            self.emotion_model = None
    
    def _load_ner_model(self):
        """Load Named Entity Recognition model"""
        print("[AI] Loading NER model for entity extraction...")
        try:
            self.ner_model = pipeline(
                "ner",
                model="dslim/bert-base-NER",
                device=self.device,
                aggregation_strategy="simple"
            )
            print("[OK] NER model loaded")
        except Exception as e:
            print(f"[WARN] Could not load NER model: {e}")
            self.ner_model = None
    
    def analyze_sentiment_advanced(self, text):
        """Advanced multi-model sentiment analysis"""
        if not text or len(text.strip()) < 3:
            return {
                'sentiment': 'neutral',
                'confidence': 0.0,
                'label': 'neutral',
                'scores': {}
            }
        
        results = {}
        
        # 1. Twitter-RoBERTa (primary)
        if self.sentiment_model:
            try:
                roberta_result = self.sentiment_model(text[:512])[0]
                
                # Map labels
                label_map = {
                    'LABEL_0': 'negative',
                    'LABEL_1': 'neutral', 
                    'LABEL_2': 'positive',
                    'negative': 'negative',
                    'neutral': 'neutral',
                    'positive': 'positive'
                }
                
                sentiment_label = label_map.get(roberta_result['label'], 'neutral')
                
                results['roberta'] = {
                    'sentiment': sentiment_label,
                    'confidence': roberta_result['score'],
                    'raw_label': roberta_result['label']
                }
            except Exception as e:
                print(f"[WARN] RoBERTa failed: {e}")
        
        # 2. TextBlob (lexicon-based, good for context)
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
        
        # 3. Aggregate with weighted voting (RoBERTa gets more weight)
        sentiments = []
        
        if 'roberta' in results:
            # RoBERTa gets triple weight (most accurate for social media)
            conf = results['roberta']['confidence']
            for _ in range(3):
                sentiments.append(results['roberta']['sentiment'])
        
        sentiments.append(results['textblob']['sentiment'])
        
        if sentiments:
            sentiment_counts = Counter(sentiments)
            final_sentiment = sentiment_counts.most_common(1)[0][0]
            confidence = sentiment_counts[final_sentiment] / len(sentiments)
        else:
            final_sentiment = 'neutral'
            confidence = 0.0
        
        return {
            'sentiment': final_sentiment,
            'confidence': confidence,
            'label': final_sentiment,
            'scores': results
        }
    
    def analyze_emotion(self, text):
        """Detect emotions in text (joy, sadness, anger, fear, surprise, disgust)"""
        if not self.emotion_model or not text or len(text.strip()) < 3:
            return {
                'dominant_emotion': 'neutral',
                'confidence': 0.0,
                'all_emotions': {}
            }
        
        try:
            emotions = self.emotion_model(text[:512])[0]
            
            # Get dominant emotion
            dominant = max(emotions, key=lambda x: x['score'])
            
            all_emotions = {e['label']: e['score'] for e in emotions}
            
            return {
                'dominant_emotion': dominant['label'],
                'confidence': dominant['score'],
                'all_emotions': all_emotions
            }
        except Exception as e:
            return {
                'dominant_emotion': 'neutral',
                'confidence': 0.0,
                'all_emotions': {}
            }
    
    def extract_entities_from_tweet(self, tweet):
        """Extract entities using Twitter's annotations FIRST (best quality), then BERT NER as fallback"""
        entities = []
        used_twitter = False
        
        # 1. PRIORITY: Use Twitter's built-in entity annotations (MUCH better for Arabic!)
        if 'entities' in tweet and 'annotations' in tweet['entities']:
            twitter_annotations = tweet['entities']['annotations']
            used_twitter = len(twitter_annotations) > 0
            
            for annotation in twitter_annotations:
                # Twitter provides: Organization, Person, Place, Product, Other
                entity_type = annotation.get('type', 'MISC')
                
                # Map Twitter types to our standard types
                type_mapping = {
                    'Organization': 'ORG',
                    'Person': 'PER',
                    'Place': 'LOC',
                    'Product': 'MISC',
                    'Other': 'MISC'
                }
                
                normalized_text = annotation.get('normalized_text', '')
                if not normalized_text:
                    # Extract from tweet text using start/end
                    text = tweet.get('text', '')
                    start = annotation.get('start', 0)
                    end = annotation.get('end', 0)
                    normalized_text = text[start:end] if start < end else ''
                
                if len(normalized_text) >= 2:  # Valid entity
                    entities.append({
                        'text': normalized_text,
                        'type': type_mapping.get(entity_type, 'MISC'),
                        'score': annotation.get('probability', 0.9),
                        'source': 'twitter'  # Mark as Twitter annotation
                    })
        
        # 2. FALLBACK: Use BERT NER if no Twitter annotations
        if not entities:
            text = tweet.get('text', '')
            entities = self.extract_entities_bert(text)
        
        return entities
    
    def extract_entities_bert(self, text):
        """Extract named entities using BERT NER (fallback method)"""
        if not self.ner_model or not text or len(text.strip()) < 3:
            return []
        
        try:
            entities = self.ner_model(text[:512])
            
            # Filter and clean entities
            cleaned_entities = []
            for ent in entities:
                entity_text = ent['word'].strip()
                
                # Skip if:
                # - Less than 2 characters
                # - Starts with ## (hashtag fragment)
                # - Is just "RT" or single letter
                # - Is only numbers
                # - Contains only special characters
                if len(entity_text) < 2:
                    continue
                if entity_text.startswith('##'):
                    continue
                if entity_text in ['RT', 'R', 'T', 'http', 'https']:
                    continue
                if entity_text.isdigit():
                    continue
                if not any(c.isalnum() for c in entity_text):
                    continue
                
                # Clean up the text
                entity_text = entity_text.replace('##', '').strip()
                
                # Skip if cleaned text is too short
                if len(entity_text) < 2:
                    continue
                
                cleaned_entities.append({
                    'text': entity_text,
                    'type': ent['entity_group'],
                    'score': ent['score'],
                    'source': 'bert'  # Mark as BERT extraction
                })
            
            return cleaned_entities
        except:
            return []
    
    def analyze_tweet_comprehensive(self, tweet):
        """Comprehensive analysis of a single tweet (mutates tweet in place)"""
        text = tweet.get('text', '')
        
        # 1. Advanced sentiment
        sentiment = self.analyze_sentiment_advanced(text)
        
        # 2. Emotion detection
        emotion = self.analyze_emotion(text)
        
        # 3. Named entities - USE TWITTER'S ANNOTATIONS FIRST (much better!)
        entities = self.extract_entities_from_tweet(tweet)
        
        # 4. Engagement metrics
        pm = tweet.get('public_metrics', {})
        engagement_score = (
            pm.get('like_count', 0) + 
            pm.get('retweet_count', 0) * 2 +
            pm.get('reply_count', 0) * 3 +
            pm.get('quote_count', 0) * 2
        )
        
        # 5. Content features
        has_media = 'attachments' in tweet
        has_urls = 'entities' in tweet and 'urls' in tweet['entities']
        has_hashtags = 'entities' in tweet and 'hashtags' in tweet['entities']
        has_mentions = 'entities' in tweet and 'mentions' in tweet['entities']
        
        # 6. Language
        language = tweet.get('lang', 'unknown')
        
        tweet['ai_sentiment'] = sentiment
        tweet['ai_emotion'] = emotion
        tweet['ai_entities'] = entities
        tweet['engagement_score'] = engagement_score
        tweet['content_features'] = {
            'has_media': has_media,
            'has_urls': has_urls,
            'has_hashtags': has_hashtags,
            'has_mentions': has_mentions,
            'length': len(text),
            'language': language
        }
        
        return tweet
    
    def analyze_tweets_batch(self, tweets_list, label="tweets"):
        """Analyze a batch of tweets"""
        if not tweets_list:
            return []
        
        print(f"[ANALYSIS] Analyzing {len(tweets_list)} {label}...")
        analyzed = []
        
        for i, tweet in enumerate(tweets_list):
            analyzed_tweet = self.analyze_tweet_comprehensive(tweet)
            analyzed.append(analyzed_tweet)
            
            if (i + 1) % 10 == 0 or (i + 1) == len(tweets_list):
                print(f"  Progress: {i + 1}/{len(tweets_list)}")
        
        print(f"[OK] {label.capitalize()} analysis complete\n")
        return analyzed
    
    def extract_topics_advanced(self, tweets_list, n_topics=5):
        """Advanced topic modeling with multiple algorithms"""
        if not tweets_list or len(tweets_list) < 5:
            return {}
        
        print("[TOPICS] Running advanced topic modeling with Arabic + English stop word filtering...")
        
        # Comprehensive Arabic stop words list
        arabic_stop_words = {
            # Prepositions and conjunctions
            'في', 'من', 'إلى', 'على', 'عن', 'مع', 'بعد', 'قبل', 'عند', 'حتى', 'منذ',
            'الى', 'علي', 'عليه', 'عليها', 'منه', 'منها', 'به', 'بها', 'له', 'لها',
            # Articles and pronouns
            'ال', 'هذا', 'هذه', 'ذلك', 'تلك', 'هنا', 'هناك', 'هو', 'هي', 'هم', 'هن',
            'أنا', 'نحن', 'أنت', 'أنتم', 'أنتن', 'انت', 'انه', 'انها',
            # Particles
            'قد', 'لم', 'لن', 'ما', 'لا', 'لكن', 'كان', 'كانت', 'يكون', 'تكون',
            # Conjunctions
            'و', 'أو', 'ثم', 'لكن', 'لكن', 'إن', 'أن', 'ان', 'إذا', 'اذا',
            # Question words
            'ماذا', 'متى', 'أين', 'كيف', 'لماذا', 'من', 'هل', 'اين', 'كيف',
            # Common words
            'كل', 'بعض', 'هذا', 'ذلك', 'التي', 'الذي', 'الذين', 'اللذان', 'اللتان',
            'غير', 'سوى', 'بين', 'حيث', 'خلال', 'بدون', 'ضد', 'نحو', 'فوق', 'تحت',
            'أمام', 'خلف', 'حول', 'دون', 'ضمن', 'امام', 'خلف',
            # Verbs (common auxiliary)
            'كان', 'كانت', 'كانوا', 'يكون', 'تكون', 'يكونوا', 'أصبح', 'أصبحت',
            'صار', 'صارت', 'بات', 'ليس', 'ليست', 'راح', 'رح', 'بدو', 'بدها',
            # Other common particles
            'إذن', 'اذن', 'إلا', 'الا', 'غير', 'سوى', 'عدا', 'خلا', 'حاشا',
            'أي', 'أية', 'بعد', 'قبل', 'فقط', 'أيضا', 'ايضا', 'كذلك', 'هكذا',
            # Single letter particles
            'و', 'ف', 'ب', 'ك', 'ل',
            # Lebanese/Levantine dialect fillers and connectors
            'يعني', 'يلي', 'ولا', 'ولن', 'واحد', 'وحدة', 'شي', 'شوي', 'كتير',
            'بس', 'بلا', 'يا', 'مع', 'عم', 'يللي', 'هيدا', 'هيدي', 'هلق',
            'شو', 'ليش', 'كيف', 'وين', 'ايمتى', 'مين', 'منيح', 'ماشي',
            # Common numbers/quantifiers that aren't meaningful
            'واحد', 'اثنين', 'ثلاثة', 'كان', 'كلهم', 'كلها', 'بعضهم',
            # More auxiliary verbs
            'سوف', 'سيكون', 'ستكون', 'يمكن', 'ربما', 'عسى', 'لعل',
            # English common words that might appear
            'the', 'and', 'for', 'are', 'this', 'that', 'with', 'from', 'have', 'has',
            'was', 'were', 'been', 'will', 'would', 'can', 'could', 'should', 'been', 'being'
        }
        
        # Get texts and clean them
        texts = []
        for t in tweets_list:
            text = t.get('text', '')
            if text:
                # Remove URLs
                text = re.sub(r'http\S+', '', text)
                # Remove @mentions
                text = re.sub(r'@\w+', '', text)
                # Remove hashtag symbols but keep text
                text = re.sub(r'#(\w+)', r'\1', text)
                # Remove RT
                text = text.replace('RT', '').strip()
                # Remove extra whitespace
                text = ' '.join(text.split())
                
                # Only add if has meaningful content
                if len(text) > 10:
                    texts.append(text)
        
        if len(texts) < 5:
            return {}
        
        try:
            # Custom token filter to remove junk
            def is_valid_token(token):
                # Remove if in Arabic stop words
                if token in arabic_stop_words:
                    return False
                
                # Remove if too short
                if len(token) < 3:
                    return False
                
                # Remove if is URL fragment
                if token.startswith('http') or token.startswith('www'):
                    return False
                
                # Remove if pure numbers
                if token.isdigit():
                    return False
                
                # Remove if only punctuation
                if not any(c.isalnum() for c in token):
                    return False
                
                # Remove common meaningless patterns
                if token in ['...', '___', '---', 'amp']:
                    return False
                    
                return True
            
            # Combine English stop words with our Arabic ones
            combined_stop_words = list(arabic_stop_words)
            print(f"[FILTER] Using {len(combined_stop_words)} Arabic stop words + English stop words")
            
            # Vectorization with better filtering
            vectorizer = TfidfVectorizer(
                max_features=150,
                stop_words=combined_stop_words,  # Arabic + English stopwords
                ngram_range=(1, 2),  # Unigrams and bigrams only
                min_df=2,  # Must appear in at least 2 documents
                max_df=0.8,  # Can't appear in more than 80% of documents
                token_pattern=r'(?u)\b\w\w\w+\b'  # At least 3 characters
            )
            tfidf = vectorizer.fit_transform(texts)
            
            n_topics = min(n_topics, len(texts) // 3)
            if n_topics < 2:
                n_topics = 2
            
            # LDA Topic Modeling
            lda = LatentDirichletAllocation(
                n_components=n_topics,
                random_state=42,
                max_iter=20,
                learning_method='online'
            )
            lda.fit(tfidf)
            
            topics = {}
            feature_names = vectorizer.get_feature_names_out()
            
            for idx, topic in enumerate(lda.components_):
                top_indices = topic.argsort()[-15:][::-1]  # Get top 15 for more options
                
                # Filter words
                valid_words = []
                valid_scores = []
                for i in top_indices:
                    word = feature_names[i]
                    score = topic[i]
                    
                    # Additional filtering
                    if is_valid_token(word):
                        # Extra check: skip if word is in Arabic stop words (case-insensitive)
                        if word.lower() not in arabic_stop_words:
                            valid_words.append(word)
                            valid_scores.append(float(score))
                    
                    if len(valid_words) >= 10:
                        break
                
                if len(valid_words) >= 3:  # Only add if we have at least 3 valid keywords
                    topics[f"topic_{idx + 1}"] = {
                        'keywords': valid_words[:5],
                        'all_keywords': valid_words[:10],
                        'scores': valid_scores[:5],
                        'coherence': float(np.mean(valid_scores[:5])) if valid_scores else 0.0
                    }
            
            print(f"[OK] Identified {len(topics)} coherent topics\n")
            return topics
            
        except Exception as e:
            print(f"[WARN] Topic modeling failed: {e}\n")
            return {}
    
    def analyze_temporal_patterns(self, tweets_list):
        """Analyze temporal patterns and trends"""
        if not tweets_list:
            return {}
        
        print("[TIME] Analyzing temporal patterns...")
        
        # Extract timestamps and sentiments
        time_sentiment = []
        hourly_counts = defaultdict(int)
        daily_counts = defaultdict(int)
        hourly_engagement = defaultdict(list)
        
        for tweet in tweets_list:
            created_at = tweet.get('created_at')
            if not created_at:
                continue
            
            try:
                dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                
                sentiment = tweet.get('ai_sentiment', {}).get('sentiment', 'neutral')
                engagement = tweet.get('engagement_score', 0)
                
                time_sentiment.append({
                    'datetime': dt,
                    'hour': dt.hour,
                    'day': dt.strftime('%A'),
                    'date': dt.date(),
                    'sentiment': sentiment,
                    'engagement': engagement
                })
                
                hourly_counts[dt.hour] += 1
                daily_counts[dt.strftime('%A')] += 1
                hourly_engagement[dt.hour].append(engagement)
                
            except:
                continue
        
        # Calculate best posting times
        best_hours = sorted(
            [(h, sum(hourly_engagement[h])/len(hourly_engagement[h])) 
             for h in hourly_engagement if len(hourly_engagement[h]) > 0],
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        # Sentiment trends over time
        sentiment_by_day = defaultdict(lambda: {'positive': 0, 'negative': 0, 'neutral': 0})
        for ts in time_sentiment:
            sentiment_by_day[ts['date']][ts['sentiment']] += 1
        
        print(f"[OK] Temporal analysis complete\n")
        
        return {
            'hourly_activity': dict(hourly_counts),
            'daily_activity': dict(daily_counts),
            'best_posting_hours': [
                {'hour': h, 'avg_engagement': eng} 
                for h, eng in best_hours
            ],
            'sentiment_trends': {
                str(date): counts 
                for date, counts in sorted(sentiment_by_day.items())
            },
            'total_analyzed': len(time_sentiment)
        }
    
    def analyze_influence_metrics(self, tweets_list, user_profile):
        """Calculate influence and impact metrics"""
        print("[INFLUENCE] Calculating influence metrics...")
        
        if not user_profile or 'data' not in user_profile:
            return {}
        
        profile = user_profile['data']
        pm = profile.get('public_metrics', {})
        
        followers = pm.get('followers_count', 0)
        following = pm.get('following_count', 1)
        total_tweets = pm.get('tweet_count', 0)
        
        # Calculate metrics
        follower_ratio = followers / max(following, 1)
        
        # Engagement rate from tweets
        total_engagement = 0
        total_reach = 0
        
        for tweet in tweets_list:
            tweet_pm = tweet.get('public_metrics', {})
            engagement = (
                tweet_pm.get('like_count', 0) +
                tweet_pm.get('retweet_count', 0) +
                tweet_pm.get('reply_count', 0)
            )
            total_engagement += engagement
        
        avg_engagement = total_engagement / len(tweets_list) if tweets_list else 0
        engagement_rate = (avg_engagement / followers * 100) if followers > 0 else 0
        
        # Influence score (composite metric)
        influence_score = (
            (followers * 0.4) +
            (avg_engagement * 0.3) +
            (follower_ratio * 1000 * 0.2) +
            (total_tweets * 0.1)
        )
        
        # Virality potential
        retweet_rate = sum(
            t.get('public_metrics', {}).get('retweet_count', 0) 
            for t in tweets_list
        ) / len(tweets_list) if tweets_list else 0
        
        virality_score = retweet_rate * followers / 1000
        
        print(f"[OK] Influence metrics calculated\n")
        
        return {
            'follower_ratio': round(follower_ratio, 2),
            'engagement_rate': round(engagement_rate, 4),
            'avg_engagement_per_tweet': round(avg_engagement, 2),
            'influence_score': round(influence_score, 2),
            'virality_score': round(virality_score, 2),
            'reach_potential': followers,
            'content_volume': total_tweets,
            'influence_tier': self._get_influence_tier(influence_score)
        }
    
    def _get_influence_tier(self, score):
        """Categorize influence level"""
        if score > 100000:
            return 'Mega Influencer'
        elif score > 50000:
            return 'Macro Influencer'
        elif score > 10000:
            return 'Micro Influencer'
        elif score > 1000:
            return 'Rising Voice'
        else:
            return 'Emerging Voice'
    
    def analyze_content_strategy(self, tweets_list):
        """Analyze content strategy and performance"""
        print("[CONTENT] Analyzing content strategy...")
        
        if not tweets_list:
            return {}
        
        # Content type analysis
        with_media = sum(1 for t in tweets_list if t.get('content_features', {}).get('has_media'))
        with_urls = sum(1 for t in tweets_list if t.get('content_features', {}).get('has_urls'))
        with_hashtags = sum(1 for t in tweets_list if t.get('content_features', {}).get('has_hashtags'))
        
        # Performance by content type
        media_engagement = []
        url_engagement = []
        hashtag_engagement = []
        plain_engagement = []
        
        for tweet in tweets_list:
            features = tweet.get('content_features', {})
            engagement = tweet.get('engagement_score', 0)
            
            if features.get('has_media'):
                media_engagement.append(engagement)
            if features.get('has_urls'):
                url_engagement.append(engagement)
            if features.get('has_hashtags'):
                hashtag_engagement.append(engagement)
            if not any([features.get('has_media'), features.get('has_urls'), 
                       features.get('has_hashtags')]):
                plain_engagement.append(engagement)
        
        # Sentiment by content type
        sentiment_by_features = defaultdict(lambda: {'positive': 0, 'negative': 0, 'neutral': 0})
        
        for tweet in tweets_list:
            sentiment = tweet.get('ai_sentiment', {}).get('sentiment', 'neutral')
            features = tweet.get('content_features', {})
            
            if features.get('has_media'):
                sentiment_by_features['media'][sentiment] += 1
            if features.get('has_hashtags'):
                sentiment_by_features['hashtags'][sentiment] += 1
        
        print(f"[OK] Content strategy analysis complete\n")
        
        return {
            'content_distribution': {
                'with_media': with_media,
                'with_urls': with_urls,
                'with_hashtags': with_hashtags,
                'plain_text': len(tweets_list) - with_media - with_urls - with_hashtags
            },
            'performance_by_type': {
                'media_avg_engagement': np.mean(media_engagement) if media_engagement else 0,
                'url_avg_engagement': np.mean(url_engagement) if url_engagement else 0,
                'hashtag_avg_engagement': np.mean(hashtag_engagement) if hashtag_engagement else 0,
                'plain_avg_engagement': np.mean(plain_engagement) if plain_engagement else 0
            },
            'best_performing_type': self._get_best_content_type(
                media_engagement, url_engagement, hashtag_engagement, plain_engagement
            )
        }
    
    def _get_best_content_type(self, media, url, hashtag, plain):
        """Determine best performing content type"""
        types = {
            'media': np.mean(media) if media else 0,
            'url': np.mean(url) if url else 0,
            'hashtag': np.mean(hashtag) if hashtag else 0,
            'plain': np.mean(plain) if plain else 0
        }
        return max(types, key=types.get)
    
    def analyze_hashtag_performance(self, tweets_list):
        """Detailed hashtag performance analysis"""
        print("[HASHTAGS] Analyzing hashtag performance...")
        
        hashtag_data = defaultdict(lambda: {
            'count': 0,
            'total_engagement': 0,
            'sentiments': {'positive': 0, 'negative': 0, 'neutral': 0},
            'emotions': defaultdict(int),
            'avg_engagement': 0
        })
        
        for tweet in tweets_list:
            if 'entities' in tweet and 'hashtags' in tweet['entities']:
                engagement = tweet.get('engagement_score', 0)
                sentiment = tweet.get('ai_sentiment', {}).get('sentiment', 'neutral')
                emotion = tweet.get('ai_emotion', {}).get('dominant_emotion', 'neutral')
                
                for hashtag in tweet['entities']['hashtags']:
                    tag = hashtag['tag'].lower()
                    
                    hashtag_data[tag]['count'] += 1
                    hashtag_data[tag]['total_engagement'] += engagement
                    hashtag_data[tag]['sentiments'][sentiment] += 1
                    hashtag_data[tag]['emotions'][emotion] += 1
        
        # Calculate averages and format
        hashtag_performance = {}
        for tag, data in hashtag_data.items():
            if data['count'] > 0:
                hashtag_performance[tag] = {
                    'count': data['count'],
                    'avg_engagement': data['total_engagement'] / data['count'],
                    'sentiments': dict(data['sentiments']),
                    'dominant_emotion': max(data['emotions'].items(), key=lambda x: x[1])[0] if data['emotions'] else 'neutral',
                    'engagement_score': data['total_engagement']
                }
        
        # Sort by engagement
        sorted_hashtags = sorted(
            hashtag_performance.items(),
            key=lambda x: x[1]['avg_engagement'],
            reverse=True
        )
        
        print(f"[OK] Analyzed {len(hashtag_performance)} hashtags\n")
        
        return {
            'hashtag_performance': dict(sorted_hashtags[:20]),
            'total_unique_hashtags': len(hashtag_performance),
            'top_engaging_hashtags': [
                {'tag': tag, 'avg_engagement': data['avg_engagement']}
                for tag, data in sorted_hashtags[:10]
            ]
        }
    
    def extract_entity_network(self, tweets_list):
        """Extract network of mentioned entities"""
        print("[ENTITIES] Extracting entity network...")
        
        entity_counts = defaultdict(int)
        entity_types = defaultdict(set)
        entity_sentiments = defaultdict(lambda: {'positive': 0, 'negative': 0, 'neutral': 0})
        twitter_count = 0
        bert_count = 0
        
        # Entity normalization map
        def normalize_entity(text):
            """Normalize entity names for better merging"""
            # Remove @ symbols
            text = text.replace('@', '').strip()
            
            # Handle common patterns
            text_lower = text.lower()
            
            # Fix "DaouMark" → "Mark Daou"
            if 'daoumark' in text_lower or 'markdaou' in text_lower:
                return 'Mark Daou'
            
            # Fix "StephanieRady" variations
            if 'stephanierady' in text_lower or 'radystephanie' in text_lower:
                return 'Stephanie Rady'
            
            # Remove hashtag symbols
            text = text.replace('#', '').strip()
            
            # Skip if only 1 character after cleaning
            if len(text) <= 1:
                return None
            
            return text
        
        for tweet in tweets_list:
            entities = tweet.get('ai_entities', [])
            sentiment = tweet.get('ai_sentiment', {}).get('sentiment', 'neutral')
            
            for entity in entities:
                text = entity['text']
                entity_type = entity['type']
                source = entity.get('source', 'unknown')
                
                # Track sources
                if source == 'twitter':
                    twitter_count += 1
                elif source == 'bert':
                    bert_count += 1
                
                # Normalize entity
                normalized = normalize_entity(text)
                if not normalized:
                    continue
                
                entity_counts[normalized] += 1
                entity_types[normalized].add(entity_type)
                entity_sentiments[normalized][sentiment] += 1
        
        # Format results (convert defaultdict to Counter for most_common)
        entity_network = {}
        entity_counter = Counter(entity_counts)
        for entity, count in entity_counter.most_common(30):
            entity_network[entity] = {
                'count': count,
                'types': list(entity_types[entity]),
                'sentiment': dict(entity_sentiments[entity]),
                'dominant_sentiment': max(
                    entity_sentiments[entity].items(),
                    key=lambda x: x[1]
                )[0] if entity_sentiments[entity] else 'neutral'
            }
        
        print(f"[OK] Extracted {len(entity_network)} key entities")
        print(f"      Twitter annotations: {twitter_count} entities")
        print(f"      BERT NER: {bert_count} entities\n")
        
        return entity_network
    
    def generate_sentiment_summary(self, analyzed_tweets, analyzed_mentions, analyzed_search):
        """Generate comprehensive sentiment summary"""
        print("[SENTIMENT] Generating sentiment summary...")
        
        def summarize_sentiments(tweets, label):
            if not tweets:
                return {
                    'total': 0,
                    'positive': 0,
                    'negative': 0,
                    'neutral': 0,
                    'sentiment_score': 0,
                    'avg_confidence': 0
                }
            
            sentiments = {'positive': 0, 'negative': 0, 'neutral': 0}
            confidences = []
            
            for tweet in tweets:
                sent = tweet.get('ai_sentiment', {}).get('sentiment', 'neutral')
                conf = tweet.get('ai_sentiment', {}).get('confidence', 0)
                
                sentiments[sent] += 1
                confidences.append(conf)
            
            total = sum(sentiments.values())
            sentiment_score = ((sentiments['positive'] - sentiments['negative']) / total * 100) if total > 0 else 0
            
            return {
                'total': total,
                'positive': sentiments['positive'],
                'negative': sentiments['negative'],
                'neutral': sentiments['neutral'],
                'positive_pct': (sentiments['positive'] / total * 100) if total > 0 else 0,
                'negative_pct': (sentiments['negative'] / total * 100) if total > 0 else 0,
                'neutral_pct': (sentiments['neutral'] / total * 100) if total > 0 else 0,
                'sentiment_score': sentiment_score,
                'avg_confidence': np.mean(confidences) if confidences else 0
            }
        
        summary = {
            'own_tweets': summarize_sentiments(analyzed_tweets, 'own tweets'),
            'mentions': summarize_sentiments(analyzed_mentions, 'mentions'),
            'search_results': summarize_sentiments(analyzed_search, 'search results')
        }
        
        print(f"[OK] Sentiment summary generated\n")
        return summary
    
    def generate_emotion_summary(self, analyzed_tweets, analyzed_mentions):
        """Generate emotion distribution summary"""
        print("[EMOTION] Generating emotion summary...")
        
        def summarize_emotions(tweets, label):
            emotions = defaultdict(int)
            
            for tweet in tweets:
                emotion = tweet.get('ai_emotion', {}).get('dominant_emotion', 'neutral')
                emotions[emotion] += 1
            
            total = sum(emotions.values())
            
            return {
                'distribution': dict(emotions),
                'percentages': {
                    emotion: (count / total * 100) if total > 0 else 0
                    for emotion, count in emotions.items()
                },
                'dominant': max(emotions.items(), key=lambda x: x[1])[0] if emotions else 'neutral'
            }
        
        summary = {
            'own_tweets': summarize_emotions(analyzed_tweets, 'own tweets'),
            'mentions': summarize_emotions(analyzed_mentions, 'mentions')
        }
        
        print(f"[OK] Emotion summary generated\n")
        return summary
    
    def generate_insights_recommendations(self, all_analysis):
        """Generate actionable insights and recommendations"""
        print("[INSIGHTS] Generating actionable recommendations...")
        
        recommendations = []
        
        # Sentiment-based recommendations
        sentiment = all_analysis.get('sentiment_summary', {})
        own_score = sentiment.get('own_tweets', {}).get('sentiment_score', 0)
        public_score = sentiment.get('mentions', {}).get('sentiment_score', 0)
        
        if public_score < -20:
            recommendations.append({
                'priority': 'HIGH',
                'category': 'Sentiment',
                'issue': f'Negative public sentiment ({public_score:.1f}%)',
                'action': 'Crisis management needed: Address concerns, increase positive messaging, engage with critics',
                'impact': 'HIGH'
            })
        elif public_score > 20:
            recommendations.append({
                'priority': 'POSITIVE',
                'category': 'Sentiment',
                'issue': f'Positive public sentiment ({public_score:.1f}%)',
                'action': 'Amplify current strategy, increase content volume, leverage momentum',
                'impact': 'HIGH'
            })
        
        # Influence-based recommendations
        influence = all_analysis.get('influence_metrics', {})
        engagement_rate = influence.get('engagement_rate', 0)
        
        if engagement_rate < 1:
            recommendations.append({
                'priority': 'MEDIUM',
                'category': 'Engagement',
                'issue': f'Low engagement rate ({engagement_rate:.2f}%)',
                'action': 'Optimize posting times, increase multimedia content, boost direct interactions',
                'impact': 'MEDIUM'
            })
        elif engagement_rate > 3:
            recommendations.append({
                'priority': 'POSITIVE',
                'category': 'Engagement',
                'issue': f'Excellent engagement rate ({engagement_rate:.2f}%)',
                'action': 'Maintain current content strategy, consider scaling up',
                'impact': 'HIGH'
            })
        
        # Content strategy recommendations
        content = all_analysis.get('content_strategy', {})
        best_type = content.get('best_performing_type', 'unknown')
        
        if best_type != 'unknown':
            recommendations.append({
                'priority': 'OPPORTUNITY',
                'category': 'Content',
                'issue': f'{best_type.capitalize()} content performs best',
                'action': f'Increase {best_type} content frequency, optimize {best_type} strategy',
                'impact': 'MEDIUM'
            })
        
        # Temporal recommendations
        temporal = all_analysis.get('temporal_patterns', {})
        best_hours = temporal.get('best_posting_hours', [])
        
        if best_hours:
            top_hour = best_hours[0]['hour']
            recommendations.append({
                'priority': 'OPPORTUNITY',
                'category': 'Timing',
                'issue': f'Peak engagement at hour {top_hour}:00',
                'action': f'Schedule important content around {top_hour}:00, increase posting frequency during peak hours',
                'impact': 'MEDIUM'
            })
        
        # Hashtag recommendations
        hashtags = all_analysis.get('hashtag_performance', {})
        top_hashtags = hashtags.get('top_engaging_hashtags', [])
        
        if top_hashtags:
            top_tag = top_hashtags[0]['tag']
            recommendations.append({
                'priority': 'OPPORTUNITY',
                'category': 'Hashtags',
                'issue': f'#{top_tag} drives highest engagement',
                'action': f'Increase use of #{top_tag}, create campaigns around top-performing hashtags',
                'impact': 'LOW'
            })
        
        print(f"[OK] Generated {len(recommendations)} recommendations\n")
        return recommendations
    
    def analyze_comprehensive_data(self, filename):
        """Main analysis function - comprehensive insights extraction"""
        print("\n" + "="*80)
        print(f"  ANALYZING: {filename}")
        print("="*80 + "\n")
        
        # Load data
        print(f"[LOAD] Loading data from: {filename}\n")
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Prepare results structure
        analysis_results = {
            'analysis_timestamp': datetime.now().isoformat(),
            'source_file': filename,
            'politician': data.get('politician', 'Unknown'),
            'username': data.get('username', 'Unknown'),
            'original_data': data,
            'advanced_analysis': {}
        }
        
        # Extract tweet lists
        own_tweets_raw = data.get('user_tweets', {}).get('data', [])
        mentions_raw = data.get('mentions', {}).get('data', [])
        search_results_raw = data.get('search_results', {}).get('data', [])

        # Remove retweets so they aren't analyzed twice (original tweet + retweet)
        own_tweets = self._filter_retweets(own_tweets_raw, "own tweets")
        mentions = self._filter_retweets(mentions_raw, "mentions")
        search_results = self._filter_retweets(search_results_raw, "search results")
        
        # Deduplicate across sources (same tweet might appear in mentions AND search_results)
        own_tweets, mentions, search_results = self._deduplicate_tweets_across_sources(
            own_tweets, mentions, search_results
        )
        
        print(f"[DATA] Found {len(own_tweets)} own tweets, {len(mentions)} mentions, {len(search_results)} search results\n")
        
        # 1. Analyze own tweets
        analyzed_tweets = []
        if own_tweets:
            analyzed_tweets = self.analyze_tweets_batch(own_tweets, "own tweets")
        
        # 2. Analyze mentions
        analyzed_mentions = []
        if mentions:
            analyzed_mentions = self.analyze_tweets_batch(mentions, "mentions")
        
        # 3. Analyze search results
        analyzed_search = []
        if search_results:
            analyzed_search = self.analyze_tweets_batch(search_results, "search results")
        
        # Store inline metadata (tweets already updated in-place inside original_data)
        inline_counts = {
            'own_tweets': len(analyzed_tweets),
            'mentions': len(analyzed_mentions),
            'search_results': len(analyzed_search)
        }
        analysis_results['advanced_analysis']['analyzed_tweets'] = []
        analysis_results['advanced_analysis']['analyzed_mentions'] = []
        analysis_results['advanced_analysis']['analyzed_search'] = []
        analysis_results['advanced_analysis']['inline_enrichment'] = True
        analysis_results['advanced_analysis']['inline_counts'] = inline_counts
        
        # 4. Advanced topic modeling
        all_tweets = analyzed_tweets + analyzed_mentions
        if all_tweets:
            analysis_results['advanced_analysis']['topics'] = self.extract_topics_advanced(all_tweets)
        
        # 5. Temporal patterns
        if analyzed_tweets:
            analysis_results['advanced_analysis']['temporal_patterns'] = self.analyze_temporal_patterns(analyzed_tweets)
        
        # 6. Influence metrics
        if analyzed_tweets and data.get('user_profile'):
            analysis_results['advanced_analysis']['influence_metrics'] = self.analyze_influence_metrics(
                analyzed_tweets,
                data.get('user_profile')
            )
        
        # 7. Content strategy
        if analyzed_tweets:
            analysis_results['advanced_analysis']['content_strategy'] = self.analyze_content_strategy(analyzed_tweets)
        
        # 8. Hashtag performance
        if analyzed_tweets:
            analysis_results['advanced_analysis']['hashtag_performance'] = self.analyze_hashtag_performance(analyzed_tweets)
        
        # 9. Entity network
        if all_tweets:
            analysis_results['advanced_analysis']['entity_network'] = self.extract_entity_network(all_tweets)
        
        # 10. Sentiment summary
        analysis_results['advanced_analysis']['sentiment_summary'] = self.generate_sentiment_summary(
            analyzed_tweets, analyzed_mentions, analyzed_search
        )
        
        # 11. Emotion summary
        analysis_results['advanced_analysis']['emotion_summary'] = self.generate_emotion_summary(
            analyzed_tweets, analyzed_mentions
        )
        
        # 12. Generate recommendations
        analysis_results['advanced_analysis']['recommendations'] = self.generate_insights_recommendations(
            analysis_results['advanced_analysis']
        )
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f'mark_daou_advanced_analysis_{timestamp}.json'
        
        print("[SAVE] Saving comprehensive analysis...")
        # Convert numpy types to Python types for JSON serialization
        serializable_results = convert_to_serializable(analysis_results)
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        print("\n" + "="*80)
        print(f"[SUCCESS] ADVANCED ANALYSIS COMPLETE!")
        print(f"[FILE] Saved to: {output_filename}")
        print("="*80 + "\n")
        
        self._print_analysis_summary(analysis_results['advanced_analysis'])
        
        return output_filename, analysis_results
    
    def _print_analysis_summary(self, analysis):
        """Print comprehensive analysis summary"""
        print("\n[ANALYSIS SUMMARY]")
        print("-" * 80)
        
        # Helper to safely print strings with unicode
        def safe_print(text):
            try:
                print(text)
            except UnicodeEncodeError:
                # Fallback: print ASCII only
                print(text.encode('ascii', 'replace').decode('ascii'))
        
        # Sentiment
        sentiment = analysis.get('sentiment_summary', {})
        if sentiment:
            print("\n[SENTIMENT ANALYSIS]")
            for category, data in sentiment.items():
                if data.get('total', 0) > 0:
                    print(f"\n  {category.replace('_', ' ').title()}:")
                    print(f"    Total: {data['total']}")
                    print(f"    Positive: {data['positive']} ({data.get('positive_pct', 0):.1f}%)")
                    print(f"    Negative: {data['negative']} ({data.get('negative_pct', 0):.1f}%)")
                    print(f"    Neutral: {data['neutral']} ({data.get('neutral_pct', 0):.1f}%)")
                    print(f"    Sentiment Score: {data.get('sentiment_score', 0):.1f}%")
                    print(f"    Confidence: {data.get('avg_confidence', 0):.2f}")
        
        # Emotion
        emotion = analysis.get('emotion_summary', {})
        if emotion:
            print("\n[EMOTION ANALYSIS]")
            for category, data in emotion.items():
                print(f"\n  {category.replace('_', ' ').title()}:")
                print(f"    Dominant: {data.get('dominant', 'unknown')}")
                dist = data.get('distribution', {})
                for emo, count in sorted(dist.items(), key=lambda x: x[1], reverse=True)[:3]:
                    pct = data.get('percentages', {}).get(emo, 0)
                    print(f"    {emo}: {count} ({pct:.1f}%)")
        
        # Influence
        influence = analysis.get('influence_metrics', {})
        if influence:
            print("\n[INFLUENCE METRICS]")
            print(f"  Influence Tier: {influence.get('influence_tier', 'Unknown')}")
            print(f"  Influence Score: {influence.get('influence_score', 0):,.0f}")
            print(f"  Engagement Rate: {influence.get('engagement_rate', 0):.3f}%")
            print(f"  Virality Score: {influence.get('virality_score', 0):.2f}")
            print(f"  Reach Potential: {influence.get('reach_potential', 0):,} followers")
        
        # Content Strategy
        content = analysis.get('content_strategy', {})
        if content:
            print("\n[CONTENT STRATEGY]")
            print(f"  Best Performing Type: {content.get('best_performing_type', 'unknown')}")
            perf = content.get('performance_by_type', {})
            for ctype, eng in perf.items():
                print(f"    {ctype}: {eng:.1f} avg engagement")
        
        # Topics
        topics = analysis.get('topics', {})
        if topics:
            print("\n[TOPICS]")
            for name, data in list(topics.items())[:5]:
                safe_print(f"  {name}: {', '.join(data.get('keywords', [])[:5])}")
        
        # Recommendations
        recs = analysis.get('recommendations', [])
        if recs:
            print("\n[TOP RECOMMENDATIONS]")
            for i, rec in enumerate(recs[:5], 1):
                print(f"\n  {i}. [{rec['priority']}] {rec['category']} - Impact: {rec['impact']}")
                print(f"     Issue: {rec['issue']}")
                print(f"     Action: {rec['action']}")
        
        print("\n" + "-" * 80)


if __name__ == "__main__":
    import sys
    import glob
    
    print("\n" + "="*80)
    print("  ADVANCED INSIGHTS ANALYZER")
    print("  State-of-the-art AI models for comprehensive political analysis")
    print("="*80 + "\n")
    
    # Auto-find latest comprehensive file
    if len(sys.argv) < 2:
        comprehensive_files = glob.glob("mark_daou_comprehensive_*.json")
        
        if not comprehensive_files:
            print("[ERROR] No comprehensive data files found")
            print("[INFO] Please run: python comprehensive_fetcher.py first")
            sys.exit(1)
        
        # Get latest file
        filename = max(comprehensive_files, key=lambda x: Path(x).stat().st_mtime)
        print(f"[AUTO] Found latest file: {filename}\n")
    else:
        filename = sys.argv[1]
    
    # Run analysis
    analyzer = AdvancedInsightsAnalyzer()
    output_file, results = analyzer.analyze_comprehensive_data(filename)
    
    print(f"\n[SUCCESS] Analysis complete!")
    print(f"[OUTPUT] {output_file}")
    print("\n[NEXT STEPS]")
    print("  1. View dashboard: streamlit run enhanced_dashboard.py")
    print("  2. The dashboard will automatically load this analysis")
    print("\n")

