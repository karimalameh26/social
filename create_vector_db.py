"""
ChromaDB Vector Database Builder for Political Insights

Creates embeddings from analyzed political data for semantic search and chat.
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import glob

class PoliticalInsightsVectorDB:
    """Build and manage vector database for political insights"""
    
    def __init__(self, db_path="./chroma_db"):
        """Initialize ChromaDB client and embedding model"""
        print("\n" + "="*80)
        print("  POLITICAL INSIGHTS VECTOR DATABASE BUILDER")
        print("="*80 + "\n")
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=db_path)
        
        # Initialize embedding model (using sentence-transformers for better semantic understanding)
        print("[LOAD] Loading embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("[OK] Embedding model loaded\n")
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name="political_insights",
            metadata={"description": "Political campaign insights and analytics"}
        )
    
    def load_latest_analysis(self) -> Dict[str, Any]:
        """Load the latest advanced analysis file"""
        analysis_files = glob.glob("*_advanced_analysis_*.json")
        
        if not analysis_files:
            raise FileNotFoundError("No advanced analysis files found. Run advanced_insights_analyzer.py first!")
        
        # Get most recent file
        latest_file = max(analysis_files, key=os.path.getmtime)
        print(f"[LOAD] Loading: {latest_file}\n")
        
        with open(latest_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def create_tweet_documents(self, data: Dict[str, Any]) -> tuple:
        """Create documents from all tweets (own, mentions, search results)"""
        print("[PROCESS] Creating tweet embeddings...")
        
        documents = []
        metadatas = []
        ids = []
        
        politician = data.get('politician', 'Unknown')
        username = data.get('username', 'Unknown')
        
        # Process own tweets
        analyzed_tweets = data.get('advanced_analysis', {}).get('analyzed_tweets', [])
        analyzed_mentions = data.get('advanced_analysis', {}).get('analyzed_mentions', [])
        analyzed_search = data.get('advanced_analysis', {}).get('analyzed_search', [])
        
        # Combine all tweets with their source
        all_tweets = [
            {'source': 'own_tweet', 'data': tweet, 'idx': idx} 
            for idx, tweet in enumerate(analyzed_tweets)
        ]
        all_tweets.extend([
            {'source': 'mention', 'data': tweet, 'idx': idx} 
            for idx, tweet in enumerate(analyzed_mentions)
        ])
        all_tweets.extend([
            {'source': 'search_result', 'data': tweet, 'idx': idx} 
            for idx, tweet in enumerate(analyzed_search)
        ])
        
        for item in all_tweets:
            source = item['source']
            tweet = item['data']
            idx = item['idx']
            
            tweet_text = tweet.get('text', '')
            if not tweet_text or len(tweet_text) < 10:
                continue
            
            # Get author info
            author_name = tweet.get('author_name', politician if source == 'own_tweet' else 'Unknown')
            author_username = tweet.get('author_username', username if source == 'own_tweet' else 'Unknown')
            
            # Get metrics
            metrics = tweet.get('public_metrics', {})
            
            # Get AI analysis from separate fields if they exist
            sentiment_data = {}
            emotion_data = {}
            
            # Try to get AI fields if they exist in the tweet
            if 'ai_sentiment' in tweet:
                sentiment_data = tweet['ai_sentiment']
            if 'ai_emotion' in tweet:
                emotion_data = tweet['ai_emotion']
            
            # Calculate engagement score
            engagement = (
                metrics.get('like_count', 0) +
                metrics.get('retweet_count', 0) * 2 +
                metrics.get('reply_count', 0) * 3
            )
            
            # Create searchable document
            sentiment_str = sentiment_data.get('sentiment', 'unknown') if sentiment_data else 'unknown'
            sentiment_conf = sentiment_data.get('confidence', 0) if sentiment_data else 0
            emotion_str = emotion_data.get('emotion', 'unknown') if emotion_data else 'unknown'
            
            # Different doc format based on source
            if source == 'own_tweet':
                doc_text = f"""
                Tweet by {author_name} (@{author_username}):
                {tweet_text}
                
                Source: Own tweet
                Sentiment: {sentiment_str} (confidence: {sentiment_conf:.2%})
                Emotion: {emotion_str}
                Engagement: {engagement} (likes: {metrics.get('like_count', 0)}, retweets: {metrics.get('retweet_count', 0)}, replies: {metrics.get('reply_count', 0)})
                """
            elif source == 'mention':
                doc_text = f"""
                Mention of {politician} by @{author_username}:
                {tweet_text}
                
                Source: Mention
                Sentiment: {sentiment_str} (confidence: {sentiment_conf:.2%})
                Emotion: {emotion_str}
                Engagement: {engagement} (likes: {metrics.get('like_count', 0)}, retweets: {metrics.get('retweet_count', 0)}, replies: {metrics.get('reply_count', 0)})
                """
            else:  # search_result
                doc_text = f"""
                Search result about {politician} by @{author_username}:
                {tweet_text}
                
                Source: Search result
                Sentiment: {sentiment_str} (confidence: {sentiment_conf:.2%})
                Emotion: {emotion_str}
                Engagement: {engagement} (likes: {metrics.get('like_count', 0)}, retweets: {metrics.get('retweet_count', 0)}, replies: {metrics.get('reply_count', 0)})
                """
            
            documents.append(doc_text.strip())
            metadatas.append({
                'type': 'tweet',
                'tweet_source': source,
                'politician': politician,
                'username': username,
                'author_name': author_name,
                'author_username': author_username,
                'tweet_id': tweet.get('id', f'{source}_{idx}'),
                'text': tweet_text,
                'sentiment': sentiment_str,
                'sentiment_score': float(sentiment_conf),
                'emotion': emotion_str,
                'engagement_score': engagement,
                'like_count': metrics.get('like_count', 0),
                'retweet_count': metrics.get('retweet_count', 0),
                'reply_count': metrics.get('reply_count', 0),
                'impression_count': metrics.get('impression_count', 0),
                'created_at': tweet.get('created_at', ''),
                'lang': tweet.get('lang', ''),
                'has_entities': 'entities' in tweet
            })
            ids.append(f"{source}_{idx}")
        
        # Count by source
        own_count = sum(1 for m in metadatas if m.get('tweet_source') == 'own_tweet')
        mention_count = sum(1 for m in metadatas if m.get('tweet_source') == 'mention')
        search_count = sum(1 for m in metadatas if m.get('tweet_source') == 'search_result')
        
        print(f"[OK] Created {len(documents)} tweet embeddings")
        print(f"     Own tweets: {own_count}")
        print(f"     Mentions: {mention_count}")
        print(f"     Search results: {search_count}\n")
        return documents, metadatas, ids
    
    def create_topic_documents(self, data: Dict[str, Any]) -> tuple:
        """Create documents from topic analysis"""
        print("[PROCESS] Creating topic embeddings...")
        
        documents = []
        metadatas = []
        ids = []
        
        politician = data.get('politician', 'Unknown')
        topics = data.get('advanced_analysis', {}).get('topics', {})
        
        for topic_name, topic_info in topics.items():
            keywords = topic_info.get('keywords', [])
            all_keywords = topic_info.get('all_keywords', [])
            coherence = topic_info.get('coherence', 0)
            
            # Create searchable document
            doc_text = f"""
            Topic from {politician}'s discourse: {topic_name}
            
            Main keywords: {', '.join(keywords)}
            Related terms: {', '.join(all_keywords)}
            Coherence score: {coherence:.3f}
            
            This topic represents a theme in the political conversation covering: {', '.join(keywords[:5])}
            """
            
            documents.append(doc_text.strip())
            metadatas.append({
                'type': 'topic',
                'politician': politician,
                'topic_id': topic_name,
                'keywords': ', '.join(keywords),
                'all_keywords': ', '.join(all_keywords),
                'coherence': float(coherence),
                'num_keywords': len(keywords)
            })
            ids.append(f"topic_{topic_name}")
        
        print(f"[OK] Created {len(documents)} topic embeddings\n")
        return documents, metadatas, ids
    
    def create_entity_documents(self, data: Dict[str, Any]) -> tuple:
        """Create documents from entity network"""
        print("[PROCESS] Creating entity embeddings...")
        
        documents = []
        metadatas = []
        ids = []
        
        politician = data.get('politician', 'Unknown')
        entities = data.get('advanced_analysis', {}).get('entity_network', {})
        
        for entity_name, entity_info in list(entities.items())[:30]:  # Top 30 entities
            entity_type = ', '.join(entity_info.get('types', []))
            count = entity_info.get('count', 0)
            sentiment_info = entity_info.get('sentiment', {})
            dominant_sentiment = entity_info.get('dominant_sentiment', 'neutral')
            
            # Create searchable document
            doc_text = f"""
            Entity mentioned by {politician}: {entity_name}
            
            Type: {entity_type}
            Mentioned {count} times
            Sentiment: {dominant_sentiment} (positive: {sentiment_info.get('positive', 0)}, negative: {sentiment_info.get('negative', 0)}, neutral: {sentiment_info.get('neutral', 0)})
            
            This entity is a {entity_type} that appears frequently in {politician}'s political discourse.
            """
            
            documents.append(doc_text.strip())
            metadatas.append({
                'type': 'entity',
                'politician': politician,
                'entity_name': entity_name,
                'entity_type': entity_type,
                'mention_count': count,
                'sentiment': dominant_sentiment,
                'positive_mentions': sentiment_info.get('positive', 0),
                'negative_mentions': sentiment_info.get('negative', 0),
                'neutral_mentions': sentiment_info.get('neutral', 0)
            })
            ids.append(f"entity_{entity_name.replace(' ', '_')}")
        
        print(f"[OK] Created {len(documents)} entity embeddings\n")
        return documents, metadatas, ids
    
    def create_insight_documents(self, data: Dict[str, Any]) -> tuple:
        """Create documents from insights and recommendations"""
        print("[PROCESS] Creating insight embeddings...")
        
        documents = []
        metadatas = []
        ids = []
        
        politician = data.get('politician', 'Unknown')
        recommendations = data.get('advanced_analysis', {}).get('recommendations', [])
        
        for idx, rec in enumerate(recommendations):
            priority = rec.get('priority', 'MEDIUM')
            category = rec.get('category', 'General')
            issue = rec.get('issue', '')
            action = rec.get('action', '')
            impact = rec.get('impact', 'MEDIUM')
            
            # Create searchable document
            doc_text = f"""
            Recommendation for {politician}:
            
            Priority: {priority}
            Category: {category}
            Impact: {impact}
            
            Issue: {issue}
            Recommended Action: {action}
            """
            
            documents.append(doc_text.strip())
            metadatas.append({
                'type': 'recommendation',
                'politician': politician,
                'priority': priority,
                'category': category,
                'impact': impact,
                'issue': issue,
                'action': action
            })
            ids.append(f"recommendation_{idx}")
        
        # Add sentiment summary
        sentiment_summary = data.get('advanced_analysis', {}).get('sentiment_summary', {})
        
        for source in ['own_tweets', 'mentions', 'search_results']:
            if source in sentiment_summary:
                summary = sentiment_summary[source]
                
                doc_text = f"""
                Sentiment analysis for {politician}'s {source.replace('_', ' ')}:
                
                Total analyzed: {summary.get('total', 0)}
                Positive: {summary.get('positive', 0)} ({summary.get('positive_pct', 0):.1f}%)
                Negative: {summary.get('negative', 0)} ({summary.get('negative_pct', 0):.1f}%)
                Neutral: {summary.get('neutral', 0)} ({summary.get('neutral_pct', 0):.1f}%)
                
                Overall sentiment score: {summary.get('sentiment_score', 0):.1f}%
                Average confidence: {summary.get('avg_confidence', 0):.1%}
                """
                
                documents.append(doc_text.strip())
                metadatas.append({
                    'type': 'sentiment_summary',
                    'politician': politician,
                    'source': source,
                    'total': summary.get('total', 0),
                    'positive': summary.get('positive', 0),
                    'negative': summary.get('negative', 0),
                    'neutral': summary.get('neutral', 0),
                    'sentiment_score': float(summary.get('sentiment_score', 0)),
                    'avg_confidence': float(summary.get('avg_confidence', 0))
                })
                ids.append(f"sentiment_{source}")
        
        # Add influence metrics
        influence = data.get('advanced_analysis', {}).get('influence_metrics', {})
        if influence:
            doc_text = f"""
            Influence metrics for {politician}:
            
            Influence tier: {influence.get('tier', 'Unknown')}
            Influence score: {influence.get('influence_score', 0):,}
            Engagement rate: {influence.get('engagement_rate', 0):.3f}%
            Virality score: {influence.get('virality_score', 0):.2f}
            Reach potential: {influence.get('reach_potential', 0):,} followers
            """
            
            documents.append(doc_text.strip())
            metadatas.append({
                'type': 'influence_metrics',
                'politician': politician,
                'influence_tier': influence.get('tier', 'Unknown'),
                'influence_score': influence.get('influence_score', 0),
                'engagement_rate': float(influence.get('engagement_rate', 0)),
                'virality_score': float(influence.get('virality_score', 0)),
                'reach_potential': influence.get('reach_potential', 0)
            })
            ids.append("influence_metrics")
        
        # Add content strategy
        content = data.get('advanced_analysis', {}).get('content_strategy', {})
        if content:
            best_type = content.get('best_performing_type', 'unknown')
            performance = content.get('performance_by_type', {})
            
            doc_text = f"""
            Content strategy for {politician}:
            
            Best performing content type: {best_type}
            """
            
            if performance:
                doc_text += "\n\nPerformance breakdown:"
                for ctype, value in performance.items():
                    # Handle both dict and float values
                    if isinstance(value, dict):
                        avg_eng = value.get('avg_engagement', 0)
                    else:
                        avg_eng = value
                    doc_text += f"\n- {ctype}: {avg_eng:.1f} average engagement"
            
            documents.append(doc_text.strip())
            metadatas.append({
                'type': 'content_strategy',
                'politician': politician,
                'best_performing_type': best_type,
                'performance_data': json.dumps(performance) if performance else ''
            })
            ids.append("content_strategy")
        
        print(f"[OK] Created {len(documents)} insight embeddings\n")
        return documents, metadatas, ids
    
    def create_hashtag_documents(self, data: Dict[str, Any]) -> tuple:
        """Create documents from hashtag performance"""
        print("[PROCESS] Creating hashtag embeddings...")
        
        documents = []
        metadatas = []
        ids = []
        
        politician = data.get('politician', 'Unknown')
        hashtags = data.get('advanced_analysis', {}).get('hashtag_performance', {})
        
        top_hashtags = hashtags.get('top_engaging_hashtags', [])[:10]  # Top 10
        
        for idx, tag_info in enumerate(top_hashtags):
            tag = tag_info.get('tag', '')
            count = tag_info.get('count', 0)
            avg_engagement = tag_info.get('avg_engagement', 0)
            
            doc_text = f"""
            Hashtag used by {politician}: #{tag}
            
            Used {count} times
            Average engagement: {avg_engagement:.1f}
            
            This hashtag is part of {politician}'s campaign strategy.
            """
            
            documents.append(doc_text.strip())
            metadatas.append({
                'type': 'hashtag',
                'politician': politician,
                'hashtag': tag,
                'count': count,
                'avg_engagement': float(avg_engagement)
            })
            ids.append(f"hashtag_{tag.replace('#', '')}")
        
        print(f"[OK] Created {len(documents)} hashtag embeddings\n")
        return documents, metadatas, ids
    
    def build_database(self, analysis_file: str = None):
        """Build complete vector database from analysis file"""
        # Load data
        if analysis_file:
            with open(analysis_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            data = self.load_latest_analysis()
        
        politician = data.get('politician', 'Unknown')
        print(f"[INFO] Building vector database for: {politician}\n")
        
        # Clear existing collection
        try:
            self.client.delete_collection("political_insights")
            self.collection = self.client.create_collection(
                name="political_insights",
                metadata={"description": "Political campaign insights and analytics"}
            )
            print("[INFO] Cleared existing database\n")
        except:
            pass
        
        # Collect all documents
        all_documents = []
        all_metadatas = []
        all_ids = []
        
        # Create documents from different sources
        docs, metas, ids = self.create_tweet_documents(data)
        all_documents.extend(docs)
        all_metadatas.extend(metas)
        all_ids.extend(ids)
        
        docs, metas, ids = self.create_topic_documents(data)
        all_documents.extend(docs)
        all_metadatas.extend(metas)
        all_ids.extend(ids)
        
        docs, metas, ids = self.create_entity_documents(data)
        all_documents.extend(docs)
        all_metadatas.extend(metas)
        all_ids.extend(ids)
        
        docs, metas, ids = self.create_insight_documents(data)
        all_documents.extend(docs)
        all_metadatas.extend(metas)
        all_ids.extend(ids)
        
        docs, metas, ids = self.create_hashtag_documents(data)
        all_documents.extend(docs)
        all_metadatas.extend(metas)
        all_ids.extend(ids)
        
        # Generate embeddings and add to ChromaDB
        print(f"[EMBED] Generating embeddings for {len(all_documents)} documents...")
        
        # Process in batches
        batch_size = 100
        for i in range(0, len(all_documents), batch_size):
            batch_docs = all_documents[i:i+batch_size]
            batch_metas = all_metadatas[i:i+batch_size]
            batch_ids = all_ids[i:i+batch_size]
            
            # Generate embeddings
            embeddings = self.embedding_model.encode(batch_docs).tolist()
            
            # Add to collection
            self.collection.add(
                documents=batch_docs,
                embeddings=embeddings,
                metadatas=batch_metas,
                ids=batch_ids
            )
            
            print(f"  Processed {min(i+batch_size, len(all_documents))}/{len(all_documents)} documents")
        
        print("\n" + "="*80)
        print("[SUCCESS] Vector database created!")
        print(f"[INFO] Total documents: {len(all_documents)}")
        print(f"[INFO] Database location: ./chroma_db")
        print("="*80 + "\n")
        
        return len(all_documents)

def main():
    """Main execution"""
    import sys
    
    # Create vector database
    db = PoliticalInsightsVectorDB()
    
    # Build database from latest or specified file
    analysis_file = sys.argv[1] if len(sys.argv) > 1 else None
    total_docs = db.build_database(analysis_file)
    
    print("[NEXT STEPS]")
    print("1. Run: python chat_with_insights.py")
    print("2. Ask questions like:")
    print("   - 'What is the top performing tweet?'")
    print("   - 'What are the highest sentiment tweets?'")
    print("   - 'What are the most engaging topics?'")
    print("   - 'Show me negative sentiment tweets'")
    print("   - 'What entities are mentioned most?'")

if __name__ == "__main__":
    main()

