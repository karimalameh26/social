# Vector Database - Setup Complete

## What Was Created

A **ChromaDB vector database** with embeddings from Mark Daou's political insights.

### Database Contents:
- **199 tweets** total:
  - **49 own tweets** by Mark Daou
  - **50 mentions** of Mark Daou
  - **100 search results** about Mark Daou
- **5 topics** from LDA analysis
- **30 entities** (people, organizations, locations)
- **9 insights** (sentiment summaries, influence metrics, recommendations)
- **6 hashtags** with performance data

**Total**: 249 searchable documents

---

## Database Location

`./chroma_db/` directory

---

## How to Use

### Rebuild Database (after new analysis)

```bash
python create_vector_db.py
```

Or specify a file:
```bash
python create_vector_db.py mark_daou_advanced_analysis_YYYYMMDD_HHMMSS.json
```

---

## What's Embedded

### 1. Tweets (All Sources)
Each tweet includes:
- **Source**: `own_tweet`, `mention`, or `search_result`
- **Author**: Name and username (politician for own tweets, others for mentions/search)
- Full text
- Engagement score (calculated from likes, retweets, replies)
- Like count, retweet count, reply count, impressions
- Sentiment and confidence (from AI analysis)
- Emotion (from AI analysis)
- Language, timestamp
- Tweet ID for reference

**Filter by source**:
- Own tweets: What Mark Daou posted
- Mentions: What others said about Mark Daou
- Search results: Related discussions about Mark Daou

### 2. Topics
Each topic includes:
- Topic ID
- Main keywords (top 5)
- All keywords (extended list)
- Coherence score

### 3. Entities
Each entity includes:
- Entity name (e.g., "Mark Daou", "حزب الله", "Lebanon")
- Type (Person, Organization, Location, Misc)
- Mention count
- Sentiment breakdown (positive/negative/neutral)

### 4. Insights
Includes:
- **Sentiment summaries** (for own tweets, mentions, search results)
- **Influence metrics** (influence score, engagement rate, virality, reach)
- **Content strategy** (best performing content type)
- **Recommendations** (priority, category, issue, action)

### 5. Hashtags
Each hashtag includes:
- Hashtag name
- Usage count
- Average engagement

---

## Metadata Fields

All documents have:
- `type`: tweet, topic, entity, recommendation, sentiment_summary, influence_metrics, hashtag
- `politician`: "Mark Daou"
- `username`: "@DaouMark"

Plus type-specific fields for filtering and sorting.

---

## Technical Details

### Embedding Model
- **Model**: `all-MiniLM-L6-v2` (sentence-transformers)
- **Dimensions**: 384
- **Speed**: Fast (<1 second per query)
- **Offline**: Works without internet

### Vector Database
- **Engine**: ChromaDB
- **Storage**: Persistent (saved to disk)
- **Search**: Semantic similarity search
- **Filtering**: Rich metadata support

---

## Next Steps

The database is ready for chatbot integration. You can now:

1. **Query by similarity**: Find documents similar to a question
2. **Filter by type**: Get only tweets, topics, entities, etc.
3. **Sort by metadata**: Find top engagement, highest sentiment, etc.
4. **Semantic search**: Understand meaning, not just keywords

---

## Example Queries (for future chatbot)

```python
# Find similar documents
results = collection.query(
    query_texts=["What is the top performing tweet?"],
    n_results=5
)

# Filter by type
results = collection.query(
    query_texts=["Show me topics about Lebanon"],
    n_results=3,
    where={"type": "topic"}
)

# Filter by tweet source
own_tweets = collection.query(
    query_texts=["What did Mark Daou say about the economy?"],
    n_results=5,
    where={"type": "tweet", "tweet_source": "own_tweet"}
)

mentions = collection.query(
    query_texts=["What are people saying about Mark Daou?"],
    n_results=5,
    where={"type": "tweet", "tweet_source": "mention"}
)

search_results = collection.query(
    query_texts=["Show me discussions about Mark Daou"],
    n_results=5,
    where={"type": "tweet", "tweet_source": "search_result"}
)

# Sort by engagement
tweets = collection.get(where={"type": "tweet"})
sorted_tweets = sorted(tweets['metadatas'], 
                      key=lambda x: x['engagement_score'], 
                      reverse=True)

# Find negative mentions
negative_mentions = collection.get(
    where={
        "type": "tweet",
        "tweet_source": "mention",
        "sentiment": "negative"
    }
)
```

---

## Files

- `create_vector_db.py` - Database builder script
- `./chroma_db/` - Database storage (auto-created)
- `requirements.txt` - Includes chromadb and sentence-transformers

---

## Performance

- **Database creation**: ~30-60 seconds
- **Query time**: <1 second
- **Memory usage**: ~500MB (model + data)
- **Disk usage**: ~100MB (database + embeddings)

---

**Status**: ✅ Database ready for chatbot integration!

