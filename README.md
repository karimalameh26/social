# Mark Daou Political Intelligence Dashboard

**Comprehensive sentiment analysis, campaign tracking, and public opinion monitoring for Lebanese politician Mark Daou using X (Twitter) API v2.**

## 🎯 Features

### Sentiment Analysis
- ✅ Sentiment classification (positive/negative/neutral)
- ✅ Public opinion tracking
- ✅ Mention sentiment analysis
- ✅ Campaign message sentiment

### Campaign Tracking
- ✅ Reach and influence metrics
- ✅ Engagement analytics
- ✅ Hashtag performance
- ✅ Posting pattern analysis
- ✅ Top performing content

### Public Opinion Monitoring
- ✅ Public opinion score (-100 to +100)
- ✅ Sentiment trends over time
- ✅ Geographic distribution
- ✅ Audience demographics
- ✅ Follower analysis

### Advanced Analytics
- ✅ Topic detection (with elevated access)
- ✅ Engagement rate calculation
- ✅ Influence scoring
- ✅ Posting time optimization
- ✅ Hashtag trending

## 📊 X API v2 Endpoints Used

This system uses multiple endpoints for comprehensive data collection:

### Core Endpoints (Free Tier)
1. **GET /2/users/by/username/:username** - User lookup
2. **GET /2/users/:id** - User profile with metrics
3. **GET /2/users/:id/tweets** - User's tweets
4. **GET /2/users/:id/mentions** - Mentions of the user
5. **GET /2/tweets/search/recent** - Search tweets (7 days)
6. **GET /2/users/:id/followers** - Follower information
7. **GET /2/users/:id/following** - Following information

### Data Fields Retrieved
- **public_metrics** - Likes, retweets, replies, quotes, impressions
- **entities** - Hashtags, mentions, URLs
- **created_at** - Timestamps for trend analysis
- **context_annotations** - Topic detection (requires elevated access)
- **user.fields** - Profile, verification, location
- **media.fields** - Attachments, images, videos

## 🚀 Quick Start

### Step 1: Get X API Credentials

Follow the detailed guide in `GET_API_CREDENTIALS.md`:

1. Create Twitter Developer account at https://developer.twitter.com/
2. Create a Project and App
3. Generate Bearer Token
4. Copy and save your token

**The FREE tier is sufficient for this project!**

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

Dependencies:
- streamlit - Dashboard framework
- pandas - Data manipulation
- plotly - Interactive visualizations
- requests - API calls
- python-docx - Read endpoint docs

### Step 3: Collect Data

```bash
python comprehensive_fetcher.py
```

When prompted:
1. Enter your X API Bearer Token
2. Enter Mark Daou's X username (e.g., `markdaou_lb`)

The script will:
- ✅ Fetch user profile and metrics
- ✅ Collect recent tweets (up to 100)
- ✅ Gather mentions (up to 100)
- ✅ Search for related tweets
- ✅ Sample followers and following
- ✅ Analyze sentiment
- ✅ Extract hashtags and topics
- ✅ Calculate engagement metrics
- ✅ Analyze posting patterns

Data is saved to: `mark_daou_comprehensive_YYYYMMDD_HHMMSS.json`

### Step 4: Launch Dashboard

```bash
streamlit run enhanced_dashboard.py
```

Dashboard opens at: **http://localhost:8501**

## 📊 Dashboard Pages

### 1. Campaign Overview
- Total reach (followers)
- Engagement rate
- Influence score
- Reach ratio
- Key performance indicators

### 2. Sentiment Analysis
- Sentiment in mentions
- Sentiment in search results
- Sentiment in own tweets
- Visual pie charts
- Percentage breakdowns

### 3. Public Opinion
- Public opinion score gauge (-100 to +100)
- Overall sentiment classification
- Response breakdown
- Trend indicators

### 4. Engagement Metrics
- Total and average engagement
- Likes, retweets, replies, quotes
- Top performing tweet
- Engagement comparisons

### 5. Hashtag Analysis
- Top hashtags used
- Hashtag frequency
- Trending campaign tags
- Mention hashtags

### 6. Topic Analysis
- Auto-detected topics (requires elevated API access)
- Topic distribution
- Trending topics in mentions

### 7. Posting Patterns
- Most active hours
- Most active days
- Posting optimization insights

### 8. Audience Analysis
- Follower demographics
- Verified follower ratio
- Geographic distribution
- Following analysis

## 📈 Metrics Explained

### Engagement Rate
```
(Total Likes + Total Retweets) / Total Tweets
```

### Influence Score
```
(Followers × 0.7) + (Total Tweets × 0.3)
```

### Public Opinion Score
```
((Positive - Negative) / Total) × 100
```
Range: -100 (all negative) to +100 (all positive)

### Reach Ratio
```
Followers / Following
```
Higher ratio indicates greater influence

## 🔄 Data Collection Schedule

### Recommended Frequency
- **Daily**: For active campaigns
- **Weekly**: For regular monitoring
- **Real-time**: Use streaming endpoints (see docs)

### Rate Limits (Free Tier)
- User lookup: 300 requests / 15 min
- Search recent: 450 requests / 15 min
- User tweets: 1,500 requests / 15 min
- Mentions: 450 requests / 15 min

The script handles rate limits automatically with wait periods.

## 🔐 Security Best Practices

### Never Commit Credentials
```bash
# Add to .gitignore
echo "*.env" >> .gitignore
echo "*_token.txt" >> .gitignore
echo "mark_daou_comprehensive_*.json" >> .gitignore
```

### Use Environment Variables
```bash
# Create .env file
TWITTER_BEARER_TOKEN=your_token_here
```

Then in Python:
```python
import os
from dotenv import load_dotenv

load_dotenv()
bearer_token = os.getenv('TWITTER_BEARER_TOKEN')
```

## 📁 File Structure

```
.
├── comprehensive_fetcher.py    # Main data collection script
├── enhanced_dashboard.py       # Sentiment & campaign dashboard
├── extract_endpoints.py        # Read endpoint documentation
├── GET_API_CREDENTIALS.md      # How to get X API access
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── X EndPoints 1.docx          # API documentation
└── X EndPoints 2.docx          # API documentation
```

## 🎓 Advanced Features

### Sentiment Analysis Algorithm
Uses keyword-based classification:
- **Positive**: reform, progress, support, hope, success, etc.
- **Negative**: corrupt, crisis, fail, disaster, problem, etc.
- **Neutral**: discuss, announce, statement, update, etc.

### For Better Sentiment Analysis
Consider integrating:
- **TextBlob** - Python sentiment analysis
- **VADER** - Social media sentiment
- **Transformers** - Deep learning models
- **Arabic NLP** - For Arabic tweets

### Streaming for Real-Time Data
Upgrade script to use:
- `GET /2/tweets/search/stream` - Filtered stream
- `POST /2/tweets/search/stream/rules` - Set filters

### Full Archive Search
Requires Pro tier ($5,000/month):
- `GET /2/tweets/search/all` - All tweets since 2006

## 🛠️ Troubleshooting

### "Could not authenticate you"
- Check Bearer Token is correct
- Ensure no extra spaces
- Regenerate token if needed

### "No data available" in dashboard
- Run `comprehensive_fetcher.py` first
- Check username is correct
- Verify API credentials

### Rate limit errors
- Script waits automatically
- Reduce max_results if needed
- Space out data collection

### Missing context annotations
- Requires elevated API access
- Apply for elevated (free but needs approval)
- Or upgrade to Basic/Pro tier

## 📚 Resources

- **X API Documentation**: https://developer.twitter.com/en/docs/twitter-api
- **Developer Portal**: https://developer.twitter.com/en/portal/dashboard
- **Rate Limits**: https://developer.twitter.com/en/docs/twitter-api/rate-limits
- **Community Forum**: https://twittercommunity.com/

## 🔄 Future Enhancements

- [ ] Arabic text sentiment analysis
- [ ] Historical trend tracking
- [ ] Automated reporting (PDF/email)
- [ ] Real-time streaming dashboard
- [ ] Competitor comparison
- [ ] Network analysis visualization
- [ ] Predictive analytics
- [ ] Media content analysis
- [ ] Scheduled data collection
- [ ] Multi-language support

## 📞 Support

For issues:
1. Check `GET_API_CREDENTIALS.md` for API setup
2. Review error messages carefully
3. Verify API rate limits
4. Check X API status page

## ⚖️ License & Ethics

This tool is for:
- ✅ Political research
- ✅ Campaign monitoring
- ✅ Public opinion analysis
- ✅ Transparency and accountability

Please use responsibly and respect:
- X Terms of Service
- Privacy regulations
- Data protection laws
- Ethical research practices

---

**Ready to start?**

1. Get credentials: See `GET_API_CREDENTIALS.md`
2. Collect data: `python comprehensive_fetcher.py`
3. View dashboard: `streamlit run enhanced_dashboard.py`

🇱🇧 **Monitor Mark Daou's political activity with powerful analytics!**

