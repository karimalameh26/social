# 🇱🇧 Mark Daou - Political Intelligence Dashboard

A comprehensive political intelligence system that tracks Mark Daou's political activity, analyzes sentiment, monitors campaigns, and provides actionable insights.

---

## ⚡ Quick Start

```bash
# 1. Collect data + auto-generate insights
python comprehensive_fetcher.py

# 2. Track campaign hashtag (optional)
python hashtag_campaign_analyzer.py

# 3. Launch dashboard
streamlit run enhanced_dashboard.py
```

**API Usage:** 200-300 calls per run (0.06% of monthly quota)

---

## 🎯 Features

### Core Analytics
- ✅ Profile & campaign metrics
- ✅ Sentiment analysis (positive/negative/neutral)
- ✅ Public opinion tracking (-100 to +100 score)
- ✅ Engagement metrics (likes, retweets, replies)
- ✅ Hashtag & topic analysis

### Advanced Features
- ✅ **Automatic insight extraction** - Generates actionable recommendations
- ✅ **Campaign hashtag tracking** - Monitor campaign performance & health
- ✅ **Follower geography** - Lebanon vs Diaspora distribution
- ✅ **Follower demographics** - Age, influence, verified %
- ✅ **Follower segmentation** - Activity & influence levels
- ✅ **AI sentiment** (optional) - BERT + VADER + TextBlob

### Dashboard
- 📊 **14 interactive pages**
- 📈 Real-time visualizations
- 💡 Prioritized recommendations
- 🎯 Campaign health scoring

---

## 📊 System Overview

| Script | Purpose | API Calls |
|--------|---------|-----------|
| `comprehensive_fetcher.py` | Collect data + insights | 200 |
| `hashtag_campaign_analyzer.py` | Track campaigns | 100 |
| `analyze_data.py` | AI sentiment (optional) | 0 |
| `enhanced_dashboard.py` | Visualize everything | 0 |

**Total: 300 API calls = 0.06% of monthly quota**

---

## 📁 Project Structure

```
end/
├── comprehensive_fetcher.py          # Main data collector
├── hashtag_campaign_analyzer.py      # Campaign tracker
├── analyze_data.py                   # AI analytics (optional)
├── enhanced_dashboard.py             # Dashboard viewer
├── requirements.txt                  # Dependencies
├── START_HERE.md                     # Setup guide
├── GET_API_CREDENTIALS.md            # API setup
├── NEW_FEATURES_INSIGHTS_CAMPAIGNS.md # Feature documentation
├── QUICK_REFERENCE_UPDATED.md        # Quick reference
└── README.md                         # This file
```

---

## 🚀 Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Get API Credentials
See `GET_API_CREDENTIALS.md` for detailed instructions.

**Quick version:**
1. Go to https://developer.twitter.com/
2. Create free developer account
3. Create project and app
4. Generate Bearer Token
5. Save the token

### 3. Run Data Collection
```bash
python comprehensive_fetcher.py
```

**Enter:**
- Your Bearer Token
- Username: `DaouMark`

**Collects:**
- 50 recent tweets
- 50 mentions
- 100 followers with demographics
- Auto-generates insights

### 4. Launch Dashboard
```bash
streamlit run enhanced_dashboard.py
```

Opens at: `http://localhost:8501`

---

## 📊 Dashboard Pages

1. **Campaign Overview** - Profile metrics & reach
2. **Actionable Insights** - Auto-generated recommendations ⭐
3. **Campaign Tracker** - Hashtag campaign analysis ⭐
4. **Sentiment Analysis** - AI sentiment (BERT/VADER/TextBlob)
5. **Public Opinion** - Opinion gauge (-100 to +100)
6. **Engagement Metrics** - Likes, retweets, replies
7. **Hashtag Analysis** - Top hashtags with sentiment
8. **Topic Analysis** - LDA topic modeling
9. **Posting Patterns** - Best times/days
10. **Audience Analysis** - Engagement patterns
11. **Follower Geography** - Lebanon vs Diaspora split
12. **Follower Demographics** - Influential followers
13. **Follower Segmentation** - Activity & influence levels
14. **Raw Data** - Full JSON export

---

## 💡 Use Cases

### Daily Monitoring
```bash
python comprehensive_fetcher.py && streamlit run enhanced_dashboard.py
```

### Campaign Tracking
```bash
python hashtag_campaign_analyzer.py  # Enter campaign hashtag
streamlit run enhanced_dashboard.py  # View [HASHTAG] page
```

### Weekly Analysis
```bash
# Run Monday & Friday
python comprehensive_fetcher.py
python hashtag_campaign_analyzer.py

# Compare insights over time
```

---

## 🎯 Key Features Explained

### Automatic Insights
- Sentiment analysis (own tweets + public perception)
- Engagement performance metrics
- Follower quality analysis
- **Prioritized recommendations** (HIGH/MEDIUM/OPPORTUNITY/POSITIVE)

### Campaign Tracking
- Campaign health score (Excellent/Good/Moderate/Needs Improvement)
- Performance metrics (engagement, reach, impressions)
- Sentiment analysis per campaign
- Top performing tweets
- Actionable recommendations

### Follower Analytics
- **Geography:** Lebanon vs Diaspora distribution, top 20 locations
- **Demographics:** Verified %, account age, influential followers (10k+)
- **Segmentation:** Activity levels, influence tiers, political interest

---

## 📈 API Usage

### Free Tier Limits
- **500,000 calls/month** available
- **200-300 calls/run** = 0.06% usage
- Can run **1,600+ times/month**

### Recommended Frequency
- **Active campaigns:** 2-3x daily
- **Regular monitoring:** Daily
- **Maintenance:** Weekly

---

## 📚 Documentation

- **`START_HERE.md`** - Quick start guide
- **`GET_API_CREDENTIALS.md`** - API setup instructions
- **`NEW_FEATURES_INSIGHTS_CAMPAIGNS.md`** - Complete feature guide
- **`QUICK_REFERENCE_UPDATED.md`** - Commands & workflows

---

## 🔧 Requirements

- Python 3.8+
- X (Twitter) API Bearer Token (free tier)
- Dependencies in `requirements.txt`

---

## 📊 Sample Output

### Console (Insights)
```
[INSIGHTS SUMMARY]
==============================================================

[DATA] Analyzed:
   Reach: 38,152 followers
   Tweets: 50
   Mentions: 50

[SENTIMENT] Public Perception:
   Dominant: POSITIVE
   Sentiment Score: 12.5%

[RECOMMENDATIONS] Actionable Insights:
   1. [POSITIVE] Sentiment
      Finding: Public sentiment is positive (12.5%)
      Action: Maintain current messaging strategy
```

### Files Generated
```
mark_daou_comprehensive_20251120_153045.json  # Full data
mark_daou_insights_20251120_153045.json       # Insights ⭐
campaign_YallaRegister_20251120_154530.json   # Campaign data ⭐
```

---

## 🎯 System Capabilities

### Data Collection
- ✅ User profile & verification status
- ✅ Recent tweets (50)
- ✅ Mentions (50)
- ✅ Followers with location & demographics (100)
- ✅ Campaign hashtag tweets (100)

### Analysis
- ✅ Sentiment analysis (basic + AI optional)
- ✅ Engagement metrics
- ✅ Geographic distribution
- ✅ Follower segmentation
- ✅ Topic modeling
- ✅ Campaign performance

### Insights
- ✅ Auto-generated recommendations
- ✅ Priority-based action items
- ✅ Campaign health scoring
- ✅ Trend identification
- ✅ Opportunity detection

---

## 🆘 Support

### Common Issues

**No data in dashboard:**
```bash
# Run data collection first
python comprehensive_fetcher.py
```

**API rate limit:**
```bash
# Wait 15 minutes for reset
# Or reduce max_results in scripts
```

**Missing insights:**
```bash
# Check for insights JSON
ls mark_daou_insights_*.json

# Re-run if missing
python comprehensive_fetcher.py
```

---

## 📝 License

This is a political intelligence tool for monitoring and analysis purposes.

---

## 🎊 Credits

Built with:
- X (Twitter) API v2
- Streamlit (dashboard)
- Plotly (visualizations)
- BERT, VADER, TextBlob (sentiment analysis)
- scikit-learn (topic modeling)

---

**Made for Mark Daou's political campaign** 🇱🇧

**Quick Start:** `python comprehensive_fetcher.py && streamlit run enhanced_dashboard.py`
