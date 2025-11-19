# Quick Reference - Mark Daou Dashboard

## 🚀 Quick Start Commands

### First Time Setup
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Collect data
python comprehensive_fetcher.py
# Enter: Bearer Token
# Enter: markdaou_lb (or his actual username)

# 3. Launch dashboard
streamlit run enhanced_dashboard.py
```

### Regular Updates
```bash
# Collect fresh data
python comprehensive_fetcher.py

# Restart dashboard
streamlit run enhanced_dashboard.py
```

---

## 📊 X API v2 Endpoints Used

| Endpoint | Purpose | Data Collected |
|----------|---------|----------------|
| `GET /2/users/by/username/:username` | Find user | Profile, metrics, verification |
| `GET /2/users/:id` | User details | Complete profile info |
| `GET /2/users/:id/tweets` | User tweets | Recent 100 tweets |
| `GET /2/users/:id/mentions` | Mentions | Tweets mentioning user |
| `GET /2/tweets/search/recent` | Search tweets | Last 7 days of tweets |
| `GET /2/users/:id/followers` | Followers | Sample of followers |
| `GET /2/users/:id/following` | Following | Accounts followed |

---

## 📈 Dashboard Pages

| Page | What It Shows |
|------|---------------|
| 📊 Campaign Overview | Reach, influence, engagement rate |
| 😊 Sentiment Analysis | Positive/negative/neutral breakdown |
| 🗳️ Public Opinion | Opinion score (-100 to +100) |
| 📈 Engagement Metrics | Likes, retweets, top tweets |
| 🏷️ Hashtag Analysis | Trending hashtags used |
| 💡 Topic Analysis | Auto-detected topics |
| ⏰ Posting Patterns | Best times to post |
| 👥 Audience Analysis | Follower demographics |
| 📄 Raw Data | Complete JSON export |

---

## 🔑 Where to Get API Credentials

1. **Developer Portal**: https://developer.twitter.com/en/portal/dashboard
2. **Create Account** → **Create Project** → **Create App**
3. **Keys and Tokens** → **Generate Bearer Token**
4. **Copy and save** the token immediately

---

## 💡 Key Metrics Explained

### Public Opinion Score
```
((Positive - Negative) / Total) × 100
```
- **+100**: All positive sentiment
- **0**: Balanced sentiment
- **-100**: All negative sentiment

### Influence Score
```
(Followers × 0.7) + (Tweets × 0.3)
```
Higher = more influential

### Engagement Rate
```
(Likes + Retweets) / Total Tweets
```
Higher = better engagement

---

## 🔄 Data Files

### Input Files
- `X EndPoints 1.docx` - API documentation
- `X EndPoints 2.docx` - Additional endpoints

### Output Files
- `mark_daou_comprehensive_YYYYMMDD_HHMMSS.json` - Collected data

### Python Scripts
- `comprehensive_fetcher.py` - Data collector
- `enhanced_dashboard.py` - Dashboard app
- `extract_endpoints.py` - Read docs

### Documentation
- `README.md` - Main documentation
- `SETUP_GUIDE.md` - Step-by-step setup
- `GET_API_CREDENTIALS.md` - API access guide
- `QUICK_REFERENCE.md` - This file

---

## ⚡ Troubleshooting

| Problem | Solution |
|---------|----------|
| "Could not authenticate" | Check Bearer Token, regenerate if needed |
| "User not found" | Verify username (no @ symbol) |
| "Rate limit exceeded" | Wait 15 minutes, script handles automatically |
| "No data available" | Run comprehensive_fetcher.py first |
| "Module not found" | Run: `pip install -r requirements.txt` |
| "Port 8501 in use" | Stop existing Streamlit or use `--server.port 8502` |

---

## 📊 Rate Limits (Free Tier)

| Operation | Limit |
|-----------|-------|
| User lookup | 300 / 15 min |
| Search tweets | 450 / 15 min |
| User tweets | 1,500 / 15 min |
| Mentions | 450 / 15 min |

Script handles limits automatically with wait periods.

---

## 🎯 What Gets Analyzed

### Sentiment Analysis
- ✅ Keywords: positive, negative, neutral
- ✅ Applied to: mentions, search results, own tweets
- ✅ Output: Percentage breakdown + opinion score

### Engagement Analysis
- ✅ Likes, retweets, replies, quotes
- ✅ Average per tweet
- ✅ Top performing content
- ✅ Total engagement

### Hashtag Analysis
- ✅ Most used hashtags
- ✅ Frequency counts
- ✅ Trending tags
- ✅ In own tweets + mentions

### Audience Analysis
- ✅ Follower count and demographics
- ✅ Verified follower ratio
- ✅ Geographic distribution
- ✅ Following patterns

### Posting Patterns
- ✅ Most active hours (UTC)
- ✅ Most active days
- ✅ Posting frequency
- ✅ Optimization insights

---

## 📁 Project Structure

```
end/
├── comprehensive_fetcher.py       # Data collector (RUN FIRST)
├── enhanced_dashboard.py          # Dashboard app (RUN SECOND)
├── extract_endpoints.py           # Read endpoint docs
│
├── README.md                      # Full documentation
├── SETUP_GUIDE.md                 # Step-by-step setup
├── GET_API_CREDENTIALS.md         # API access guide
├── QUICK_REFERENCE.md             # This file
├── requirements.txt               # Python packages
│
├── X EndPoints 1.docx             # API documentation
├── X EndPoints 2.docx             # API documentation
│
└── mark_daou_comprehensive_*.json # Collected data (after run)
```

---

## 🔐 Security Tips

```bash
# Don't commit credentials
echo "*.env" >> .gitignore
echo "*_token.txt" >> .gitignore
echo "mark_daou_comprehensive_*.json" >> .gitignore

# Use environment variables (optional)
# Create .env file:
TWITTER_BEARER_TOKEN=your_token_here
```

---

## 📚 Quick Links

- **X API Docs**: https://developer.twitter.com/en/docs/twitter-api
- **Developer Portal**: https://developer.twitter.com/en/portal/dashboard
- **Streamlit Docs**: https://docs.streamlit.io/
- **Rate Limits**: https://developer.twitter.com/en/docs/twitter-api/rate-limits

---

## ✅ Success Checklist

- [ ] Python 3.7+ installed
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] X Developer account created
- [ ] Bearer Token obtained
- [ ] Mark Daou's username found
- [ ] Data collected (`comprehensive_fetcher.py`)
- [ ] Dashboard launched (`enhanced_dashboard.py`)
- [ ] All 9 pages visible
- [ ] Data updates working

---

## 🎓 Pro Tips

1. **Run daily** for trend tracking
2. **Compare** data files over time
3. **Export** insights from Raw Data page
4. **Monitor** sentiment changes
5. **Track** engagement patterns
6. **Identify** best posting times
7. **Analyze** successful hashtags

---

**Need detailed help?**
- Setup: See `SETUP_GUIDE.md`
- API: See `GET_API_CREDENTIALS.md`
- Features: See `README.md`

**Ready to go?**
```bash
python comprehensive_fetcher.py
streamlit run enhanced_dashboard.py
```

🇱🇧 **Monitor Mark Daou's political activity!**

