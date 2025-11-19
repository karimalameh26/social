# Complete Setup Guide
## Mark Daou Political Intelligence Dashboard

Follow these steps to set up and run the dashboard.

---

## Prerequisites

- ✅ Python 3.7 or higher
- ✅ Internet connection
- ✅ X (Twitter) account
- ✅ Windows, Mac, or Linux

---

## Step 1: Get X API Credentials (15 minutes)

### 1.1 Create Developer Account

1. Go to https://developer.twitter.com/
2. Click "Sign up" or "Apply"
3. Sign in with your X account
4. Fill out the application:
   - Use case: "Making a bot" or "Exploring the API"
   - Description: "Building a political monitoring dashboard for research"
   - Country: Select your country
5. Accept terms and submit
6. Verify your email

### 1.2 Create Project and App

1. Access dashboard: https://developer.twitter.com/en/portal/dashboard
2. Click "Create Project"
   - Name: "Mark Daou Dashboard"
   - Use case: Select appropriate option
   - Description: "Political monitoring and sentiment analysis"
3. Create App
   - Name: "MarkDaouMonitor" (must be unique)
4. Complete setup

### 1.3 Generate Bearer Token

1. In your app, go to "Keys and tokens"
2. Under "Authentication Tokens"
3. Click "Generate" for **Bearer Token**
4. **COPY IT IMMEDIATELY** - you won't see it again!
5. Save it somewhere secure

Example token format:
```
AAAAAAAAAAAAAAAAAAAAABcdefghijklmnopqrstuvwxyz1234567890ABCDEFG
```

### 1.4 Verify Free Tier Access

Check your dashboard shows:
- ✅ 500,000 tweets/month
- ✅ Access to basic endpoints
- ✅ App environment: Development

**This is all you need!**

---

## Step 2: Install Python Dependencies (5 minutes)

### 2.1 Open Terminal/Command Prompt

**Windows:**
- Press `Win + R`
- Type `cmd` and press Enter

**Mac/Linux:**
- Open Terminal from Applications

### 2.2 Navigate to Project Directory

```bash
cd C:\Users\Karoo\Downloads\end
```

Or wherever you downloaded the files.

### 2.3 Install Requirements

```bash
pip install -r requirements.txt
```

This installs:
- streamlit (dashboard)
- pandas (data processing)
- plotly (charts)
- requests (API calls)
- python-docx (read docs)

**Expected output:**
```
Successfully installed streamlit-1.29.0 pandas-2.1.4 plotly-5.18.0 ...
```

---

## Step 3: Find Mark Daou's Username (2 minutes)

You need Mark Daou's X username. 

### Option A: You Know It
If you already know it (e.g., `markdaou_lb`), skip to Step 4.

### Option B: Find It

1. Go to https://twitter.com/
2. Search for "Mark Daou Lebanon"
3. Find his official account
4. His username is after the @ symbol
5. Example: If profile is `twitter.com/markdaou_lb`, username is `markdaou_lb`

**Note:** Write it down without the @ symbol.

---

## Step 4: Collect Data (10-15 minutes)

### 4.1 Run Data Collector

```bash
python comprehensive_fetcher.py
```

### 4.2 Enter Credentials

**Prompt 1:** Enter your X API Bearer Token
```
Enter your X API Bearer Token: [paste your token here]
```

**Prompt 2:** Enter Mark Daou's username
```
Enter Mark Daou's X username (e.g., markdaou_lb): [enter username]
```

### 4.3 Wait for Collection

The script will:
1. ✅ Look up user profile
2. ✅ Fetch 100 recent tweets
3. ✅ Gather 100 mentions
4. ✅ Search for related tweets
5. ✅ Sample 100 followers
6. ✅ Sample 100 following accounts
7. ✅ Analyze sentiment
8. ✅ Extract hashtags
9. ✅ Calculate engagement metrics

**Progress output:**
```
🔍 Looking up user: @markdaou_lb
✅ Found user: @markdaou_lb
   Followers: 15,420
   Following: 892

📱 Fetching tweets from @markdaou_lb...
✅ Fetched 100 tweets

💬 Fetching mentions of @markdaou_lb...
✅ Fetched 100 mentions

...
```

### 4.4 Completion

When finished:
```
✅ Data collection complete!
📁 Saved to: mark_daou_comprehensive_20251119_140530.json
```

The JSON file contains all collected data.

---

## Step 5: Launch Dashboard (1 minute)

### 5.1 Start Streamlit

```bash
streamlit run enhanced_dashboard.py
```

### 5.2 Dashboard Opens

Your browser automatically opens to:
```
http://localhost:8501
```

**If it doesn't open automatically:**
- Open browser manually
- Go to http://localhost:8501

### 5.3 Explore the Dashboard

You'll see 9 pages:

1. **📊 Campaign Overview** - Key metrics, reach, influence
2. **😊 Sentiment Analysis** - Positive/negative/neutral breakdown
3. **🗳️ Public Opinion** - Opinion score gauge
4. **📈 Engagement Metrics** - Likes, retweets, top tweets
5. **🏷️ Hashtag Analysis** - Trending hashtags
6. **💡 Topic Analysis** - Topic detection
7. **⏰ Posting Patterns** - Best times to post
8. **👥 Audience Analysis** - Follower demographics
9. **📄 Raw Data** - Complete JSON data

---

## Step 6: Regular Updates

### Update Data Daily/Weekly

To get fresh data:

1. **Stop the dashboard** (press `Ctrl+C` in terminal)

2. **Run collector again:**
   ```bash
   python comprehensive_fetcher.py
   ```

3. **Restart dashboard:**
   ```bash
   streamlit run enhanced_dashboard.py
   ```

4. **Click "🔄 Refresh Data"** in sidebar

The dashboard automatically loads the newest data file.

---

## Troubleshooting

### Issue: "Could not authenticate you"
**Solution:**
- Check Bearer Token is correct (no spaces)
- Make sure you copied the full token
- Try regenerating in Developer Portal

### Issue: "User not found"
**Solution:**
- Verify username is correct
- Don't include @ symbol
- Check account isn't suspended
- Try alternative spelling

### Issue: "Rate limit exceeded"
**Solution:**
- Wait 15 minutes
- Script should wait automatically
- Reduce max_results if needed

### Issue: "Module not found"
**Solution:**
```bash
pip install --upgrade -r requirements.txt
```

### Issue: "No data available" in dashboard
**Solution:**
- Run `comprehensive_fetcher.py` first
- Make sure JSON file was created
- Check file exists: `dir *.json` (Windows) or `ls *.json` (Mac/Linux)

### Issue: Port 8501 already in use
**Solution:**
```bash
# Kill existing Streamlit
taskkill /F /IM streamlit.exe  # Windows

# Or use different port
streamlit run enhanced_dashboard.py --server.port 8502
```

---

## Command Reference

### Collect Data
```bash
python comprehensive_fetcher.py
```

### Launch Dashboard
```bash
streamlit run enhanced_dashboard.py
```

### Stop Dashboard
```
Press Ctrl+C in terminal
```

### Check Installed Packages
```bash
pip list
```

### Update Packages
```bash
pip install --upgrade streamlit pandas plotly requests
```

### View Data Files
```bash
dir *.json           # Windows
ls *.json            # Mac/Linux
```

---

## File Locations

After setup, you'll have:

```
C:\Users\Karoo\Downloads\end\
├── comprehensive_fetcher.py
├── enhanced_dashboard.py
├── GET_API_CREDENTIALS.md
├── README.md
├── requirements.txt
└── mark_daou_comprehensive_*.json (created after data collection)
```

---

## API Rate Limits

Free tier limits (per 15 minutes):
- User lookup: 300 requests
- Search: 450 requests
- User tweets: 1,500 requests
- Mentions: 450 requests

**The fetcher stays within limits automatically.**

---

## Next Steps

Once running:

1. ✅ Explore all dashboard pages
2. ✅ Schedule regular data collection
3. ✅ Monitor sentiment trends
4. ✅ Track campaign performance
5. ✅ Export insights

---

## Getting Help

1. **API Issues**: Check GET_API_CREDENTIALS.md
2. **Python Errors**: Verify Python 3.7+ installed
3. **Dashboard Issues**: Check Streamlit documentation
4. **X API Status**: https://api.twitterstat.us/

---

## Success Checklist

- [ ] Twitter Developer account created
- [ ] Project and App created
- [ ] Bearer Token generated and saved
- [ ] Python dependencies installed
- [ ] Mark Daou's username found
- [ ] Data collected successfully
- [ ] JSON file created
- [ ] Dashboard launched
- [ ] All pages visible
- [ ] Data refreshing works

---

**✅ Setup Complete!**

You now have a fully functional political intelligence dashboard tracking Mark Daou's social media presence and public sentiment.

**Enjoy monitoring! 🇱🇧**

