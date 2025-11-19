# How to Get X (Twitter) API v2 Credentials

## Step-by-Step Guide

### 1. Create a Twitter Developer Account

1. **Go to the Twitter Developer Portal**
   - Visit: https://developer.twitter.com/
   - Click "Sign up" or "Apply for a developer account"

2. **Sign in with your Twitter/X account**
   - Use your existing X account
   - If you don't have one, create it at https://twitter.com/

3. **Apply for Developer Access**
   - Select your use case: "Making a bot" or "Exploring the API"
   - Fill out the application form:
     - What country do you live in?
     - What's your use case? (Select "Making a bot" or "Doing something else")
     - Describe your use case: "Building a political monitoring dashboard to track Lebanese politician Mark Daou's social media presence and public sentiment for research purposes"
     - Will you make Twitter content or derived information available to government? (Select No)

4. **Review and Accept Terms**
   - Review Developer Agreement and Policy
   - Click "Submit application"

5. **Verify Your Email**
   - Check your email for verification link
   - Click the link to verify your account

### 2. Create a Project and App

1. **Access Developer Portal**
   - Go to https://developer.twitter.com/en/portal/dashboard
   - You should now see your dashboard

2. **Create a New Project**
   - Click "Create Project"
   - Project name: "Mark Daou Political Dashboard"
   - Use case: Select "Making a bot" or "Exploring the API"
   - Project description: "Monitoring Lebanese politician Mark Daou's social media activity and public opinion"

3. **Create an App**
   - App name: "MarkDaouMonitor" (must be unique)
   - Click "Complete"

### 3. Get Your API Credentials

1. **Generate Bearer Token**
   - In your app settings, go to "Keys and tokens" tab
   - Under "Authentication Tokens" section
   - Click "Generate" for **Bearer Token**
   - **IMPORTANT**: Copy and save this token immediately!
   - You won't be able to see it again

2. **Optional: Get API Key and Secret** (for OAuth)
   - API Key
   - API Key Secret
   - (Not required for basic read operations)

3. **Optional: Generate Access Token** (for posting tweets)
   - Access Token
   - Access Token Secret
   - (Not required for monitoring only)

### 4. Set Your API Access Level

1. **Check Access Level**
   - Go to your app settings
   - Check "App permissions"
   - Default is "Read" - perfect for monitoring

2. **If You Need More Access**
   - Click "Edit" on App permissions
   - Select appropriate level:
     - **Read**: Get tweets, users, followers (recommended for monitoring)
     - **Read and Write**: Also post tweets
     - **Read, Write, and Direct Messages**: Full access

### 5. API Access Tiers

#### Free Tier (Perfect for This Project)
- ✅ 500,000 tweets per month
- ✅ GET /2/tweets/search/recent (7 days)
- ✅ GET /2/users endpoints
- ✅ GET /2/tweets endpoints
- ✅ Public metrics
- ❌ Full archive search
- ❌ Context annotations (topic detection)

#### Basic Tier ($100/month)
- ✅ 10,000 tweets per month at app level
- ✅ All Free tier features
- ✅ 3 App IDs

#### Pro Tier ($5,000/month)
- ✅ 1,000,000 tweets per month
- ✅ Full archive search (all tweets since 2006)
- ✅ Context annotations
- ✅ Advanced filtering

#### Enterprise Tier (Custom pricing)
- ✅ Volume discounts
- ✅ Premium support
- ✅ Custom solutions

**For this project, the FREE tier is sufficient!**

## Your Credentials Checklist

Once you complete the setup, you should have:

- ✅ **Bearer Token** (REQUIRED) - for authentication
- ⬜ API Key (optional)
- ⬜ API Key Secret (optional)
- ⬜ Access Token (optional - for posting)
- ⬜ Access Token Secret (optional - for posting)

## Security Best Practices

### 1. Keep Your Credentials Secret
```bash
# DO NOT commit to git
echo "*.env" >> .gitignore
echo "*credentials*" >> .gitignore
```

### 2. Use Environment Variables
```bash
# Create .env file
TWITTER_BEARER_TOKEN=your_bearer_token_here
```

### 3. Never Share Your Tokens
- Don't post them online
- Don't commit them to GitHub
- Don't share them in screenshots

### 4. Regenerate If Compromised
- If you accidentally expose your token, regenerate it immediately
- Go to Developer Portal > Keys and tokens > Regenerate

## Testing Your Credentials

Once you have your Bearer Token, test it:

```bash
# Run the fetcher
python fetch_politician_data.py

# When prompted, paste your Bearer Token
# Enter: your_bearer_token_here
```

Or test directly with curl:

```bash
curl -H "Authorization: Bearer YOUR_BEARER_TOKEN" \
  "https://api.twitter.com/2/tweets/search/recent?query=Mark%20Daou"
```

If you see JSON data, your token works! ✅

## Troubleshooting

### "Could not authenticate you"
- Check that you copied the full Bearer Token
- Make sure there are no extra spaces
- Regenerate the token if needed

### "Rate limit exceeded"
- You've made too many requests
- Wait 15 minutes for the limit to reset
- Free tier: 450 requests per 15 minutes for search

### "403 Forbidden"
- Check your app permissions
- Make sure your account is verified
- Check if the endpoint requires higher access tier

### "App does not have access to this endpoint"
- Some endpoints require elevated access
- Apply for Elevated access in Developer Portal
- Or upgrade to Basic/Pro tier

## Getting Elevated Access (Free)

If you need more features:

1. Go to Developer Portal
2. Click "Products" > "Elevated"
3. Click "Apply for Elevated"
4. Fill out form explaining your use case
5. Usually approved within 1-2 days

## Rate Limits (Free Tier)

| Endpoint | Limit |
|----------|-------|
| Tweet lookup | 300 requests / 15 min |
| User lookup | 300 requests / 15 min |
| Search recent | 450 requests / 15 min |
| User tweets | 1,500 requests / 15 min |
| Mentions | 450 requests / 15 min |

## Quick Reference Links

- **Developer Portal**: https://developer.twitter.com/en/portal/dashboard
- **API Documentation**: https://developer.twitter.com/en/docs/twitter-api
- **Rate Limits**: https://developer.twitter.com/en/docs/twitter-api/rate-limits
- **API Reference**: https://developer.twitter.com/en/docs/api-reference-index
- **Support**: https://twittercommunity.com/

## Next Steps

Once you have your Bearer Token:

1. ✅ Copy your Bearer Token
2. ✅ Run: `python fetch_politician_data.py`
3. ✅ Paste your token when prompted
4. ✅ The script will fetch Mark Daou's data
5. ✅ Launch dashboard: `streamlit run dashboard.py`

---

**Ready to start?** Get your credentials at: https://developer.twitter.com/en/portal/dashboard

