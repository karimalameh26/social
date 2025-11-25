# 🤖 AI Campaign Analyst Chatbot - Setup Guide

## Overview

The AI Campaign Analyst Chatbot is now integrated into the dashboard! It uses:
- **ChromaDB Vector Database** for semantic search of your campaign data
- **OpenAI GPT-4o-mini** for generating intelligent campaign analysis and advice
- **Embeddings** from your social media listening data

## Quick Setup

### 1. Set Up OpenAI API Key

You have two options:

**Option A: Environment Variable (Recommended)**
Create a `.env` file in the project root with:
```
OPENAI_API_KEY=your_api_key_here
```

**Option B: Dashboard Sidebar**
- Go to the chatbot page in the dashboard
- Click "⚙️ OpenAI API Configuration" in the sidebar
- Enter your API key (saved for the session)

### 2. Ensure Vector Database Exists

The chatbot requires the vector database to be created first:

```bash
# 1. Collect data (if not already done)
python comprehensive_fetcher.py

# 2. Generate advanced analysis (if not already done)
python advanced_insights_analyzer.py

# 3. Create vector database
python create_vector_db.py
```

### 3. Launch Dashboard

```bash
streamlit run enhanced_dashboard.py
```

### 4. Access Chatbot

- Navigate to **[CHATBOT] AI Campaign Analyst** in the sidebar
- Start asking questions about your campaign!

## Example Questions

The chatbot can answer questions like:

- "What are the top performing tweets and why?"
- "What is the overall sentiment of mentions about the campaign?"
- "What topics are most discussed in relation to the campaign?"
- "What are the main concerns or negative feedback from the audience?"
- "Which hashtags are performing best?"
- "What are the key recommendations for improving engagement?"
- "What entities or topics should I focus on in my messaging?"
- "How is the campaign performing compared to engagement metrics?"
- "What are the best times to post based on engagement data?"
- "What are the main themes in negative sentiment mentions?"

## How It Works

1. **Semantic Search**: Your question is converted to an embedding and matched against campaign data
2. **Context Retrieval**: The most relevant documents (tweets, insights, metrics) are retrieved
3. **AI Analysis**: OpenAI GPT generates comprehensive analysis based on the retrieved context
4. **Actionable Advice**: You get strategic recommendations grounded in your actual data

## Features

- ✅ **Semantic Search**: Finds relevant information even with different wording
- ✅ **Context-Aware**: Answers are based on your actual campaign data
- ✅ **Source Citations**: See which documents were used to generate each answer
- ✅ **Chat History**: Conversation history is maintained during your session
- ✅ **Example Questions**: Quick-start buttons for common queries

## Troubleshooting

**"Vector database not found"**
- Run `python create_vector_db.py` to create the database

**"OpenAI API key not configured"**
- Set `OPENAI_API_KEY` in your `.env` file or enter it in the sidebar

**"Vector database is empty"**
- Make sure you've run the data collection and analysis scripts first
- Check that `*_advanced_analysis_*.json` files exist

**"I couldn't find relevant information"**
- Try rephrasing your question
- Ask about different aspects of the campaign
- Make sure the vector database has been populated with recent data

## API Costs

The chatbot uses OpenAI's `gpt-4o-mini` model, which is cost-effective:
- ~$0.15 per 1M input tokens
- ~$0.60 per 1M output tokens
- Typical query: ~$0.001-0.01 per question

## Security

- API keys are stored in `.env` (which is in `.gitignore`)
- Session-based API key storage (cleared when dashboard closes)
- No API keys are logged or transmitted outside OpenAI

---

**Status**: ✅ Chatbot ready to use!

