# Core Files Overview

## 📁 Essential Project Files

---

## 🎯 **THE 3 MAIN FILES** (Core Workflow)

### 1. 📥 **`comprehensive_fetcher.py`** - Data Collection
**Purpose**: Fetches political data from Twitter/X API

**What it does**:
- Collects user tweets (your own tweets)
- Fetches mentions (who's talking about you)
- Searches hashtags and keywords
- Gets user profile and metrics
- Saves everything to `*_comprehensive_*.json`

**When to use**:
```bash
python comprehensive_fetcher.py
```

**Output**: `mark_daou_comprehensive_YYYYMMDD_HHMMSS.json`

---

### 2. 🤖 **`advanced_insights_analyzer.py`** - AI Analysis Engine
**Purpose**: Analyzes data with advanced AI models

**What it does**:
- **Sentiment Analysis**: RoBERTa (Twitter-optimized)
- **Emotion Detection**: 6 emotions (joy, anger, fear, sadness, surprise, neutral)
- **Entity Extraction**: Twitter annotations + BERT NER
- **Topic Modeling**: LDA with 174 Arabic stop words
- **Influence Metrics**: Engagement rate, virality, reach
- **Temporal Patterns**: Best posting times, activity patterns
- **Content Strategy**: What content performs best
- **Hashtag Performance**: Which hashtags drive engagement

**When to use**:
```bash
python advanced_insights_analyzer.py <comprehensive_file.json>
```

**Input**: `mark_daou_comprehensive_*.json`  
**Output**: `mark_daou_advanced_analysis_*.json`

**Key Features**:
- 174 Arabic + dialect stop words for clean topic modeling
- Twitter API annotations for high-quality entity extraction
- Multiple AI models for comprehensive insights

---

### 3. 📊 **`enhanced_dashboard.py`** - Interactive Dashboard
**Purpose**: Visualizes all data and insights

**What it does**:
- 13 interactive pages
- Real-time data visualization
- Campaign analytics
- Geographic analysis
- Sentiment trends
- Entity networks
- Topic breakdowns
- Engagement metrics

**When to use**:
```bash
streamlit run enhanced_dashboard.py
```

**Access**: http://localhost:8501

**Dashboard Pages**:
1. 🏠 Campaign Overview
2. 😊 Sentiment Analysis
3. 🌍 Engagement & Geography
4. 📌 Topic Analysis (LDA)
5. ⏰ Posting Patterns
6. 👥 Audience Analysis
7. 📊 Public Opinion
8. 📈 Engagement Metrics
9. #️⃣ Hashtag Analysis
10. 🏷️ Entity Network
11. ⏱️ Temporal Insights
12. 🎯 Insights & Recommendations
13. #️⃣ Campaign Tracker

---

## 🛠️ **SUPPORTING FILES**

### 4. 📋 **`requirements.txt`** - Dependencies
**Purpose**: Lists all Python packages needed

**Key packages**:
- `streamlit` - Dashboard framework
- `transformers` + `torch` - AI models
- `scikit-learn` - Topic modeling (LDA)
- `tweepy` - Twitter API
- `pandas` - Data manipulation
- `plotly` - Interactive charts

**Install**:
```bash
pip install -r requirements.txt
```

---

### 5. 🔖 **`hashtag_campaign_analyzer.py`** - Campaign Tracking
**Purpose**: Track specific hashtag campaigns

**What it does**:
- Monitor campaign hashtags over time
- Track reach and engagement
- Analyze sentiment around hashtags
- Compare campaign performance

**When to use**:
```bash
python hashtag_campaign_analyzer.py
```

**Output**: `campaign_*.json`

---

### 6. 🧪 **`test_advanced_analysis.py`** - Testing Suite
**Purpose**: Verify the analysis system works

**What it tests**:
- AI model loading
- Sentiment analysis
- Emotion detection
- Entity extraction
- Data file handling
- Dashboard loading

**When to use**:
```bash
python test_advanced_analysis.py
```

---

### 7. 📊 **`analyze_data.py`** - Legacy Analysis
**Purpose**: Original basic analysis script (before advanced AI)

**Status**: ⚠️ **Legacy** - Use `advanced_insights_analyzer.py` instead

---

## 📄 **DATA FILES**

### Input Files (Raw Data):
```
mark_daou_comprehensive_20251125_112709.json
```
**Contains**:
- User profile
- Own tweets (49)
- Mentions (50)
- Search results (100)
- Raw Twitter data

**Size**: ~14,700 lines

---

### Output Files (Analyzed Data):
```
mark_daou_advanced_analysis_20251125_121245.json
```
**Contains**:
- Original raw data
- AI sentiment analysis
- Emotion detection results
- Entity network (Twitter annotations + BERT)
- Topic modeling (5 topics with 174 stop words filtered)
- Influence metrics
- Temporal patterns
- Content strategy
- Hashtag performance
- Recommendations

**Size**: Larger than input (includes all analysis)

---

## 🔄 **TYPICAL WORKFLOW**

### Step 1: Collect Data
```bash
python comprehensive_fetcher.py
```
→ Creates: `mark_daou_comprehensive_YYYYMMDD_HHMMSS.json`

### Step 2: Analyze Data
```bash
python advanced_insights_analyzer.py mark_daou_comprehensive_YYYYMMDD_HHMMSS.json
```
→ Creates: `mark_daou_advanced_analysis_YYYYMMDD_HHMMSS.json`

### Step 3: View Dashboard
```bash
streamlit run enhanced_dashboard.py
```
→ Opens: http://localhost:8501
→ Automatically loads latest analysis file

---

## 📚 **DOCUMENTATION FILES**

These explain how everything works:

- **`CORE_FILES_OVERVIEW.md`** ← You are here
- **`ARABIC_TOPIC_MODELING_FIX.md`** - Arabic stop word filtering
- **`ENTITY_AND_TOPIC_IMPROVEMENTS.md`** - Entity extraction improvements
- **`TOPIC_MODELING_EXPLANATION.md`** - How LDA works
- **`ENTITY_AND_PAGE_FIXES.md`** - Dashboard page fixes
- **`ADVANCED_ANALYSIS_GUIDE.md`** - Complete guide to advanced analysis
- **`IMPLEMENTATION_COMPLETE.md`** - Technical implementation details
- **`SUMMARY_FOR_USER.md`** - Quick start guide

---

## 🗂️ **FILE STRUCTURE**

```
end/
├── 🎯 CORE WORKFLOW
│   ├── comprehensive_fetcher.py          # 1. Collect data
│   ├── advanced_insights_analyzer.py     # 2. Analyze with AI
│   └── enhanced_dashboard.py             # 3. Visualize results
│
├── 🛠️ SUPPORTING
│   ├── requirements.txt                  # Dependencies
│   ├── hashtag_campaign_analyzer.py      # Campaign tracking
│   ├── test_advanced_analysis.py         # Testing
│   └── analyze_data.py                   # Legacy (don't use)
│
├── 📄 DATA
│   ├── *_comprehensive_*.json            # Raw Twitter data
│   ├── *_advanced_analysis_*.json        # AI-analyzed data
│   └── campaign_*.json                   # Campaign tracking data
│
├── 📚 DOCS
│   ├── CORE_FILES_OVERVIEW.md            # This file
│   ├── ARABIC_TOPIC_MODELING_FIX.md      # Topic modeling guide
│   ├── ENTITY_AND_TOPIC_IMPROVEMENTS.md  # Entity extraction
│   └── ... (other documentation)
│
└── ⚙️ CONFIG
    └── venv/                             # Python virtual environment
```

---

## 🎯 **QUICK REFERENCE**

### What file do I need for...?

| **Task** | **File** | **Command** |
|----------|----------|-------------|
| Collect new data | `comprehensive_fetcher.py` | `python comprehensive_fetcher.py` |
| Analyze existing data | `advanced_insights_analyzer.py` | `python advanced_insights_analyzer.py <file.json>` |
| View dashboard | `enhanced_dashboard.py` | `streamlit run enhanced_dashboard.py` |
| Track campaign | `hashtag_campaign_analyzer.py` | `python hashtag_campaign_analyzer.py` |
| Test system | `test_advanced_analysis.py` | `python test_advanced_analysis.py` |
| Install packages | `requirements.txt` | `pip install -r requirements.txt` |

---

## 🔑 **KEY FEATURES BY FILE**

### `comprehensive_fetcher.py`
✅ Twitter API v2 integration  
✅ User tweets, mentions, search  
✅ Profile metrics  
✅ Rate limit handling  
✅ Automatic triggering of analysis  

### `advanced_insights_analyzer.py`
✅ Twitter-RoBERTa sentiment (96% confidence)  
✅ 6-emotion detection  
✅ Twitter annotations + BERT NER  
✅ LDA topic modeling with 174 Arabic stop words  
✅ Influence scoring (engagement, virality, reach)  
✅ Temporal pattern analysis  
✅ Content strategy recommendations  
✅ Hashtag performance tracking  

### `enhanced_dashboard.py`
✅ 13 interactive pages  
✅ Real-time data loading  
✅ Plotly interactive charts  
✅ Geographic visualization  
✅ Sentiment trends over time  
✅ Entity network graphs  
✅ Topic coherence scoring  
✅ Actionable recommendations  

---

## 💾 **File Sizes**

| File | Size | Type |
|------|------|------|
| `comprehensive_fetcher.py` | ~400 lines | Python |
| `advanced_insights_analyzer.py` | ~1,300 lines | Python |
| `enhanced_dashboard.py` | ~2,550 lines | Python |
| `requirements.txt` | 27 lines | Config |
| `*_comprehensive_*.json` | ~14,700 lines | Data |
| `*_advanced_analysis_*.json` | ~20,000+ lines | Data |

---

## 🚀 **What Makes This System Powerful**

1. **Twitter API Integration**: Direct access to real-time political data
2. **Advanced AI Models**: State-of-the-art NLP (RoBERTa, BERT, LDA)
3. **Arabic Support**: 174 stop words for clean topic modeling
4. **Twitter Annotations**: High-quality entity extraction (حزب الله, الليطاني, etc.)
5. **Comprehensive Analysis**: Sentiment, emotion, entities, topics, influence
6. **Interactive Dashboard**: 13 pages of visualizations
7. **Actionable Insights**: Specific recommendations for campaign strategy

---

## 📞 **Need Help?**

**Check documentation**:
- Read `TOPIC_MODELING_EXPLANATION.md` for topic analysis
- Read `ARABIC_TOPIC_MODELING_FIX.md` for Arabic filtering
- Read `ENTITY_AND_TOPIC_IMPROVEMENTS.md` for entity extraction

**Test the system**:
```bash
python test_advanced_analysis.py
```

**View latest analysis**:
```bash
streamlit run enhanced_dashboard.py
```

---

**Generated**: November 25, 2025  
**Status**: ✅ All core files documented and functional!

