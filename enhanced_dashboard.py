"""
Enhanced Political Dashboard for Mark Daou
Displays sentiment analysis, campaign metrics, and public opinion data
"""
import streamlit as st
import json
import pandas as pd
from datetime import datetime, timezone
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import glob
import os
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Mark Daou - Political Intelligence Dashboard",
    page_icon=":flag-lb:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .big-metric {
        font-size: 2rem;
        font-weight: bold;
    }
    .sentiment-positive {
        color: #28a745;
    }
    .sentiment-negative {
        color: #dc3545;
    }
    .sentiment-neutral {
        color: #6c757d;
    }
</style>
""", unsafe_allow_html=True)

def get_raw_data(data):
    """
    Extract raw/original data from any data structure
    Handles: advanced_analysis files, analyzed files, and comprehensive files
    """
    # Check for advanced analysis format (has original_data)
    if 'original_data' in data:
        return data['original_data']
    # Check for analyzed format (has raw_data)
    elif 'raw_data' in data:
        return data['raw_data']
    # Otherwise, assume it's already raw data
    else:
        return data

def get_advanced_analysis(data):
    """Get advanced analysis section if it exists"""
    return data.get('advanced_analysis', {})

def load_latest_data():
    """Load the most recent data file (prioritize advanced analysis)"""
    # Get all types of analysis files
    advanced_files = glob.glob("mark_daou_advanced_analysis_*.json")
    analyzed_files = glob.glob("mark_daou_analyzed_*.json")
    comprehensive_files = glob.glob("mark_daou_comprehensive_*.json")
    
    # Prioritize advanced analysis, then analyzed, then comprehensive
    if advanced_files:
        latest_file = max(advanced_files, key=lambda x: Path(x).stat().st_mtime)
        print(f"[DASHBOARD] Loading advanced analysis: {latest_file}")
    elif analyzed_files:
        latest_file = max(analyzed_files, key=lambda x: Path(x).stat().st_mtime)
        print(f"[DASHBOARD] Loading analyzed data: {latest_file}")
    elif comprehensive_files:
        latest_file = max(comprehensive_files, key=lambda x: Path(x).stat().st_mtime)
        print(f"[DASHBOARD] Loading comprehensive data: {latest_file}")
    else:
        return None
    
    with open(latest_file, 'r', encoding='utf-8') as f:
        return json.load(f), latest_file

def load_latest_insights():
    """Load the most recent insights file"""
    insight_files = glob.glob("mark_daou_insights_*.json")
    if not insight_files:
        return None
    
    latest_file = max(insight_files, key=lambda x: Path(x).stat().st_mtime)
    with open(latest_file, 'r', encoding='utf-8') as f:
        return json.load(f), latest_file

def load_latest_campaign(hashtag=None):
    """Load the most recent campaign analysis"""
    if hashtag:
        campaign_files = glob.glob(f"campaign_{hashtag}_*.json")
    else:
        campaign_files = glob.glob("campaign_*.json")
    
    if not campaign_files:
        return None
    
    latest_file = max(campaign_files, key=lambda x: Path(x).stat().st_mtime)
    with open(latest_file, 'r', encoding='utf-8') as f:
        return json.load(f), latest_file

def display_campaign_overview(data):
    """Display high-level campaign metrics"""
    st.header("[CAMPAIGN] Campaign Overview")
    
    # Get raw and advanced data
    raw_data = get_raw_data(data)
    advanced = get_advanced_analysis(data)
    profile_data = raw_data.get('user_profile', {})
    
    if not profile_data or 'data' not in profile_data:
        st.warning("No profile data available")
        return
    
    profile = profile_data['data']
    pm = profile['public_metrics']
    campaign = data.get('campaign_metrics', {})
    
    # Top metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    col1.metric(
        "Total Reach",
        f"{pm.get('followers_count', 0):,}",
        help="Number of followers"
    )
    
    col2.metric(
        "Total Tweets",
        f"{pm.get('tweet_count', 0):,}",
        help="Lifetime tweets"
    )
    
    # Try to get engagement rate from advanced analysis, fallback to campaign metrics
    eng_rate = advanced.get('influence_metrics', {}).get('engagement_rate', campaign.get('engagement_rate', 0))
    col3.metric(
        "Engagement Rate",
        f"{eng_rate:.3f}%",
        help="Average engagement per tweet"
    )
    
    # Try advanced influence score first
    influence_score = advanced.get('influence_metrics', {}).get('influence_score', campaign.get('influence_score', 0))
    col4.metric(
        "Influence Score",
        f"{influence_score:,.0f}",
        help="Calculated influence metric"
    )
    
    # Calculate reach/following ratio
    ratio = pm.get('followers_count', 0) / max(pm.get('following_count', 1), 1)
    col5.metric(
        "Reach Ratio",
        f"{ratio:.1f}x",
        help="Followers to following ratio"
    )
    
    # Geographic highlights
    st.markdown("---")
    st.subheader("🌍 Geographic Reach Highlights")
    
    # Collect location data
    all_locations = []
    mentions_loc = raw_data.get('location_analysis', {}).get('mentions', {})
    search_loc = raw_data.get('location_analysis', {}).get('search_results', {})
    engagement_geo = raw_data.get('engagement_geography', {})
    
    if mentions_loc and mentions_loc.get('user_locations'):
        all_locations.extend(mentions_loc['user_locations'])
    if search_loc and search_loc.get('user_locations'):
        all_locations.extend(search_loc['user_locations'])
    
    # From engagement geography
    for tweet_id, tweet_geo in engagement_geo.items():
        retweet_geo = tweet_geo.get('retweeting_users_geography', {})
        if retweet_geo and retweet_geo.get('top_locations'):
            for loc, count in retweet_geo['top_locations'].items():
                all_locations.extend([loc] * count)
    
    if all_locations:
        # Analyze Lebanon vs Diaspora
        lebanese_keywords = [
            'beirut', 'بيروت', 'lebanon', 'لبنان', 'tripoli', 'طرابلس',
            'sidon', 'tyre', 'baalbek', 'zahle', 'jounieh', 'choueifat', 'bekaa',
            'mount lebanon', 'liban'
        ]
        
        lebanon_count = sum(1 for loc in all_locations if any(k in loc.lower() for k in lebanese_keywords))
        diaspora_count = len(all_locations) - lebanon_count
        
        col1, col2, col3 = st.columns(3)
        
        col1.metric(
            "🇱🇧 Lebanon-Based Engagement", 
            f"{lebanon_count}",
            f"{lebanon_count/(lebanon_count+diaspora_count)*100:.1f}%" if all_locations else "0%"
        )
        
        col2.metric(
            "🌍 Diaspora Engagement",
            f"{diaspora_count}",
            f"{diaspora_count/(lebanon_count+diaspora_count)*100:.1f}%" if all_locations else "0%"
        )
        
        from collections import Counter
        top_loc = Counter(all_locations).most_common(1)[0] if all_locations else ("Unknown", 0)
        col3.metric(
            "📍 Top Location",
            top_loc[0][:20],
            f"{top_loc[1]} engagements"
        )
        
        st.info("See **Engagement Geography** page for detailed geographic analysis")
    else:
        st.info("Run the comprehensive fetcher to collect geographic data from search results and retweet users.")

def display_sentiment_analysis(data):
    """Display comprehensive sentiment analysis"""
    st.header("[AI] AI-Powered Sentiment Analysis")
    
    # Get advanced analysis for AI sentiment
    advanced = get_advanced_analysis(data)
    sentiment_data = advanced.get('sentiment_summary', {})
    
    if not sentiment_data:
        # Fallback to old format
        sentiment_data = data.get('sentiment_analysis', {})
    
    if not sentiment_data:
        st.warning("No sentiment data available")
        return
    
    # Tabs for different sentiment sources
    tab1, tab2, tab3 = st.tabs(["Own Tweets (AI)", "Mentions (AI)", "Search Results"])
    
    with tab1:
        st.subheader("AI Sentiment in Own Tweets")
        if advanced:
            st.markdown("**Analyzed with**: Twitter-RoBERTa (95%+ accuracy) + TextBlob")
        else:
            st.markdown("**Analyzed with**: Keyword-based sentiment")
        
        sent = sentiment_data.get('own_tweets', {})
        if sent and sent.get('total', 0) > 0:
            display_sentiment_chart(sent, "Own Tweets")
            display_sentiment_breakdown(sent)
            
            # Show AI model info if using advanced
            if advanced:
                avg_conf = sent.get('avg_confidence', 0)
                st.success(f"**Advanced AI Analysis**: Twitter-RoBERTa model with {avg_conf:.1%} average confidence")
            else:
                st.info("**Tip**: Run advanced_insights_analyzer.py for better sentiment analysis with AI models!")
        else:
            st.info("No tweet sentiment data")
    
    with tab2:
        st.subheader("AI Sentiment in Mentions (Public Perception)")
        if advanced:
            st.markdown("**Analyzed with**: Twitter-RoBERTa (95%+ accuracy)")
        else:
            st.markdown("**Analyzed with**: Keyword-based sentiment")
        
        sent = sentiment_data.get('mentions', {})
        if sent and sent.get('total', 0) > 0:
            display_sentiment_chart(sent, "Mentions")
            display_sentiment_breakdown(sent)
            
            # Show AI confidence if available
            if advanced and 'avg_confidence' in sent:
                st.success(f"AI Confidence: {sent.get('avg_confidence', 0):.1%}")
        else:
            st.info("No mention sentiment data")
    
    with tab3:
        st.subheader("Sentiment in Search Results")
        sent = sentiment_data.get('search_results', {})
        if sent and sent.get('total', 0) > 0:
            display_sentiment_chart(sent, "Search Results")
            display_sentiment_breakdown(sent)
        else:
            st.info("No search sentiment data")

def display_sentiment_chart(sentiment, title):
    """Create pie chart for sentiment"""
    labels = ['Positive', 'Negative', 'Neutral']
    values = [
        sentiment.get('positive', 0),
        sentiment.get('negative', 0),
        sentiment.get('neutral', 0)
    ]
    colors = ['#28a745', '#dc3545', '#6c757d']
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        marker=dict(colors=colors),
        hole=0.4
    )])
    
    fig.update_layout(
        title=f"Sentiment Distribution - {title}",
        showlegend=True,
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_sentiment_breakdown(sentiment):
    """Display sentiment numbers"""
    total = sum(sentiment.values())
    if total == 0:
        return
    
    col1, col2, col3 = st.columns(3)
    
    pos = sentiment.get('positive', 0)
    col1.markdown(f"""
    <div style='text-align: center; padding: 20px; background-color: #d4edda; border-radius: 10px;'>
        <h2 class='sentiment-positive'>{pos}</h2>
        <p>Positive ({pos/total*100:.1f}%)</p>
    </div>
    """, unsafe_allow_html=True)
    
    neg = sentiment.get('negative', 0)
    col2.markdown(f"""
    <div style='text-align: center; padding: 20px; background-color: #f8d7da; border-radius: 10px;'>
        <h2 class='sentiment-negative'>{neg}</h2>
        <p>Negative ({neg/total*100:.1f}%)</p>
    </div>
    """, unsafe_allow_html=True)
    
    neu = sentiment.get('neutral', 0)
    col3.markdown(f"""
    <div style='text-align: center; padding: 20px; background-color: #e2e3e5; border-radius: 10px;'>
        <h2 class='sentiment-neutral'>{neu}</h2>
        <p>Neutral ({neu/total*100:.1f}%)</p>
    </div>
    """, unsafe_allow_html=True)

def display_engagement_metrics(data):
    """Display detailed engagement analysis"""
    st.header("[STATS] Engagement Analytics")
    
    raw_data = get_raw_data(data)
    advanced = get_advanced_analysis(data)
    
    # Get analyzed tweets
    analyzed_tweets = advanced.get('analyzed_tweets', [])
    if not analyzed_tweets:
        analyzed_tweets = raw_data.get('user_tweets', {}).get('data', [])
    
    # Get engagement metrics
    eng_data = raw_data.get('engagement_metrics', {})
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Own Tweets Performance")
        eng = eng_data.get('user_tweets', {})
        
        if eng:
            metrics_df = pd.DataFrame({
                'Metric': ['Likes', 'Retweets', 'Replies', 'Quotes'],
                'Total': [
                    eng.get('total_likes', 0),
                    eng.get('total_retweets', 0),
                    eng.get('total_replies', 0),
                    eng.get('total_quotes', 0)
                ],
                'Average': [
                    eng.get('avg_likes', 0),
                    eng.get('avg_retweets', 0),
                    eng.get('avg_replies', 0),
                    0
                ]
            })
            
            fig = go.Figure(data=[
                go.Bar(name='Total', x=metrics_df['Metric'], y=metrics_df['Total']),
                go.Bar(name='Average', x=metrics_df['Metric'], y=metrics_df['Average'])
            ])
            fig.update_layout(barmode='group', title="Engagement Metrics")
            st.plotly_chart(fig, use_container_width=True)
            
            # Top performing tweet
            if eng.get('top_tweet'):
                st.info("**Top Performing Tweet**")
                top = eng['top_tweet']
                st.write(f"**Engagement:** {top['engagement']:,}")
                st.write(f"**Text:** {top['text'][:200]}...")
                
                tm = top['metrics']
                tcol1, tcol2, tcol3, tcol4 = st.columns(4)
                tcol1.metric("Likes", f"{tm.get('like_count', 0):,}")
                tcol2.metric("Retweets", f"{tm.get('retweet_count', 0):,}")
                tcol3.metric("Replies", f"{tm.get('reply_count', 0):,}")
                tcol4.metric("Quotes", f"{tm.get('quote_count', 0):,}")
    
    with col2:
        st.subheader("Mention Engagement")
        eng = eng_data.get('mentions', {})
        
        if eng:
            st.metric("Total Mentions Analyzed", eng.get('total_tweets', 0))
            st.metric("Total Engagement on Mentions", 
                     f"{eng.get('total_likes', 0) + eng.get('total_retweets', 0):,}")
            st.metric("Average Engagement per Mention",
                     f"{eng.get('avg_likes', 0) + eng.get('avg_retweets', 0):.1f}")

def display_hashtag_analysis(data):
    """Display hashtag analysis with AI sentiment"""
    st.header("[HASHTAGS] Hashtag Sentiment Analysis")
    
    raw_data = get_raw_data(data)
    advanced = get_advanced_analysis(data)
    
    # Try advanced hashtag performance first, then fallback
    hashtag_perf = advanced.get('hashtag_performance', {})
    if hashtag_perf and 'hashtag_performance' in hashtag_perf:
        hashtag_data = hashtag_perf['hashtag_performance']
        st.success("**Advanced Analysis**: Hashtag performance with engagement and emotion tracking")
    else:
        # Fallback to raw hashtag analysis
        hashtag_data = raw_data.get('hashtag_analysis', {})
        if hashtag_data:
            st.info("📊 Basic hashtag analysis")
    
    if not hashtag_data:
        st.warning("No hashtag data available.")
        st.info("**Tip**: Use hashtags in your tweets to track campaign performance!")
        return
    
    # Convert to DataFrame with sentiment
    hashtag_list = []
    for tag, data_dict in hashtag_data.items():
        # Handle both old and new formats
        if isinstance(data_dict, dict):
            # New format from advanced analysis
            sentiments = data_dict.get('sentiments', {})
            hashtag_list.append({
                'Hashtag': f"#{tag}",
                'Count': data_dict.get('count', 0),
                'Positive': sentiments.get('positive', data_dict.get('positive', 0)),
                'Negative': sentiments.get('negative', data_dict.get('negative', 0)),
                'Neutral': sentiments.get('neutral', data_dict.get('neutral', 0)),
                'Avg_Engagement': data_dict.get('avg_engagement', 0)
            })
        else:
            # Old simple count format
            hashtag_list.append({
                'Hashtag': f"#{tag}",
                'Count': data_dict if isinstance(data_dict, int) else 0,
                'Positive': 0,
                'Negative': 0,
                'Neutral': 0,
                'Avg_Engagement': 0
            })
    
    df = pd.DataFrame(hashtag_list)
    df = df.sort_values('Count', ascending=False).head(20)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Top Hashtags by Usage")
        fig = px.bar(df, x='Count', y='Hashtag', orientation='h',
                    title="Most Used Hashtags",
                    color='Avg_Engagement',
                    color_continuous_scale='Blues')
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Hashtag Sentiment Distribution")
        
        # Stacked bar chart
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Positive', x=df['Hashtag'], y=df['Positive'], marker_color='green'))
        fig.add_trace(go.Bar(name='Neutral', x=df['Hashtag'], y=df['Neutral'], marker_color='gray'))
        fig.add_trace(go.Bar(name='Negative', x=df['Hashtag'], y=df['Negative'], marker_color='red'))
        
        fig.update_layout(barmode='stack', title="Sentiment per Hashtag", xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    # Hashtag details table
    st.subheader("Detailed Hashtag Analysis")
    st.dataframe(df, use_container_width=True)

def display_topic_analysis(data):
    """Display AI topic modeling results"""
    st.header("[TOPICS] AI Topic Modeling (LDA)")
    
    # Get raw and advanced data
    raw_data = get_raw_data(data)
    advanced = get_advanced_analysis(data)
    
    # Try advanced topics first, then fallback
    topic_data = advanced.get('topics', {})
    if not topic_data:
        topic_data = raw_data.get('topic_analysis', {})
    
    if not topic_data:
        st.warning("No topic data available.")
        st.info("**Tip**: Run `python advanced_insights_analyzer.py` to extract topics using AI!")
        return
    
    if advanced:
        st.success("**Advanced AI Topics**: Latent Dirichlet Allocation (LDA) with TF-IDF + Coherence Scoring")
    else:
        st.markdown("**Method**: Latent Dirichlet Allocation (LDA)")
    
    # Comprehensive explanation of topic modeling
    with st.expander("📚 What is Topic Modeling? How Does It Work?", expanded=False):
        st.markdown("""
        ## 🧠 What is Topic Modeling?
        
        **Topic modeling** is an AI technique that automatically discovers hidden themes (topics) in a collection of texts.
        Instead of reading thousands of tweets manually, the AI analyzes them all and groups similar content together.
        
        **Real-World Analogy:**
        - Imagine you have 1,000 news articles
        - A human would take days to categorize them
        - Topic modeling does it in seconds, finding patterns like "Sports", "Politics", "Technology"
        
        ---
        
        ## ⚙️ How Does LDA Work?
        
        **LDA (Latent Dirichlet Allocation)** is the most popular topic modeling algorithm. Here's how it works:
        
        ### Step 1: Text Preprocessing 🧹
        - Remove URLs, @mentions, hashtag symbols
        - Filter out short words (< 3 characters)
        - Remove stop words (common words like "the", "is", "a")
        - Clean Arabic and English text
        
        ### Step 2: TF-IDF Vectorization 📊
        - **TF (Term Frequency)**: How often a word appears in a tweet
        - **IDF (Inverse Document Frequency)**: How unique/important a word is
        - **Result**: Words that are frequent BUT unique get high scores
        - Example: "Lebanon" appears often BUT is important → high score
        - Example: "the" appears often BUT is common → low score
        
        ### Step 3: LDA Algorithm 🤖
        The algorithm assumes:
        - Each tweet is a **mixture of topics**
        - Each topic is a **mixture of words**
        
        **How it learns:**
        1. Randomly assigns words to topics
        2. Iterates many times, adjusting assignments
        3. Learns which words tend to appear together
        4. Groups similar content into topics
        
        **Example:**
        - Topic 1: "lebanon", "government", "parliament", "elections"
        - Topic 2: "security", "conflict", "ceasefire", "border"
        - Topic 3: "economy", "crisis", "inflation", "reforms"
        
        ### Step 4: Coherence Scoring ✅
        - Measures how "meaningful" each topic is
        - Higher coherence = keywords make sense together
        - Lower coherence = random words grouped together
        - **Good topic**: "elections", "vote", "parliament", "democracy" (coherent!)
        - **Bad topic**: "the", "is", "random", "xyzabc" (incoherent!)
        
        ---
        
        ## 📈 What Do the Results Mean?
        
        **Keywords**: The most important words for each topic
        - These define what the topic is about
        - Example: If keywords are "lebanon", "government", "parliament" → Political governance topic
        
        **Coherence Score**: Quality indicator (0.0 - 1.0)
        - **0.3-0.4**: Decent topic
        - **0.4-0.5**: Good topic
        - **0.5+**: Excellent topic
        - **< 0.3**: May need more data or better filtering
        
        **All Keywords**: Extended list showing related terms
        - Helps understand the full scope of the topic
        
        ---
        
        ## 🎯 Why Is This Useful?
        
        For political campaigns, topic modeling reveals:
        1. **What people are talking about** in your tweets and mentions
        2. **Key themes** in political discourse
        3. **Which topics dominate** the conversation
        4. **Emerging issues** that might need attention
        
        **Example Use Cases:**
        - Identify that "economic crisis" is discussed more than "security"
        - Discover new topics emerging in mentions (public concerns)
        - Track which topics get most engagement
        - Adjust campaign messaging based on dominant topics
        
        ---
        
        ## 🔧 Our Implementation
        
        **Parameters Used:**
        - **Max Features**: 150 words (focus on most important terms)
        - **N-grams**: 1-2 (single words + two-word phrases)
        - **Min Document Frequency**: 2 (word must appear in ≥2 tweets)
        - **Max Document Frequency**: 80% (ignore words in >80% of tweets)
        - **Token Length**: Minimum 3 characters
        
        **Filtering:**
        - ✅ Removes URLs, @mentions, RT markers
        - ✅ Filters Arabic and English stop words
        - ✅ Removes pure numbers and punctuation
        - ✅ Cleans hashtag symbols (keeps the text)
        
        **Result**: Clean, meaningful topics that reveal conversation themes!
        """)
    
    st.markdown(f"**Topics Identified**: {len(topic_data)}")
    st.markdown("---")
    
    # Display topics as expandable sections
    for idx, (topic_name, topic_info) in enumerate(topic_data.items(), 1):
        # Get coherence score if available (advanced analysis)
        coherence = topic_info.get('coherence', 0)
        title = f"📌 Topic {idx}: {topic_name}" + (f" (Coherence: {coherence:.3f})" if coherence > 0 else "")
        
        with st.expander(title, expanded=(idx <= 3)):  # Expand first 3
            keywords = topic_info.get('keywords', [])
            all_keywords = topic_info.get('all_keywords', keywords)
            
            # Top keywords with emoji
            st.markdown("**🔑 Key Terms:**")
            st.markdown(" • ".join(f"**{kw}**" for kw in keywords[:5]))
            
            # All keywords if advanced
            if len(all_keywords) > len(keywords):
                st.markdown("")
                st.markdown("**📋 Related Terms:**")
                st.markdown(", ".join(all_keywords[5:10]))
            
            # Coherence score explanation
            if coherence > 0:
                st.caption(f"Coherence Score: {coherence:.3f} - Higher scores indicate more meaningful/interpretable topics")

def display_posting_patterns(data):
    """Display posting pattern analysis"""
    st.header("[TIME] Posting Patterns")
    
    raw_data = get_raw_data(data)
    advanced = get_advanced_analysis(data)
    
    # Try advanced temporal patterns first
    temporal = advanced.get('temporal_patterns', {})
    patterns = raw_data.get('posting_patterns', {})
    
    # Use temporal if available (has hourly_activity), otherwise use patterns
    if temporal and 'hourly_activity' in temporal:
        patterns = {
            'most_active_hours': temporal.get('hourly_activity', {}),
            'most_active_days': temporal.get('daily_activity', {})
        }
        st.success("**Advanced Temporal Analysis**: Activity patterns analyzed by AI")
    elif not patterns:
        st.warning("No posting pattern data available.")
        st.info("**Tip**: Run `python advanced_insights_analyzer.py` to get temporal insights!")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Most Active Hours (UTC)")
        hours = patterns.get('most_active_hours', {})
        if hours:
            df = pd.DataFrame(list(hours.items()), columns=['Hour', 'Tweets'])
            df['Hour'] = df['Hour'].astype(int)
            df = df.sort_values('Hour')
            
            fig = px.bar(df, x='Hour', y='Tweets',
                        title="Tweet Activity by Hour")
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Most Active Days")
        days = patterns.get('most_active_days', {})
        if days:
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            df = pd.DataFrame(list(days.items()), columns=['Day', 'Tweets'])
            df['Day'] = pd.Categorical(df['Day'], categories=day_order, ordered=True)
            df = df.sort_values('Day')
            
            fig = px.bar(df, x='Day', y='Tweets',
                        title="Tweet Activity by Day",
                        color='Tweets',
                        color_continuous_scale='Blues')
            st.plotly_chart(fig, use_container_width=True)

def display_audience_analysis(data):
    """Display audience engagement patterns from AI analysis"""
    st.header("[AUDIENCE] Audience Engagement Analysis")
    
    raw_data = get_raw_data(data)
    advanced = get_advanced_analysis(data)
    
    # Try to get analyzed tweets from advanced analysis
    analyzed_tweets = advanced.get('analyzed_tweets', [])
    analyzed_mentions = advanced.get('analyzed_mentions', [])
    
    # Fallback to raw data
    if not analyzed_tweets:
        analyzed_tweets = raw_data.get('user_tweets', {}).get('data', [])
    if not analyzed_mentions:
        analyzed_mentions = raw_data.get('mentions', {}).get('data', [])
    
    # Get audience engagement data (old format) or compute from tweets
    audience_data = raw_data.get('analysis', {}).get('audience_engagement', {})
    
    if not analyzed_tweets and not analyzed_mentions:
        st.warning("No tweet data available.")
        return
    
    # Display engagement metrics
    st.subheader("📊 Engagement Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Own Tweets", len(analyzed_tweets))
        if analyzed_tweets:
            avg_eng = sum(t.get('public_metrics', {}).get('like_count', 0) + 
                         t.get('public_metrics', {}).get('retweet_count', 0) 
                         for t in analyzed_tweets) / len(analyzed_tweets)
            st.metric("Avg Engagement", f"{avg_eng:.1f}")
    
    with col2:
        st.metric("Mentions", len(analyzed_mentions))
        if analyzed_mentions:
            total_reach = sum(t.get('public_metrics', {}).get('like_count', 0) for t in analyzed_mentions)
            st.metric("Total Reach (Likes)", f"{total_reach:,}")
    
    with col3:
        # Count tweets with high engagement
        high_eng_count = sum(1 for t in analyzed_tweets 
                            if (t.get('public_metrics', {}).get('like_count', 0) + 
                               t.get('public_metrics', {}).get('retweet_count', 0)) > 100)
        st.metric("High Engagement Tweets", high_eng_count, help="Tweets with 100+ likes+retweets")
    
    st.markdown("---")
    
    # Top performing tweets
    st.subheader("🔥 Top Performing Content")
    
    if analyzed_tweets:
        # Sort by engagement
        tweets_with_eng = []
        for tweet in analyzed_tweets:
            pm = tweet.get('public_metrics', {})
            engagement = (pm.get('like_count', 0) + 
                         pm.get('retweet_count', 0) * 2 + 
                         pm.get('reply_count', 0) * 3)
            tweets_with_eng.append({
                'text': tweet.get('text', ''),
                'likes': pm.get('like_count', 0),
                'retweets': pm.get('retweet_count', 0),
                'replies': pm.get('reply_count', 0),
                'engagement': engagement,
                'sentiment': tweet.get('ai_sentiment', {}).get('sentiment', 'N/A') if advanced else 'N/A'
            })
        
        top_tweets = sorted(tweets_with_eng, key=lambda x: x['engagement'], reverse=True)[:5]
        
        for i, tweet in enumerate(top_tweets, 1):
            with st.expander(f"#{i} - {tweet['engagement']:,} total engagement", expanded=(i==1)):
                st.write(f"**Text:** {tweet['text'][:200]}...")
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("❤️ Likes", f"{tweet['likes']:,}")
                col2.metric("🔁 Retweets", f"{tweet['retweets']:,}")
                col3.metric("💬 Replies", f"{tweet['replies']:,}")
                if advanced and tweet['sentiment'] != 'N/A':
                    col4.metric("😊 Sentiment", tweet['sentiment'].title())
    
    st.markdown("---")
    
    # Legacy audience data display
    if audience_data:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("[DISCUSS] Most Discussed")
            discussed = audience_data.get('most_discussed', [])
            st.metric("High Reply Tweets", len(discussed))
            for tweet in discussed[:5]:
                st.write(f"[REPLIES] {tweet['replies']} replies")
                st.caption(tweet['text'][:80] + "...")
                st.markdown("---")
        
        with col2:
            st.subheader("[HOT] Controversial")
            controversial = audience_data.get('most_controversial', [])
            st.metric("High Debate Tweets", len(controversial))
            for tweet in controversial[:5]:
                ratio = tweet.get('controversy_ratio', 0)
                st.write(f"[SCORE] Controversy: {ratio:.2f}")
                st.caption(tweet['text'][:80] + "...")
                st.markdown("---")
        
        with col3:
            st.subheader("[VIRAL] Highest Engagement")
            high_eng = audience_data.get('highest_engagement', [])
            st.metric("Top Performers", len(high_eng))
            for tweet in high_eng[:5]:
                st.write(f"[TOTAL] {tweet['total_engagement']:,} total")
                st.caption(tweet['text'][:80] + "...")
                st.markdown("---")
    
    # Follower analysis
    raw_data = get_raw_data(data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Follower Sample Analysis")
        followers = raw_data.get('followers', {})
        
        if followers and 'data' in followers:
            follower_list = followers['data']
            
            # Count verified followers
            verified_count = sum(1 for f in follower_list if f.get('verified', False))
            total = len(follower_list)
            
            st.metric("Sample Size", total)
            st.metric("Verified Followers", f"{verified_count} ({verified_count/total*100:.1f}%)")
            
            # Follower locations
            locations = {}
            for f in follower_list:
                loc = f.get('location', 'Unknown')
                if loc:
                    locations[loc] = locations.get(loc, 0) + 1
            
            if locations:
                top_locs = sorted(locations.items(), key=lambda x: x[1], reverse=True)[:10]
                st.write("**Top Follower Locations:**")
                for loc, count in top_locs:
                    st.write(f"- {loc}: {count}")
        else:
            st.info("ℹ️ Follower data not available (endpoint requires Pro/Enterprise tier)")
            st.markdown("**Available**: See **🌍 Engagement Geography** page for location data from active engagers")
    
    with col2:
        st.subheader("Following Analysis")
        following = raw_data.get('following', {})
        
        if following and 'data' in following:
            following_list = following['data']
            
            verified_count = sum(1 for f in following_list if f.get('verified', False))
            total = len(following_list)
            
            st.metric("Sample Size", total)
            st.metric("Following Verified Accounts", f"{verified_count} ({verified_count/total*100:.1f}%)")
            
            # Average follower count of accounts being followed
            avg_followers = sum(f.get('public_metrics', {}).get('followers_count', 0) 
                               for f in following_list) / total if total > 0 else 0
            
            st.metric("Avg Followers of Followed Accounts", f"{avg_followers:,.0f}")
        else:
            st.info("ℹ️ Following data not available (endpoint requires Pro/Enterprise tier)")

def display_engagement_geography(data):
    """Display geographic analysis from search results and retweet users"""
    st.header("[GEOGRAPHY] Engagement Geographic Analysis")
    
    st.markdown("""
    This page shows **WHERE people who engage with Mark Daou's content are located**.
    
    Data sources:
    - 📍 **Search Results**: Users discussing Mark Daou
    - 🔁 **Retweet Users**: People actively spreading his content
    - 💬 **Mentions**: People mentioning Mark Daou
    
    Locations are extracted from user profile `location` field (self-reported).
    """)
    
    raw_data = get_raw_data(data)
    
    # Debug info (collapsible)
    with st.expander("🔍 Data Availability Check", expanded=False):
        has_location_analysis = 'location_analysis' in raw_data
        has_engagement_geo = 'engagement_geography' in raw_data
        
        st.write(f"**Location Analysis Present**: {'✅ Yes' if has_location_analysis else '❌ No'}")
        st.write(f"**Engagement Geography Present**: {'✅ Yes' if has_engagement_geo else '❌ No'}")
        
        if has_location_analysis:
            loc_analysis = raw_data.get('location_analysis', {})
            st.write(f"- Mentions locations: {len(loc_analysis.get('mentions', {}).get('user_locations', []))}")
            st.write(f"- Search locations: {len(loc_analysis.get('search_results', {}).get('user_locations', []))}")
        
        if has_engagement_geo:
            eng_geo = raw_data.get('engagement_geography', {})
            st.write(f"- Engagement geography tweets: {len(eng_geo)}")
    
    # === TAB 1: Overall Geographic Distribution ===
    tab1, tab2, tab3 = st.tabs(["📊 Overall Analysis", "🔥 Top Tweets Engagement", "🔍 Data Sources"])
    
    with tab1:
        st.subheader("📍 Combined Geographic Distribution")
        
        # Collect all locations from different sources
        all_locations = []
        source_counts = {'mentions': 0, 'search': 0, 'retweets': 0}
        
        # From mentions
        mentions_loc = raw_data.get('location_analysis', {}).get('mentions', {})
        if mentions_loc and 'user_locations' in mentions_loc:
            user_locs = mentions_loc.get('user_locations', [])
            if user_locs:  # Check if list is not empty
                all_locations.extend(user_locs)
                source_counts['mentions'] = len(user_locs)
        
        # From search results
        search_loc = raw_data.get('location_analysis', {}).get('search_results', {})
        if search_loc and 'user_locations' in search_loc:
            user_locs = search_loc.get('user_locations', [])
            if user_locs:  # Check if list is not empty
                all_locations.extend(user_locs)
                source_counts['search'] = len(user_locs)
        
        # From engagement geography (retweets)
        engagement_geo = raw_data.get('engagement_geography', {})
        if engagement_geo:
            for tweet_id, tweet_geo in engagement_geo.items():
                retweet_geo = tweet_geo.get('retweeting_users_geography', {})
                if retweet_geo and retweet_geo.get('top_locations'):
                    # Add based on counts
                    for loc, count in retweet_geo['top_locations'].items():
                        all_locations.extend([loc] * count)
                        source_counts['retweets'] += count
        
        if not all_locations:
            st.warning("""
            No geographic data found in the loaded file.
            
            Please ensure you're using the latest comprehensive JSON file with geographic data.
            Run `python comprehensive_fetcher.py` to collect fresh data with the latest updates.
            """)
            return
        
        # Overview metrics
        st.markdown("### 📈 Data Collection Summary")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("💬 Mentions", f"{source_counts['mentions']}")
        col2.metric("🔍 Search Results", f"{source_counts['search']}")
        col3.metric("🔁 Retweet Users", f"{source_counts['retweets']}")
        col4.metric("📊 Total Locations", f"{len(all_locations)}")
        
        # Analyze Lebanon vs Diaspora
        lebanese_keywords = [
            'beirut', 'بيروت', 'lebanon', 'لبنان', 'tripoli', 'طرابلس',
            'sidon', 'صيدا', 'tyre', 'صور', 'baalbek', 'بعلبك',
            'zahle', 'زحلة', 'jounieh', 'جونية', 'byblos', 'جبيل',
            'choueifat', 'الشويفات', 'bekaa', 'البقاع', 'mount lebanon',
            'north lebanon', 'south lebanon', 'nabatieh', 'النبطية',
            'saida', 'tyr', 'jbeil', 'liban'
        ]
        
        lebanon_count = 0
        diaspora_count = 0
        
        for loc in all_locations:
            if any(keyword in loc.lower() for keyword in lebanese_keywords):
                lebanon_count += 1
            else:
                diaspora_count += 1
        
        # Lebanon vs Diaspora chart
        st.markdown("### 🇱🇧 Lebanon vs Diaspora Distribution")
        
        if lebanon_count + diaspora_count > 0:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig = go.Figure(data=[go.Pie(
                    labels=['Lebanon-Based', 'Diaspora'],
                    values=[lebanon_count, diaspora_count],
                    marker=dict(colors=['#00A651', '#EE161F']),  # Lebanese flag colors
                    hole=0.4,
                    textinfo='label+percent+value'
                )])
                
                fig.update_layout(
                    title="Geographic Distribution of Engagement",
                    showlegend=True,
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.metric("🇱🇧 Lebanon-Based", f"{lebanon_count}", 
                         f"{lebanon_count/(lebanon_count+diaspora_count)*100:.1f}%")
                st.metric("🌍 Diaspora", f"{diaspora_count}",
                         f"{diaspora_count/(lebanon_count+diaspora_count)*100:.1f}%")
                
                st.info(f"""
                **Insight:**  
                {'Lebanon-focused' if lebanon_count > diaspora_count else 'Diaspora-driven'} engagement
                """)
        
        # Top locations
        from collections import Counter
        location_counts = Counter(all_locations)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📍 Top 15 Locations")
            top_locs = location_counts.most_common(15)
            if top_locs:
                df = pd.DataFrame(top_locs, columns=['Location', 'Count'])
                fig = px.bar(df, x='Count', y='Location', orientation='h',
                            title="Most Common User Locations",
                            color='Count',
                            color_continuous_scale='Greens')
                fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=500)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 🌍 Top 15 Locations (Table)")
            if top_locs:
                df = pd.DataFrame(top_locs, columns=['Location', 'Engagements'])
                df['Percentage'] = (df['Engagements'] / df['Engagements'].sum() * 100).round(1)
                df['Percentage'] = df['Percentage'].astype(str) + '%'
                st.dataframe(df, use_container_width=True, height=500)
    
    with tab2:
        st.subheader("🔥 Top Performing Tweets - Geographic Breakdown")
        
        engagement_geo = raw_data.get('engagement_geography', {})
        
        if not engagement_geo:
            st.info("No engagement geography data available.")
            return
        
        for idx, (tweet_id, tweet_geo) in enumerate(engagement_geo.items(), 1):
            tweet_info = tweet_geo.get('tweet_info', {})
            retweet_geo = tweet_geo.get('retweeting_users_geography', {})
            
            st.markdown(f"---")
            st.markdown(f"### 📌 Tweet #{idx}")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"**Text:** {tweet_info.get('text', 'N/A')}")
                st.markdown(f"**Total Engagement:** {tweet_info.get('engagement', 0):,}")
            
            with col2:
                if retweet_geo.get('total_users', 0) > 0:
                    st.metric("🔁 Retweet Users", f"{retweet_geo['total_users']}")
                    st.metric("📍 With Location", f"{retweet_geo['with_location']} ({retweet_geo.get('location_coverage', 0):.0f}%)")
            
            # Geographic breakdown for this tweet
            if retweet_geo.get('total_users', 0) > 0:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Lebanon vs Diaspora for this tweet
                    if retweet_geo['lebanon_based'] + retweet_geo['diaspora'] > 0:
                        fig = go.Figure(data=[go.Pie(
                            labels=['Lebanon', 'Diaspora'],
                            values=[retweet_geo['lebanon_based'], retweet_geo['diaspora']],
                            marker=dict(colors=['#00A651', '#EE161F']),
                            hole=0.3
                        )])
                        fig.update_layout(title="Retweet Distribution", height=300, showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.markdown(f"🇱🇧 **{retweet_geo.get('lebanon_percentage', 0):.1f}%** Lebanon")
                        st.markdown(f"🌍 **{retweet_geo.get('diaspora_percentage', 0):.1f}%** Diaspora")
                
                with col2:
                    # Top locations for this tweet
                    st.markdown("**Top Locations:**")
                    top_locs = list(retweet_geo.get('top_locations', {}).items())[:5]
                    for loc, count in top_locs:
                        st.markdown(f"- {loc}: **{count}**")
                
                with col3:
                    # Top countries for this tweet
                    st.markdown("**Top Countries:**")
                    top_countries = list(retweet_geo.get('top_countries', {}).items())[:5]
                    for country, count in top_countries:
                        st.markdown(f"- {country}: **{count}**")
    
    with tab3:
        st.subheader("📊 Data Sources Breakdown")
        
        st.markdown("""
        ### How Geographic Data is Collected
        
        The system extracts location from **3 sources**:
        
        1. **User Profile Location** (Primary - ~60-80% coverage)
           - Self-reported text field: `user.location`
           - Examples: "Beirut, Lebanon", "Paris, France", "بيروت"
           - What users enter in their Twitter profile
        
        2. **Tweet Geo-tagging** (Rare - <1% of tweets)
           - GPS coordinates if user enables location
           - Most users have this disabled for privacy
        
        3. **Place Objects** (When geo is enabled)
           - Structured location data
           - Very rare due to low geo-tagging adoption
        
        ### Classification Method
        
        **Lebanon vs Diaspora** is determined by keyword matching:
        - **Lebanon keywords**: beirut, بيروت, lebanon, لبنان, tripoli, sidon, etc.
        - If location contains Lebanese keyword → Lebanon-based
        - Otherwise → Diaspora
        
        ### Data Sources in This Analysis
        """)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 💬 Mentions")
            st.markdown(f"Users: **{source_counts['mentions']}**")
            st.markdown("Source: Users mentioning @DaouMark")
            
        with col2:
            st.markdown("#### 🔍 Search Results")
            st.markdown(f"Users: **{source_counts['search']}**")
            st.markdown('Source: Tweets about "Mark Daou"')
        
        with col3:
            st.markdown("#### 🔁 Retweet Users")
            st.markdown(f"Users: **{source_counts['retweets']}**")
            st.markdown("Source: Users who retweeted top 3 tweets")
        
        st.success(f"""
        **Total Geographic Data Points:** {len(all_locations)}
        
        This provides a comprehensive view of WHERE Mark Daou's content resonates and is being amplified.
        """)

def display_follower_geography(data):
    """Display advanced geographic analysis of followers"""
    st.header("[GEOGRAPHY] Follower Geographic Distribution")
    
    raw_data = get_raw_data(data)
    geo_data = raw_data.get('follower_geography', {})
    
    if not geo_data or geo_data.get('total_followers', 0) == 0:
        st.warning("""
        ⚠️ **Follower geographic data not available.**
        
        The `GET /2/users/:id/followers` endpoint is not available in the Basic tier.
        
        **See "🌍 Engagement Geography" page instead** - it shows geographic data from:
        - Search results
        - Retweet users  
        - Mentions
        """)
        return
    
    # Overview metrics
    st.subheader("📍 Geographic Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("Total Followers Analyzed", f"{geo_data.get('total_followers', 0):,}")
    col2.metric("With Location Data", f"{geo_data.get('with_location', 0):,}")
    col3.metric("Lebanon-Based", f"{geo_data.get('lebanon_based', 0):,}")
    col4.metric("Diaspora", f"{geo_data.get('diaspora', 0):,}")
    
    # Lebanon vs Diaspora pie chart
    st.subheader("🇱🇧 Lebanon vs Diaspora Distribution")
    
    lebanon_count = geo_data.get('lebanon_based', 0)
    diaspora_count = geo_data.get('diaspora', 0)
    
    if lebanon_count + diaspora_count > 0:
        fig = go.Figure(data=[go.Pie(
            labels=['Lebanon-Based', 'Diaspora'],
            values=[lebanon_count, diaspora_count],
            marker=dict(colors=['#00A651', '#EE161F']),  # Lebanese flag colors
            hole=0.4
        )])
        
        fig.update_layout(
            title="Follower Distribution: Lebanon vs Diaspora",
            showlegend=True,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        col1.metric("Lebanon %", f"{geo_data.get('lebanon_percentage', 0):.1f}%")
        col2.metric("Diaspora %", f"{geo_data.get('diaspora_percentage', 0):.1f}%")
    
    # Top locations bar chart
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Top Follower Locations")
        top_locs = geo_data.get('top_locations', {})
        if top_locs:
            df = pd.DataFrame(list(top_locs.items())[:15], columns=['Location', 'Followers'])
            fig = px.bar(df, x='Followers', y='Location', orientation='h',
                        title="Top 15 Follower Locations",
                        color='Followers',
                        color_continuous_scale='Greens')
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No location data available")
    
    with col2:
        st.subheader("🌍 Top Countries")
        top_countries = geo_data.get('top_countries', {})
        if top_countries:
            df = pd.DataFrame(list(top_countries.items())[:15], columns=['Country', 'Followers'])
            fig = px.bar(df, x='Followers', y='Country', orientation='h',
                        title="Top 15 Countries",
                        color='Followers',
                        color_continuous_scale='Blues')
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No country data available")
    
    # Location diversity metric
    st.metric("Location Diversity", 
             f"{geo_data.get('location_diversity', 0)} unique locations",
             help="Number of unique geographic locations among followers")

def display_follower_demographics(data):
    """Display follower demographic analysis"""
    st.header("[DEMOGRAPHICS] Follower Demographics & Influence")
    
    raw_data = get_raw_data(data)
    demo_data = raw_data.get('follower_demographics', {})
    
    if not demo_data or demo_data.get('total_analyzed', 0) == 0:
        st.warning("No demographic data available. Collect follower data first.")
        return
    
    # Overview metrics
    st.subheader("👥 Demographic Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("Followers Analyzed", f"{demo_data.get('total_analyzed', 0):,}")
    col2.metric("Verified", f"{demo_data.get('verified_count', 0):,} ({demo_data.get('verified_percentage', 0):.1f}%)")
    col3.metric("Avg Account Age", f"{demo_data.get('avg_account_age_years', 0):.1f} years")
    col4.metric("Influential Followers", f"{demo_data.get('influential_count', 0):,}", help="Followers with >10k followers")
    
    # Average metrics
    st.subheader("📊 Average Follower Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("Avg Followers", f"{demo_data.get('avg_followers_per_follower', 0):,.0f}")
    col2.metric("Avg Following", f"{demo_data.get('avg_following_per_follower', 0):,.0f}")
    col3.metric("Avg Tweets", f"{demo_data.get('avg_tweets_per_follower', 0):,.0f}")
    col4.metric("Avg Follower Ratio", f"{demo_data.get('avg_follower_ratio', 0):.2f}", 
               help="Average followers/following ratio")
    
    # Account age distribution
    st.subheader("📅 Account Age Distribution")
    age_dist = demo_data.get('account_age_distribution', {})
    
    if age_dist:
        df = pd.DataFrame({
            'Age Range': ['Under 1 year', '1-3 years', '3-5 years', '5-10 years', 'Over 10 years'],
            'Count': [
                age_dist.get('under_1_year', 0),
                age_dist.get('1_3_years', 0),
                age_dist.get('3_5_years', 0),
                age_dist.get('5_10_years', 0),
                age_dist.get('over_10_years', 0)
            ]
        })
        
        fig = px.bar(df, x='Age Range', y='Count',
                    title="Follower Account Age Distribution",
                    color='Count',
                    color_continuous_scale='Viridis')
        st.plotly_chart(fig, use_container_width=True)
    
    # Influential followers
    st.subheader("⭐ Most Influential Followers")
    influential = demo_data.get('influential_followers', [])
    
    if influential:
        st.markdown(f"**Top {min(10, len(influential))} most influential followers (10k+ followers):**")
        
        for i, follower in enumerate(influential[:10], 1):
            with st.expander(f"#{i} @{follower['username']} - {follower['followers']:,} followers"):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**Name:** {follower['name']}")
                    st.write(f"**Location:** {follower.get('location', 'Unknown')}")
                with col2:
                    if follower.get('verified'):
                        st.success("Verified")
                    st.metric("Followers", f"{follower['followers']:,}")
    else:
        st.info("No influential followers found in sample")

def display_follower_segmentation(data):
    """Display follower segmentation analysis"""
    st.header("[SEGMENTS] Follower Segmentation & Analysis")
    
    raw_data = get_raw_data(data)
    seg_data = raw_data.get('follower_segmentation', {})
    
    if not seg_data:
        st.warning("No segmentation data available. Collect follower data first.")
        return
    
    # Activity levels
    st.subheader("📈 Activity Level Segmentation")
    activity = seg_data.get('activity_levels', {})
    
    if activity:
        col1, col2, col3, col4 = st.columns(4)
        
        very_active = activity.get('very_active', {})
        col1.metric("Very Active", 
                   f"{very_active.get('count', 0)} ({very_active.get('percentage', 0):.1f}%)",
                   help=very_active.get('description', ''))
        
        active = activity.get('active', {})
        col2.metric("Active", 
                   f"{active.get('count', 0)} ({active.get('percentage', 0):.1f}%)",
                   help=active.get('description', ''))
        
        moderate = activity.get('moderate', {})
        col3.metric("Moderate", 
                   f"{moderate.get('count', 0)} ({moderate.get('percentage', 0):.1f}%)",
                   help=moderate.get('description', ''))
        
        low = activity.get('low_activity', {})
        col4.metric("Low Activity", 
                   f"{low.get('count', 0)} ({low.get('percentage', 0):.1f}%)",
                   help=low.get('description', ''))
        
        # Activity pie chart
        fig = go.Figure(data=[go.Pie(
            labels=['Very Active', 'Active', 'Moderate', 'Low Activity'],
            values=[
                very_active.get('count', 0),
                active.get('count', 0),
                moderate.get('count', 0),
                low.get('count', 0)
            ],
            marker=dict(colors=['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728'])
        )])
        fig.update_layout(title="Follower Activity Distribution", height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Influence levels
    st.subheader("💫 Influence Level Distribution")
    influence = seg_data.get('influence_levels', {})
    
    if influence:
        col1, col2, col3, col4 = st.columns(4)
        
        mega = influence.get('mega_influencers', {})
        col1.metric("Mega Influencers", 
                   f"{mega.get('count', 0)} ({mega.get('percentage', 0):.1f}%)",
                   help=mega.get('description', ''))
        
        macro = influence.get('macro_influencers', {})
        col2.metric("Macro Influencers", 
                   f"{macro.get('count', 0)} ({macro.get('percentage', 0):.1f}%)",
                   help=macro.get('description', ''))
        
        micro = influence.get('micro_influencers', {})
        col3.metric("Micro Influencers", 
                   f"{micro.get('count', 0)} ({micro.get('percentage', 0):.1f}%)",
                   help=micro.get('description', ''))
        
        regular = influence.get('regular_users', {})
        col4.metric("Regular Users", 
                   f"{regular.get('count', 0)} ({regular.get('percentage', 0):.1f}%)",
                   help=regular.get('description', ''))
        
        # Influence bar chart
        df = pd.DataFrame({
            'Level': ['Mega (>100k)', 'Macro (10k-100k)', 'Micro (1k-10k)', 'Regular (<1k)'],
            'Count': [
                mega.get('count', 0),
                macro.get('count', 0),
                micro.get('count', 0),
                regular.get('count', 0)
            ]
        })
        
        fig = px.bar(df, x='Level', y='Count',
                    title="Follower Influence Distribution",
                    color='Count',
                    color_continuous_scale='Purples')
        st.plotly_chart(fig, use_container_width=True)
    
    # Engagement types
    st.subheader("🔄 Engagement Type Distribution")
    engagement = seg_data.get('engagement_types', {})
    
    if engagement:
        col1, col2, col3 = st.columns(3)
        
        broadcasters = engagement.get('broadcasters', {})
        col1.metric("Broadcasters", 
                   f"{broadcasters.get('count', 0)} ({broadcasters.get('percentage', 0):.1f}%)",
                   help=broadcasters.get('description', ''))
        
        connectors = engagement.get('connectors', {})
        col2.metric("Connectors", 
                   f"{connectors.get('count', 0)} ({connectors.get('percentage', 0):.1f}%)",
                   help=connectors.get('description', ''))
        
        followers_type = engagement.get('followers', {})
        col3.metric("Followers", 
                   f"{followers_type.get('count', 0)} ({followers_type.get('percentage', 0):.1f}%)",
                   help=followers_type.get('description', ''))
    
    # Political interest
    st.subheader("🏛️ Political Interest Analysis")
    political = seg_data.get('political_interest', {})
    
    if political:
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.metric("Politically Interested Followers", 
                     f"{political.get('count', 0)} ({political.get('percentage', 0):.1f}%)",
                     help="Followers with politics-related keywords in bio")
        
        with col2:
            sample = political.get('sample', [])
            if sample:
                st.markdown("**Sample of politically-interested followers:**")
                for follower in sample[:5]:
                    st.write(f"• **@{follower['username']}** - {follower['name']}")
                    st.caption(follower['bio'][:100] + "...")

def display_insights(data):
    """Display extracted insights"""
    st.header("[INSIGHTS] Actionable Insights & Recommendations")
    
    advanced = get_advanced_analysis(data)
    raw_data = get_raw_data(data)
    
    # Try to load insights from advanced analysis first, then separate file
    insights = None
    if advanced:
        # Build insights from advanced analysis
        inline_counts = advanced.get('inline_counts', {})
        tweets_analyzed = len(advanced.get('analyzed_tweets', [])) or inline_counts.get('own_tweets', len(raw_data.get('user_tweets', {}).get('data', [])))
        mentions_analyzed = len(advanced.get('analyzed_mentions', [])) or inline_counts.get('mentions', len(raw_data.get('mentions', {}).get('data', [])))
        
        insights = {
            'summary': {
                'total_reach': raw_data.get('user_profile', {}).get('data', {}).get('public_metrics', {}).get('followers_count', 0),
                'tweets_analyzed': tweets_analyzed,
                'mentions_analyzed': mentions_analyzed,
            },
            'sentiment_insights': {},
            'engagement_insights': {},
            'recommendations': advanced.get('recommendations', [])
        }
        
        # Add sentiment from advanced
        sentiment_summary = advanced.get('sentiment_summary', {})
        if sentiment_summary.get('own_tweets'):
            own = sentiment_summary['own_tweets']
            insights['sentiment_insights']['own_tweets'] = {
                'dominant_sentiment': 'positive' if own.get('positive', 0) > own.get('negative', 0) else 'negative' if own.get('negative', 0) > own.get('positive', 0) else 'neutral',
                'sentiment_score': own.get('sentiment_score', 0),
                'positive_rate': own.get('positive_pct', 0),
                'negative_rate': own.get('negative_pct', 0)
            }
        
        if sentiment_summary.get('mentions'):
            mentions = sentiment_summary['mentions']
            insights['sentiment_insights']['public_perception'] = {
                'dominant_sentiment': 'positive' if mentions.get('positive', 0) > mentions.get('negative', 0) else 'negative' if mentions.get('negative', 0) > mentions.get('positive', 0) else 'neutral',
                'sentiment_score': mentions.get('sentiment_score', 0),
                'positive_rate': mentions.get('positive_pct', 0),
                'negative_rate': mentions.get('negative_pct', 0)
            }
        
        st.success("Using insights from advanced AI analysis")
    else:
        # Try to load insights file
        insights_result = load_latest_insights()
        
        if not insights_result:
            st.warning("No insights available. Run `python advanced_insights_analyzer.py` for AI-generated insights!")
            return
        
        insights, filename = insights_result
        st.sidebar.info(f"**Insights from:** {Path(filename).name}")
    
    # Summary
    st.subheader("📊 Summary")
    summary = insights.get('summary', {})
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Reach", f"{summary.get('total_reach', 0):,}")
    col2.metric("Tweets Analyzed", summary.get('tweets_analyzed', 0))
    col3.metric("Mentions Analyzed", summary.get('mentions_analyzed', 0))
    
    # Show search results instead of followers since followers isn't available
    search_count = len(data.get('raw_data', data).get('search_results', {}).get('data', []))
    col4.metric("Search Results", search_count if search_count > 0 else "N/A")
    
    # Sentiment Insights
    st.subheader("😊 Sentiment Insights")
    sent_insights = insights.get('sentiment_insights', {})
    
    col1, col2 = st.columns(2)
    
    with col1:
        if sent_insights.get('own_tweets'):
            own = sent_insights['own_tweets']
            st.metric("Own Tweets Sentiment", 
                     own['dominant_sentiment'].upper(),
                     f"{own['sentiment_score']:.1f}%")
            
            st.progress((own['positive_rate'] / 100))
            st.caption(f"Positive: {own['positive_rate']:.1f}% | Negative: {own['negative_rate']:.1f}%")
    
    with col2:
        if sent_insights.get('public_perception'):
            pub = sent_insights['public_perception']
            st.metric("Public Perception", 
                     pub['dominant_sentiment'].upper(),
                     f"{pub['sentiment_score']:.1f}%")
            
            st.progress((pub['positive_rate'] / 100))
            st.caption(f"Positive: {pub['positive_rate']:.1f}% | Negative: {pub['negative_rate']:.1f}%")
    
    # Engagement Insights
    st.subheader("📈 Engagement Insights")
    eng_insights = insights.get('engagement_insights', {})
    
    if eng_insights:
        col1, col2, col3 = st.columns(3)
        
        col1.metric("Avg Engagement/Tweet", 
                   f"{eng_insights.get('avg_engagement_per_tweet', 0):.1f}")
        col2.metric("Best Metric", 
                   eng_insights.get('best_performing_metric', 'N/A').title())
        col3.metric("Top Tweet", 
                   f"{eng_insights.get('top_tweet_engagement', 0):,}")
    
    # Geographic Insights (from engagement, not followers)
    st.subheader("🌍 Geographic Insights")
    follower_insights = insights.get('follower_insights', {})
    
    # Try follower geography first, fallback to engagement geography
    if follower_insights.get('geography') and follower_insights['geography'].get('lebanon_percentage', 0) > 0:
        geo = follower_insights['geography']
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Lebanon vs Diaspora
            fig = go.Figure(data=[go.Pie(
                labels=['Lebanon', 'Diaspora'],
                values=[geo['lebanon_percentage'], geo['diaspora_percentage']],
                marker=dict(colors=['#00A651', '#EE161F'])
            )])
            fig.update_layout(title="Follower Geographic Split", height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.metric("Location Diversity", f"{geo['location_diversity']} locations")
            if geo['top_location']:
                st.write(f"**Top Location:** {geo['top_location'][0]}")
                st.write(f"**Followers:** {geo['top_location'][1]}")
    else:
        # Show engagement geography instead
        from collections import Counter
        raw_data = get_raw_data(data)
        all_locations = []
        
        # Collect from mentions, search, and retweets
        mentions_loc = raw_data.get('location_analysis', {}).get('mentions', {})
        if mentions_loc and mentions_loc.get('user_locations'):
            all_locations.extend(mentions_loc['user_locations'])
        
        search_loc = raw_data.get('location_analysis', {}).get('search_results', {})
        if search_loc and search_loc.get('user_locations'):
            all_locations.extend(search_loc['user_locations'])
        
        engagement_geo = raw_data.get('engagement_geography', {})
        for tweet_id, tweet_geo in engagement_geo.items():
            retweet_geo = tweet_geo.get('retweeting_users_geography', {})
            if retweet_geo and retweet_geo.get('top_locations'):
                for loc, count in retweet_geo['top_locations'].items():
                    all_locations.extend([loc] * count)
        
        if all_locations:
            lebanese_keywords = [
                'beirut', 'بيروت', 'lebanon', 'لبنان', 'tripoli', 'sidon', 'tyre', 
                'baalbek', 'zahle', 'jounieh', 'choueifat', 'bekaa', 'mount lebanon', 'liban'
            ]
            
            lebanon_count = sum(1 for loc in all_locations if any(k in loc.lower() for k in lebanese_keywords))
            diaspora_count = len(all_locations) - lebanon_count
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = go.Figure(data=[go.Pie(
                    labels=['Lebanon', 'Diaspora'],
                    values=[lebanon_count, diaspora_count],
                    marker=dict(colors=['#00A651', '#EE161F'])
                )])
                fig.update_layout(title="Engagement Geographic Split", height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.metric("Total Locations", len(all_locations))
                top_loc = Counter(all_locations).most_common(1)[0]
                st.write(f"**Top Location:** {top_loc[0]}")
                st.write(f"**Engagements:** {top_loc[1]}")
                st.info("📍 Based on engagement data (search, mentions, retweets)")
        else:
            st.info("ℹ️ Geographic data not available. Run comprehensive fetcher to collect engagement geography.")
    
    # Follower Demographics (if available)
    if follower_insights.get('demographics') and follower_insights['demographics'].get('total_analyzed', 0) > 0:
        st.subheader("👥 Follower Demographics")
        demo = follower_insights['demographics']
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Verified", f"{demo['verified_percentage']:.1f}%")
        col2.metric("Avg Age", f"{demo['avg_account_age']:.1f}y")
        col3.metric("Influencers", demo['influential_count'])
        col4.metric("Avg Followers", f"{demo['avg_follower_quality']:.0f}")
    
    # Follower Segmentation (if available)
    if follower_insights.get('segmentation') and follower_insights['segmentation'].get('activity_levels'):
        st.subheader("📊 Follower Segmentation")
        seg = follower_insights['segmentation']
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Active Users", f"{seg['active_percentage']:.1f}%")
        col2.metric("Influencers", f"{seg['influencer_percentage']:.1f}%")
        col3.metric("Political Interest", f"{seg['politically_engaged_percentage']:.1f}%")
    
    # Recommendations
    st.subheader("💡 Recommendations")
    recommendations = insights.get('recommendations', [])
    
    if recommendations:
        for rec in recommendations:
            priority_color = {
                'HIGH': '🔴',
                'MEDIUM': '🟡',
                'POSITIVE': '🟢',
                'OPPORTUNITY': '🔵'
            }.get(rec['priority'], '⚪')
            
            with st.expander(f"{priority_color} [{rec['priority']}] {rec['category']} - {rec.get('impact', 'MEDIUM')} Impact"):
                st.write(f"**Issue:** {rec.get('issue', 'N/A')}")
                st.write(f"**Action:** {rec.get('action', 'N/A')}")
    else:
        st.info("No specific recommendations at this time. System is running well!")

def display_hashtag_campaign(data):
    """Display hashtag campaign analysis"""
    st.header("[CAMPAIGN] Hashtag Campaign Analytics")
    
    st.markdown("""
    This page analyzes campaign hashtags to track performance, sentiment, and reach.
    Run `hashtag_campaign_analyzer.py` first to collect campaign data.
    """)
    
    # Load available campaigns
    campaign_files = glob.glob("campaign_*.json")
    
    if not campaign_files:
        st.warning("No campaign data found.")
        st.info("""
        **To analyze a hashtag campaign:**
        
        ```bash
        python hashtag_campaign_analyzer.py
        ```
        
        Enter your Bearer Token and hashtag (e.g., YallaRegister, Lebanon2024)
        """)
        return
    
    # List available campaigns
    campaigns = {}
    for file in campaign_files:
        filename = Path(file).stem
        hashtag = filename.replace('campaign_', '').rsplit('_', 2)[0]
        if hashtag not in campaigns or Path(file).stat().st_mtime > Path(campaigns[hashtag]).stat().st_mtime:
            campaigns[hashtag] = file
    
    # Campaign selector
    selected_hashtag = st.selectbox("Select Campaign", list(campaigns.keys()))
    
    if not selected_hashtag:
        return
    
    # Load campaign data
    with open(campaigns[selected_hashtag], 'r', encoding='utf-8') as f:
        campaign = json.load(f)
    
    performance = campaign.get('performance', {})
    sentiment = campaign.get('sentiment', {})
    insights = campaign.get('insights', {})
    
    # Overview
    st.subheader(f"📊 Campaign: #{selected_hashtag}")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Tweets", performance.get('total_tweets', 0))
    col2.metric("Unique Authors", performance.get('unique_authors', 0))
    col3.metric("Campaign Health", insights.get('campaign_health', 'Unknown'))
    col4.metric("Verified Users", f"{performance.get('verified_percentage', 0):.1f}%")
    
    # Engagement Metrics
    st.subheader("📈 Engagement Metrics")
    eng = performance.get('engagement_metrics', {})
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Engagement", f"{eng.get('total_engagement', 0):,}")
    col2.metric("Avg Likes", f"{eng.get('avg_likes', 0):.1f}")
    col3.metric("Avg Retweets", f"{eng.get('avg_retweets', 0):.1f}")
    
    # Engagement breakdown
    df = pd.DataFrame({
        'Metric': ['Likes', 'Retweets', 'Replies', 'Quotes'],
        'Total': [
            eng.get('total_likes', 0),
            eng.get('total_retweets', 0),
            eng.get('total_replies', 0),
            eng.get('total_quotes', 0)
        ]
    })
    
    fig = px.bar(df, x='Metric', y='Total', title="Engagement Breakdown",
                color='Total', color_continuous_scale='Blues')
    st.plotly_chart(fig, use_container_width=True)
    
    # Reach Metrics
    st.subheader("🌍 Reach Metrics")
    reach = performance.get('reach_metrics', {})
    
    col1, col2 = st.columns(2)
    col1.metric("Total Impressions", f"{reach.get('total_impressions', 0):,}")
    col2.metric("Potential Reach", f"{reach.get('potential_reach', 0):,}")
    
    # Sentiment Analysis
    st.subheader("😊 Campaign Sentiment")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Sentiment pie chart
        fig = go.Figure(data=[go.Pie(
            labels=['Positive', 'Negative', 'Neutral'],
            values=[
                sentiment.get('positive', 0),
                sentiment.get('negative', 0),
                sentiment.get('neutral', 0)
            ],
            marker=dict(colors=['#28a745', '#dc3545', '#6c757d'])
        )])
        fig.update_layout(title="Sentiment Distribution", height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.metric("Sentiment Score", f"{sentiment.get('sentiment_score', 0):.1f}%")
        st.metric("Positive", f"{sentiment.get('positive_percentage', 0):.1f}%")
        st.metric("Negative", f"{sentiment.get('negative_percentage', 0):.1f}%")
        st.metric("Neutral", f"{sentiment.get('neutral_percentage', 0):.1f}%")
    
    # Key Insights
    st.subheader("💡 Campaign Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**✅ Strengths:**")
        for strength in insights.get('strengths', []):
            st.success(f"{strength}")
        
        if not insights.get('strengths'):
            st.info("Analyzing campaign strengths...")
    
    with col2:
        st.markdown("**⚠️ Areas for Improvement:**")
        for weakness in insights.get('weaknesses', []):
            st.warning(f"✗ {weakness}")
        
        if not insights.get('weaknesses'):
            st.success("No major issues detected!")
    
    # Recommendations
    st.subheader("🎯 Recommendations")
    recommendations = insights.get('recommendations', [])
    
    if recommendations:
        for rec in recommendations:
            priority_color = {
                'HIGH': 'error',
                'MEDIUM': 'warning',
                'OPPORTUNITY': 'info'
            }.get(rec['priority'], 'info')
            
            with st.expander(f"[{rec['priority']}] {rec['action']}"):
                st.write(f"**Details:** {rec['details']}")
    else:
        st.info("Campaign is performing well. Keep monitoring!")
    
    # Top Tweets
    st.subheader("🔥 Top Performing Tweets")
    top_tweets = performance.get('top_tweets', [])
    
    for i, tweet in enumerate(top_tweets[:5], 1):
        with st.expander(f"#{i} - @{tweet['author']} ({tweet['total_engagement']:,} engagements)"):
            st.write(tweet['text'])
            col1, col2, col3 = st.columns(3)
            col1.metric("Likes", tweet['likes'])
            col2.metric("Retweets", tweet['retweets'])
            col3.metric("Replies", tweet['replies'])

def display_public_opinion(data):
    """Display public opinion dashboard with AI sentiment"""
    st.header("[OPINION] Public Opinion Tracker (AI-Analyzed)")
    
    # Get AI sentiment summary
    advanced = get_advanced_analysis(data)
    sentiment_summary = advanced.get('sentiment_summary', {})
    
    # Fallback to old format if advanced not available
    if not sentiment_summary:
        raw_data = get_raw_data(data)
        sentiment_summary = raw_data.get('sentiment_analysis', {})
    
    mentions_sent = sentiment_summary.get('mentions', {})
    search_sent = sentiment_summary.get('search_results', {})
    
    # Calculate overall sentiment
    total_pos = mentions_sent.get('positive', 0) + search_sent.get('positive', 0)
    total_neg = mentions_sent.get('negative', 0) + search_sent.get('negative', 0)
    total_neu = mentions_sent.get('neutral', 0) + search_sent.get('neutral', 0)
    total = total_pos + total_neg + total_neu
    
    if total == 0:
        st.warning("No public opinion data available")
        return
    
    # Opinion score (-100 to +100)
    opinion_score = ((total_pos - total_neg) / total) * 100
    
    # Display opinion gauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=opinion_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Public Opinion Score"},
        delta={'reference': 0},
        gauge={
            'axis': {'range': [-100, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [-100, -33], 'color': "lightcoral"},
                {'range': [-33, 33], 'color': "lightyellow"},
                {'range': [33, 100], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 0
            }
        }
    ))
    
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Interpretation
    if opinion_score > 33:
        st.success(f"[POSITIVE] **Strong Positive** public sentiment ({opinion_score:.1f}/100)")
    elif opinion_score > 0:
        st.info(f"[POSITIVE] **Slightly Positive** public sentiment ({opinion_score:.1f}/100)")
    elif opinion_score > -33:
        st.warning(f"[NEGATIVE] **Slightly Negative** public sentiment ({opinion_score:.1f}/100)")
    else:
        st.error(f"[NEGATIVE] **Strong Negative** public sentiment ({opinion_score:.1f}/100)")
    
    # Breakdown
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Responses", f"{total:,}")
    col2.metric("Positive", f"{total_pos:,}", f"{total_pos/total*100:.1f}%")
    col3.metric("Negative", f"{total_neg:,}", f"{total_neg/total*100:.1f}%")
    col4.metric("Neutral", f"{total_neu:,}", f"{total_neu/total*100:.1f}%")

def display_emotion_analysis(data):
    """Display emotion analysis from advanced AI"""
    st.header("[EMOTION] Emotion Detection Analysis")
    st.markdown("**Powered by**: DistilRoBERTa Emotion Detection Model")
    
    # Get advanced analysis
    advanced = data.get('advanced_analysis', {})
    emotion_summary = advanced.get('emotion_summary', {})
    
    if not emotion_summary:
        st.warning("No emotion analysis available. Run advanced_insights_analyzer.py on your data.")
        return
    
    # Display emotion breakdown
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Own Tweets Emotions")
        own_emotions = emotion_summary.get('own_tweets', {})
        if own_emotions and own_emotions.get('distribution'):
            dist = own_emotions['distribution']
            dominant = own_emotions.get('dominant', 'neutral')
            
            st.metric("Dominant Emotion", dominant.upper(), help="Most common emotion in your tweets")
            
            # Create pie chart
            fig = px.pie(
                values=list(dist.values()),
                names=list(dist.keys()),
                title="Emotion Distribution in Own Tweets"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Public Emotions (Mentions)")
        mentions_emotions = emotion_summary.get('mentions', {})
        if mentions_emotions and mentions_emotions.get('distribution'):
            dist = mentions_emotions['distribution']
            dominant = mentions_emotions.get('dominant', 'neutral')
            
            st.metric("Dominant Emotion", dominant.upper(), help="Most common emotion in public mentions")
            
            # Create pie chart
            fig = px.pie(
                values=list(dist.values()),
                names=list(dist.keys()),
                title="Emotion Distribution in Mentions"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Detailed percentages
    st.markdown("---")
    st.subheader("📊 Detailed Emotion Percentages")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if own_emotions and own_emotions.get('percentages'):
            st.markdown("**Own Tweets:**")
            for emotion, pct in sorted(own_emotions['percentages'].items(), key=lambda x: x[1], reverse=True):
                st.write(f"- **{emotion.title()}**: {pct:.1f}%")
    
    with col2:
        if mentions_emotions and mentions_emotions.get('percentages'):
            st.markdown("**Mentions:**")
            for emotion, pct in sorted(mentions_emotions['percentages'].items(), key=lambda x: x[1], reverse=True):
                st.write(f"- **{emotion.title()}**: {pct:.1f}%")

def display_influence_metrics(data):
    """Display influence and impact metrics"""
    st.header("[INFLUENCE] Influence & Impact Metrics")
    
    advanced = data.get('advanced_analysis', {})
    influence = advanced.get('influence_metrics', {})
    
    if not influence:
        st.warning("No influence metrics available. Run advanced_insights_analyzer.py on your data.")
        return
    
    # Top metrics
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric(
        "Influence Tier",
        influence.get('influence_tier', 'Unknown'),
        help="Your influence classification"
    )
    
    col2.metric(
        "Influence Score",
        f"{influence.get('influence_score', 0):,.0f}",
        help="Composite influence metric"
    )
    
    col3.metric(
        "Engagement Rate",
        f"{influence.get('engagement_rate', 0):.3f}%",
        help="Percentage of followers who engage"
    )
    
    col4.metric(
        "Virality Score",
        f"{influence.get('virality_score', 0):.2f}",
        help="Potential for content to go viral"
    )
    
    st.markdown("---")
    
    # Detailed metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📊 Reach Metrics")
        st.metric("Reach Potential", f"{influence.get('reach_potential', 0):,} followers")
        st.metric("Follower Ratio", f"{influence.get('follower_ratio', 0):.2f}x")
        st.caption("Followers/Following ratio")
    
    with col2:
        st.subheader("💬 Engagement Metrics")
        st.metric("Avg Engagement/Tweet", f"{influence.get('avg_engagement_per_tweet', 0):.1f}")
        st.caption("Likes + Retweets + Replies per tweet")
    
    with col3:
        st.subheader("📝 Content Volume")
        st.metric("Total Content", f"{influence.get('content_volume', 0):,}")
        st.caption("Lifetime tweets published")
    
    # Interpretation
    st.markdown("---")
    st.subheader("💡 What This Means")
    
    tier = influence.get('influence_tier', '')
    if 'Mega' in tier:
        st.success("🌟 **Mega Influencer**: You have massive reach and significant impact on public discourse.")
    elif 'Macro' in tier:
        st.success("✨ **Macro Influencer**: You have substantial influence and a strong voice in your domain.")
    elif 'Micro' in tier:
        st.info("📢 **Micro Influencer**: You have meaningful influence within your community.")
    elif 'Rising' in tier:
        st.info("📈 **Rising Voice**: Your influence is growing. Keep building momentum!")
    else:
        st.info("🌱 **Emerging Voice**: You're building your presence. Focus on engagement and consistency.")

def display_content_strategy(data):
    """Display content strategy analysis"""
    st.header("[CONTENT] Content Strategy Analysis")
    
    advanced = data.get('advanced_analysis', {})
    content = advanced.get('content_strategy', {})
    
    if not content:
        st.warning("No content strategy analysis available.")
        return
    
    # Best performing content type
    st.subheader("🎯 Best Performing Content Type")
    best_type = content.get('best_performing_type', 'unknown')
    
    if best_type != 'unknown':
        st.success(f"**{best_type.upper()}** content performs best!")
        st.info(f"**Recommendation**: Increase your use of {best_type} in future posts to maximize engagement.")
    
    st.markdown("---")
    
    # Content distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Content Type Distribution")
        dist = content.get('content_distribution', {})
        if dist:
            fig = px.bar(
                x=list(dist.keys()),
                y=list(dist.values()),
                title="Number of Posts by Content Type",
                labels={'x': 'Content Type', 'y': 'Count'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("⚡ Performance by Content Type")
        perf = content.get('performance_by_type', {})
        if perf:
            fig = px.bar(
                x=list(perf.keys()),
                y=list(perf.values()),
                title="Average Engagement by Content Type",
                labels={'x': 'Content Type', 'y': 'Avg Engagement'},
                color=list(perf.values()),
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Detailed stats
    st.markdown("---")
    st.subheader("📈 Detailed Performance Stats")
    
    cols = st.columns(4)
    for idx, (content_type, engagement) in enumerate(perf.items()):
        with cols[idx % 4]:
            st.metric(
                content_type.title(),
                f"{engagement:.1f}",
                help=f"Average engagement for {content_type} posts"
            )

def display_entity_network(data):
    """Display named entity network"""
    st.header("[ENTITIES] Named Entity Network")
    
    advanced = get_advanced_analysis(data)
    entities = advanced.get('entity_network', {})
    
    if not entities:
        st.warning("No entity network available.")
        return
    
    st.success("**Primary Source**: Twitter API Annotations (Arabic + English)")
    st.info("**Fallback**: BERT NER for tweets without annotations")
    st.markdown("---")
    
    # Entity type explanations with emojis and better formatting
    with st.expander("📚 Understanding Entity Types & Indicators", expanded=False):
        st.markdown("""
        ### Entity Types:
        
        - 👤 **PER (Person)**: Politicians, public figures, individuals
        - 🏢 **ORG (Organization)**: Political parties, institutions, companies
          - Example: حزب الله (Hezbollah), Parliament, Government bodies
        - 📍 **LOC (Location)**: Cities, countries, regions, landmarks
          - Example: الليطاني (Litani River), Beirut, Lebanon
        - 🏷️ **MISC (Miscellaneous)**: Events, products, other entities
        
        ### Sentiment Indicators:
        - 😊 **Positive**: Entity mentioned positively (supportive, praise)
        - 😐 **Neutral**: Entity mentioned factually (no opinion)
        - 😟 **Negative**: Entity mentioned negatively (criticism, opposition)
        
        ### What This Shows:
        - **Count**: How often the entity is mentioned
        - **Dominant Sentiment**: Overall sentiment in mentions
        - **Sentiment Breakdown**: Positive/Negative/Neutral distribution
        """)
    
    # Top entities by count
    st.subheader("📊 Most Mentioned Entities")
    
    entity_data = []
    for entity, info in list(entities.items())[:20]:
        types_str = ', '.join(info['types'])
        sentiment = info['dominant_sentiment']
        
        entity_data.append({
            'Entity': entity,
            'Mentions': info['count'],
            'Type': types_str,
            'Sentiment': sentiment.title(),
            'Positive': info['sentiment'].get('positive', 0),
            'Negative': info['sentiment'].get('negative', 0),
            'Neutral': info['sentiment'].get('neutral', 0)
        })
    
    if entity_data:
        df = pd.DataFrame(entity_data)
        
        # Bar chart of top entities
        fig = px.bar(
            df.head(15),
            x='Mentions',
            y='Entity',
            orientation='h',
            title='Top 15 Most Mentioned Entities',
            color='Type',
            hover_data=['Sentiment', 'Positive', 'Negative', 'Neutral']
        )
        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            showlegend=True,
            height=500
        )
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
        
        # Data table
        st.subheader("📋 Entity Details")
        st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Entity sentiment breakdown
    st.markdown("---")
    st.subheader("😊 Entity Sentiment Breakdown")
    
    cols = st.columns(3)
    with cols[0]:
        positive_entities = sum(1 for e in entities.values() if e['dominant_sentiment'] == 'positive')
        st.metric("Positive Entities", positive_entities, help="Entities mentioned in positive context")
    
    with cols[1]:
        negative_entities = sum(1 for e in entities.values() if e['dominant_sentiment'] == 'negative')
        st.metric("Negative Entities", negative_entities, help="Entities mentioned in negative context")
    
    with cols[2]:
        neutral_entities = sum(1 for e in entities.values() if e['dominant_sentiment'] == 'neutral')
        st.metric("Neutral Entities", neutral_entities, help="Entities mentioned in neutral context")

def display_temporal_insights(data):
    """Display temporal patterns and trends"""
    st.header("[TIME] Temporal Patterns & Trends")
    
    advanced = data.get('advanced_analysis', {})
    temporal = advanced.get('temporal_patterns', {})
    
    if not temporal:
        st.warning("No temporal analysis available.")
        return
    
    # Best posting times
    st.subheader("🎯 Optimal Posting Times")
    best_hours = temporal.get('best_posting_hours', [])
    
    if best_hours:
        st.success(f"**Peak Hour**: {best_hours[0]['hour']}:00 with {best_hours[0]['avg_engagement']:.1f} avg engagement")
        
        # Chart of best hours
        hours = [h['hour'] for h in best_hours]
        engagement = [h['avg_engagement'] for h in best_hours]
        
        fig = px.bar(
            x=hours,
            y=engagement,
            labels={'x': 'Hour of Day', 'y': 'Average Engagement'},
            title='Top 5 Hours for Maximum Engagement',
            color=engagement,
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.info(f"**Recommendation**: Schedule important posts around **{best_hours[0]['hour']}:00** for maximum impact.")
    
    st.markdown("---")
    
    # Activity patterns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📅 Hourly Activity")
        hourly = temporal.get('hourly_activity', {})
        if hourly:
            fig = px.bar(
                x=list(hourly.keys()),
                y=list(hourly.values()),
                title='Posts by Hour of Day',
                labels={'x': 'Hour', 'y': 'Number of Posts'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📆 Daily Activity")
        daily = temporal.get('daily_activity', {})
        if daily:
            fig = px.bar(
                x=list(daily.keys()),
                y=list(daily.values()),
                title='Posts by Day of Week',
                labels={'x': 'Day', 'y': 'Number of Posts'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Sentiment trends
    st.markdown("---")
    st.subheader("📈 Sentiment Trends Over Time")
    
    sentiment_trends = temporal.get('sentiment_trends', {})
    if sentiment_trends:
        # Convert to dataframe
        dates = []
        positive = []
        negative = []
        neutral = []
        
        for date_str, counts in sentiment_trends.items():
            dates.append(date_str)
            positive.append(counts.get('positive', 0))
            negative.append(counts.get('negative', 0))
            neutral.append(counts.get('neutral', 0))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=positive, name='Positive', mode='lines+markers', line=dict(color='green')))
        fig.add_trace(go.Scatter(x=dates, y=negative, name='Negative', mode='lines+markers', line=dict(color='red')))
        fig.add_trace(go.Scatter(x=dates, y=neutral, name='Neutral', mode='lines+markers', line=dict(color='gray')))
        
        fig.update_layout(
            title='Sentiment Trends Over Time',
            xaxis_title='Date',
            yaxis_title='Number of Tweets',
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)

def display_advanced_recommendations(data):
    """Display AI-generated recommendations"""
    st.header("[💡 RECOMMENDATIONS] AI-Generated Action Items")
    
    advanced = data.get('advanced_analysis', {})
    recommendations = advanced.get('recommendations', [])
    
    if not recommendations:
        st.warning("No recommendations available.")
        return
    
    st.markdown("**Based on**: Comprehensive analysis of sentiment, engagement, content, and timing patterns")
    
    # Group by priority
    high_priority = [r for r in recommendations if r.get('priority') == 'HIGH']
    medium_priority = [r for r in recommendations if r.get('priority') == 'MEDIUM']
    opportunities = [r for r in recommendations if r.get('priority') in ['OPPORTUNITY', 'POSITIVE']]
    
    # High priority issues
    if high_priority:
        st.subheader("🔴 High Priority Actions")
        for idx, rec in enumerate(high_priority, 1):
            with st.expander(f"#{idx} - {rec.get('category', 'General')}: {rec.get('issue', 'Action needed')}", expanded=True):
                st.markdown(f"**Issue:** {rec.get('issue', 'N/A')}")
                st.markdown(f"**Action:** {rec.get('action', 'N/A')}")
                st.markdown(f"**Impact:** `{rec.get('impact', 'Unknown')}`")
    
    # Medium priority
    if medium_priority:
        st.subheader("🟡 Medium Priority Actions")
        for idx, rec in enumerate(medium_priority, 1):
            with st.expander(f"#{idx} - {rec.get('category', 'General')}: {rec.get('issue', 'Action needed')}"):
                st.markdown(f"**Issue:** {rec.get('issue', 'N/A')}")
                st.markdown(f"**Action:** {rec.get('action', 'N/A')}")
                st.markdown(f"**Impact:** `{rec.get('impact', 'Unknown')}`")
    
    # Opportunities
    if opportunities:
        st.subheader("🟢 Opportunities & Strengths")
        for idx, rec in enumerate(opportunities, 1):
            with st.expander(f"#{idx} - {rec.get('category', 'General')}: {rec.get('issue', 'Opportunity')}"):
                st.markdown(f"**Opportunity:** {rec.get('issue', 'N/A')}")
                st.markdown(f"**Action:** {rec.get('action', 'N/A')}")
                st.markdown(f"**Impact:** `{rec.get('impact', 'Unknown')}`")
    
    # Summary
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    col1.metric("High Priority", len(high_priority))
    col2.metric("Medium Priority", len(medium_priority))
    col3.metric("Opportunities", len(opportunities))

@st.cache_resource
def init_vector_db():
    """Initialize ChromaDB connection and embedding model"""
    try:
        client = chromadb.PersistentClient(path="./chroma_db")
        collection = client.get_or_create_collection(
            name="political_insights",
            metadata={"description": "Political campaign insights and analytics"}
        )
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        return collection, embedding_model
    except Exception as e:
        st.error(f"Error initializing vector database: {e}")
        return None, None

def query_vector_db(collection, embedding_model, query_text, n_results=5, filters=None, sort_by_engagement=False):
    """Query the vector database for relevant documents
    
    Args:
        sort_by_engagement: If True, sort results by engagement_score instead of similarity
    """
    if not collection or not embedding_model:
        return None
    
    try:
        # Engagement-based queries: fetch all matching tweets and sort by engagement
        if sort_by_engagement:
            where_clause = dict(filters) if filters else {}
            where_clause['type'] = 'tweet'
            
            records = collection.get(where=where_clause)
            if records and records.get('documents'):
                combined = list(zip(
                    records['documents'],
                    records['metadatas'],
                    records['ids']
                ))
                combined = [
                    item for item in combined
                    if isinstance(item[1].get('engagement_score'), (int, float))
                ]
                combined.sort(
                    key=lambda x: x[1].get('engagement_score', 0),
                    reverse=True
                )
                combined = combined[:n_results]
                
                if combined:
                    return {
                        'documents': [[item[0] for item in combined]],
                        'metadatas': [[item[1] for item in combined]],
                        'ids': [[item[2] for item in combined]]
                    }
            sort_by_engagement = False
        
        query_embedding = embedding_model.encode([query_text]).tolist()[0]
        
        # If sorting by engagement, get more results to sort from
        query_n_results = n_results * 6 if sort_by_engagement else n_results
        
        # Query with filters if provided
        if filters:
            # Build where clause - handle multiple conditions
            where_clause = {}
            
            # Check if we need to filter by is_retweet
            if "is_retweet" in filters:
                is_retweet_value = filters["is_retweet"]
                where_clause["is_retweet"] = is_retweet_value
            
            # Check if we need to filter by tweet_source
            if "tweet_source" in filters:
                where_clause["tweet_source"] = filters["tweet_source"]
            
            # Query with the filter
            if where_clause:
                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=query_n_results * 2,  # Get more results to filter
                    where=where_clause
                )
                
                # If we have multiple filter conditions, filter in Python
                if len(filters) > 1:
                    filtered_docs = []
                    filtered_metas = []
                    filtered_ids = []
                    filtered_distances = []
                    
                    if results and results['documents'] and len(results['documents']) > 0:
                        for i, metadata in enumerate(results['metadatas'][0]):
                            # Check all filter conditions
                            matches = True
                            for key, value in filters.items():
                                if metadata.get(key) != value:
                                    matches = False
                                    break
                            
                            if matches:
                                filtered_docs.append(results['documents'][0][i])
                                filtered_metas.append(metadata)
                                filtered_ids.append(results['ids'][0][i])
                                if results.get('distances') and len(results['distances']) > 0:
                                    filtered_distances.append(results['distances'][0][i])
                        
                        # Limit to n_results
                        filtered_docs = filtered_docs[:n_results]
                        filtered_metas = filtered_metas[:n_results]
                        filtered_ids = filtered_ids[:n_results]
                        if filtered_distances:
                            filtered_distances = filtered_distances[:n_results]
                        
                        results = {
                            'documents': [filtered_docs],
                            'metadatas': [filtered_metas],
                            'ids': [filtered_ids]
                        }
                        if filtered_distances:
                            results['distances'] = [filtered_distances]
                else:
                    # Single condition - limit results
                    if results and results['documents'] and len(results['documents']) > 0:
                        results['documents'] = [results['documents'][0][:n_results]]
                        results['metadatas'] = [results['metadatas'][0][:n_results]]
                        results['ids'] = [results['ids'][0][:n_results]]
                        if results.get('distances'):
                            results['distances'] = [results['distances'][0][:n_results]]
            else:
                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=query_n_results
                )
        else:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=query_n_results
            )
        
        # If sorting by engagement, sort the results by engagement_score
        if sort_by_engagement and results and results.get('metadatas') and len(results['metadatas']) > 0:
            # Combine documents, metadatas, ids, and distances for sorting
            combined = list(zip(
                results['documents'][0],
                results['metadatas'][0],
                results['ids'][0],
                results.get('distances', [[]])[0] if results.get('distances') else [0] * len(results['documents'][0])
            ))
            
            # Sort by engagement_score (descending), fallback to distance if no engagement_score
            combined.sort(key=lambda x: (
                x[1].get('engagement_score', 0) if isinstance(x[1].get('engagement_score'), (int, float)) else 0
            ), reverse=True)
            
            # Take top n_results
            combined = combined[:n_results]
            
            # Reconstruct results
            results = {
                'documents': [[item[0] for item in combined]],
                'metadatas': [[item[1] for item in combined]],
                'ids': [[item[2] for item in combined]]
            }
            if results.get('distances'):
                results['distances'] = [[item[3] for item in combined]]
        
        return results
    except Exception as e:
        st.error(f"Error querying vector database: {e}")
        return None

def get_openai_client():
    """Get OpenAI client with API key from environment or user input"""
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        # Try to get from session state (user might have entered it)
        api_key = st.session_state.get('openai_api_key')
    
    if not api_key:
        return None
    
    try:
        return OpenAI(api_key=api_key)
    except Exception as e:
        st.error(f"Error initializing OpenAI client: {e}")
        return None

def generate_chatbot_response(client, query, context_docs):
    """Generate response using OpenAI based on retrieved context"""
    if not client:
        return "OpenAI API key not configured. Please set OPENAI_API_KEY in your .env file or enter it in the sidebar."
    
    # Build context from retrieved documents
    context_text = "\n\n".join([
        f"[{i+1}] {doc}" for i, doc in enumerate(context_docs[:5])
    ])
    
    system_prompt = """You are an expert political campaign analyst and social media strategist specializing in Lebanese politics. 
Your role is to provide actionable campaign analysis and strategic advice based on social media listening data.

When answering questions:
1. Base your analysis on the provided context from social media data
2. Provide specific, actionable recommendations
3. Reference specific metrics, trends, or insights from the data when relevant
4. Focus on campaign strategy, public sentiment, engagement optimization, and audience insights
5. Be concise but thorough
6. If the context doesn't contain relevant information, say so clearly

Always structure your response to be helpful for campaign decision-making."""
    
    user_prompt = f"""Based on the following social media listening data and campaign insights, please answer this question:

Question: {query}

Relevant Context from Social Media Data:
{context_text}

Please provide a comprehensive analysis and actionable advice."""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating response: {e}"

def display_chatbot(data):
    """Display the AI chatbot interface for campaign analysis"""
    st.header("🤖 AI Campaign Analyst Chatbot")
    st.markdown("**Ask questions about your campaign performance, sentiment, engagement, and get strategic advice based on social media listening data.**")
    st.markdown("---")
    
    # Initialize session state for chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Sidebar for API key configuration
    with st.sidebar.expander("⚙️ OpenAI API Configuration", expanded=False):
        api_key_input = st.text_input(
            "OpenAI API Key",
            value=st.session_state.get('openai_api_key', ''),
            type="password",
            help="Enter your OpenAI API key. You can also set OPENAI_API_KEY in your .env file."
        )
        if api_key_input:
            st.session_state.openai_api_key = api_key_input
            os.environ['OPENAI_API_KEY'] = api_key_input
            st.success("API key saved for this session")
    
    # Check if vector DB exists
    if not os.path.exists("./chroma_db"):
        st.warning("⚠️ Vector database not found. Please run `python create_vector_db.py` first to create the embeddings database.")
        st.info("""
        **To set up the chatbot:**
        1. Make sure you have run `python comprehensive_fetcher.py` to collect data
        2. Run `python advanced_insights_analyzer.py` to generate advanced analysis
        3. Run `python create_vector_db.py` to create the vector database
        4. Refresh this page
        """)
        return
    
    # Initialize vector DB
    collection, embedding_model = init_vector_db()
    
    if not collection or not embedding_model:
        st.error("Failed to initialize vector database. Please check that the database exists.")
        return
    
    # Check collection count
    try:
        count = collection.count()
        if count == 0:
            st.warning("⚠️ Vector database is empty. Please run `python create_vector_db.py` to populate it.")
            return
        st.info(f"📊 Connected to vector database with {count} documents")
    except:
        st.warning("⚠️ Could not verify vector database. Please ensure it's properly set up.")
        return
    
    # Example questions
    st.markdown("### 💡 Example Questions")
    example_questions = [
        "What are the top performing tweets and why?",
        "What is the overall sentiment of mentions about the campaign?",
        "What topics are most discussed in relation to the campaign?",
        "What are the main concerns or negative feedback from the audience?",
        "Which hashtags are performing best?",
        "What are the key recommendations for improving engagement?",
        "What entities or topics should I focus on in my messaging?",
        "How is the campaign performing compared to engagement metrics?",
        "What are the best times to post based on engagement data?",
        "What are the main themes in negative sentiment mentions?"
    ]
    
    selected_filter = None
    cols = st.columns(2)
    for i, question in enumerate(example_questions):
        with cols[i % 2]:
            if st.button(f"💬 {question}", key=f"example_{i}", use_container_width=True):
                # Process the question directly
                if 'chat_history' not in st.session_state:
                    st.session_state.chat_history = []
                
                # Add user message to history
                st.session_state.chat_history.append(("user", question))
                
                # Initialize vector DB if not already done
                collection, embedding_model = init_vector_db()
                
                if collection and embedding_model:
                    with st.spinner("🔍 Searching campaign data and generating analysis..."):
                        # Check if question is about top performing/highest engagement
                        sort_by_engagement = any(phrase in question.lower() for phrase in [
                            'top performing', 'highest engagement', 'best performing', 
                            'most engagement', 'top tweets', 'best tweets'
                        ])
                        
                        # Query vector database
                        results = query_vector_db(
                            collection, embedding_model, question, 
                            n_results=5, 
                            filters=selected_filter,
                            sort_by_engagement=sort_by_engagement
                        )
                        
                        if results and results['documents'] and len(results['documents'][0]) > 0:
                            # Get relevant documents
                            context_docs = results['documents'][0]
                            
                            # Get OpenAI client
                            client = get_openai_client()
                            
                            # Generate response
                            response = generate_chatbot_response(client, question, context_docs)
                            
                            # Add response to history
                            st.session_state.chat_history.append(("assistant", response))
                        else:
                            error_msg = "I couldn't find relevant information in the campaign data to answer your question. Please try rephrasing or ask about a different aspect of the campaign."
                            st.session_state.chat_history.append(("assistant", error_msg))
                
                st.rerun()
    
    st.markdown("---")
    
    # Chat interface
    st.markdown("### 💬 Chat with Campaign Analyst")
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for i, (role, message) in enumerate(st.session_state.chat_history):
            if role == "user":
                with st.chat_message("user"):
                    st.write(message)
            else:
                with st.chat_message("assistant"):
                    st.write(message)
    
    # Source filter
    source_options = {
        "All sources": None,
        "Own tweets": {"tweet_source": "own_tweet", "is_retweet": False},
        "Retweets": {"is_retweet": True},
        "Mentions": {"tweet_source": "mention"},
        "Search results": {"tweet_source": "search_result"}
    }
    source_selection = st.selectbox(
        "Filter by source (optional):",
        list(source_options.keys()),
        index=0
    )
    selected_filter = source_options[source_selection]
    
    # User input
    user_input = st.text_input(
        "Ask a question about your campaign:",
        key="user_input",
        placeholder="e.g., What are the main concerns from negative sentiment mentions?"
    )
    
    # Process query
    col1, col2 = st.columns([1, 10])
    with col1:
        send_button = st.button("Send", type="primary")
    
    if send_button and user_input:
        # Add user message to history
        st.session_state.chat_history.append(("user", user_input))
        
        # Show loading
        with st.spinner("🔍 Searching campaign data and generating analysis..."):
            # Check if question is about top performing/highest engagement
            sort_by_engagement = any(phrase in user_input.lower() for phrase in [
                'top performing', 'highest engagement', 'best performing', 
                'most engagement', 'top tweets', 'best tweets'
            ])
            
            # Query vector database
            results = query_vector_db(
                collection, embedding_model, user_input, 
                n_results=5, 
                filters=selected_filter,
                sort_by_engagement=sort_by_engagement
            )
            
            if results and results['documents'] and len(results['documents'][0]) > 0:
                # Get relevant documents
                context_docs = results['documents'][0]
                
                # Get OpenAI client
                client = get_openai_client()
                
                # Generate response
                response = generate_chatbot_response(client, user_input, context_docs)
                
                # Add response to history
                st.session_state.chat_history.append(("assistant", response))
                
                # Show sources
                with st.expander("📚 Sources (Retrieved Documents)", expanded=False):
                    for i, (doc, metadata) in enumerate(zip(context_docs, results['metadatas'][0])):
                        st.markdown(f"**Source {i+1}** ({metadata.get('type', 'unknown')}):")
                        st.caption(doc[:300] + "..." if len(doc) > 300 else doc)
                        st.markdown("---")
            else:
                error_msg = "I couldn't find relevant information in the campaign data to answer your question. Please try rephrasing or ask about a different aspect of the campaign."
                st.session_state.chat_history.append(("assistant", error_msg))
        
        # Rerun to show new messages
        st.rerun()
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()
    
    st.markdown("---")
    st.markdown("""
    **How it works:**
    1. Your question is converted to an embedding using semantic search
    2. The system searches the vector database for relevant campaign insights
    3. The most relevant documents are retrieved as context
    4. OpenAI GPT generates a comprehensive analysis based on the retrieved data
    5. You get actionable campaign advice grounded in your actual social media data
    """)

def main():
    # Title
    st.title("[LEBANON] Mark Daou - Political Intelligence Dashboard")
    st.markdown("**Advanced AI-Powered Analysis with RoBERTa, Emotion Detection & NER**")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.header("[CONTROLS] Dashboard Controls")
    
    # Load data
    result = load_latest_data()
    
    if result is None:
        st.error("[ERROR] No data available")
        st.info("""
        **To collect data:**
        
        1. Get X API credentials (see `GET_API_CREDENTIALS.md`)
        2. Run: `python comprehensive_fetcher.py`
        3. Enter your Bearer Token and username
        4. Refresh this dashboard
        """)
        return
    
    data, filename = result
    
    # Display metadata
    st.sidebar.success(f"[OK] Data Loaded")
    st.sidebar.info(f"**File:** {Path(filename).name}")
    st.sidebar.info(f"**Updated:** {data.get('timestamp', 'Unknown')[:19]}")
    
    # Refresh button
    if st.sidebar.button("[REFRESH] Refresh Data"):
        st.rerun()
    
    # Get raw data
    raw_data = get_raw_data(data)
    
    # Data availability status
    st.sidebar.markdown("---")
    st.sidebar.header("📊 Data Status")
    
    # Check what's available
    has_tweets = raw_data.get('user_tweets') and raw_data.get('user_tweets', {}).get('data')
    has_mentions = raw_data.get('mentions') and raw_data.get('mentions', {}).get('data')
    has_search = raw_data.get('search_results') and raw_data.get('search_results', {}).get('data')
    has_engagement_geo = raw_data.get('engagement_geography')
    
    st.sidebar.markdown(f"""
    ✅ **Available:**
    - User Profile
    - Tweets ({len(raw_data.get('user_tweets', {}).get('data', []))})
    - Mentions ({len(raw_data.get('mentions', {}).get('data', []))})
    - Search Results ({len(raw_data.get('search_results', {}).get('data', []))})
    - Engagement Geography
    
    ⚠️ **Not Available** (requires higher tier):
    - Follower Lists
    - Following Lists
    - Like Users
    """)
    
    # Navigation
    st.sidebar.markdown("---")
    st.sidebar.header("Navigation")
    has_follower_data = (
        raw_data.get('followers') is not None and 
        raw_data.get('follower_geography', {}) and
        raw_data.get('follower_geography', {}).get('total_followers', 0) > 0
    )
    
    # Build navigation menu based on available data
    pages = [
        "[CAMPAIGN] Campaign Overview",
        "[CHATBOT] AI Campaign Analyst",
        "[INSIGHTS] Actionable Insights",
        "[AI] Advanced Recommendations",
        "[HASHTAG] Campaign Tracker",
        "[AI] Sentiment Analysis",
        "[EMOTION] Emotion Analysis",
        "[OPINION] Public Opinion",
        "[STATS] Engagement Metrics",
        "[INFLUENCE] Influence Metrics",
        "[CONTENT] Content Strategy",
        "[HASHTAGS] Hashtag Analysis",
        "[TOPICS] Topic Analysis",
        "[TIME] Temporal Insights",
        "[TIME] Posting Patterns",
        "[ENTITIES] Entity Network",
        "[AUDIENCE] Audience Analysis",
        "[GEO] Engagement Geography",
        "[DATA] Raw Data"
    ]
    
    # Only add follower pages if data exists
    if has_follower_data:
        pages.insert(-1, "[GEOGRAPHY] Follower Geography")
        pages.insert(-1, "[DEMOGRAPHICS] Follower Demographics")
        pages.insert(-1, "[SEGMENTS] Follower Segmentation")
    
    page = st.sidebar.radio("Select View:", pages)
    
    # Display selected page
    if page == "[CAMPAIGN] Campaign Overview":
        display_campaign_overview(data)
    elif page == "[CHATBOT] AI Campaign Analyst":
        display_chatbot(data)
    elif page == "[INSIGHTS] Actionable Insights":
        display_insights(data)
    elif page == "[AI] Advanced Recommendations":
        display_advanced_recommendations(data)
    elif page == "[HASHTAG] Campaign Tracker":
        display_hashtag_campaign(data)
    elif page == "[AI] Sentiment Analysis":
        display_sentiment_analysis(data)
    elif page == "[EMOTION] Emotion Analysis":
        display_emotion_analysis(data)
    elif page == "[OPINION] Public Opinion":
        display_public_opinion(data)
    elif page == "[STATS] Engagement Metrics":
        display_engagement_metrics(data)
    elif page == "[INFLUENCE] Influence Metrics":
        display_influence_metrics(data)
    elif page == "[CONTENT] Content Strategy":
        display_content_strategy(data)
    elif page == "[HASHTAGS] Hashtag Analysis":
        display_hashtag_analysis(data)
    elif page == "[TOPICS] Topic Analysis":
        display_topic_analysis(data)
    elif page == "[TIME] Temporal Insights":
        display_temporal_insights(data)
    elif page == "[TIME] Posting Patterns":
        display_posting_patterns(data)
    elif page == "[ENTITIES] Entity Network":
        display_entity_network(data)
    elif page == "[AUDIENCE] Audience Analysis":
        display_audience_analysis(data)
    elif page == "[GEO] Engagement Geography":
        display_engagement_geography(data)
    elif page == "[GEOGRAPHY] Follower Geography":
        display_follower_geography(data)
    elif page == "[DEMOGRAPHICS] Follower Demographics":
        display_follower_demographics(data)
    elif page == "[SEGMENTS] Follower Segmentation":
        display_follower_segmentation(data)
    elif page == "[DATA] Raw Data":
        st.header("[DATA] Raw Data")
        st.json(data)

if __name__ == "__main__":
    main()

