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

def load_latest_data():
    """Load the most recent analyzed data file"""
    # Try analyzed files first (with AI sentiment)
    data_files = glob.glob("mark_daou_analyzed_*.json")
    if not data_files:
        # Fallback to comprehensive files
        data_files = glob.glob("mark_daou_comprehensive_*.json")
    if not data_files:
        return None
    
    latest_file = max(data_files, key=lambda x: Path(x).stat().st_mtime)
    with open(latest_file, 'r', encoding='utf-8') as f:
        return json.load(f), latest_file

def display_campaign_overview(data):
    """Display high-level campaign metrics"""
    st.header("[CAMPAIGN] Campaign Overview")
    
    # Handle both analyzed and raw data formats
    raw_data = data.get('raw_data', data)
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
    
    eng_rate = campaign.get('engagement_rate', 0)
    col3.metric(
        "Engagement Rate",
        f"{eng_rate:.1f}",
        help="Average engagement per tweet"
    )
    
    col4.metric(
        "Influence Score",
        f"{campaign.get('influence_score', 0):.0f}",
        help="Calculated influence metric"
    )
    
    # Calculate reach/following ratio
    ratio = pm.get('followers_count', 0) / max(pm.get('following_count', 1), 1)
    col5.metric(
        "Reach Ratio",
        f"{ratio:.1f}x",
        help="Followers to following ratio"
    )

def display_sentiment_analysis(data):
    """Display comprehensive sentiment analysis"""
    st.header("[AI] AI-Powered Sentiment Analysis")
    
    # Handle analyzed data format with AI sentiment
    analysis = data.get('analysis', {})
    sentiment_data = analysis.get('sentiment_summary', {})
    
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
        st.markdown("**Analyzed with**: BERT (multilingual) + VADER + TextBlob")
        sent = sentiment_data.get('own_tweets', {})
        if sent and sent.get('total', 0) > 0:
            display_sentiment_chart(sent, "Own Tweets")
            display_sentiment_breakdown(sent)
            
            # Show AI model info
            st.info("**AI Models Used**: Multilingual BERT for Arabic/English + VADER (social media optimized) + TextBlob (polarity)")
        else:
            st.info("No tweet sentiment data")
    
    with tab2:
        st.subheader("AI Sentiment in Mentions")
        st.markdown("**Analyzed with**: BERT (multilingual) + VADER + TextBlob")
        sent = sentiment_data.get('mentions', {})
        if sent and sent.get('total', 0) > 0:
            display_sentiment_chart(sent, "Mentions")
            display_sentiment_breakdown(sent)
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
    
    # Get analyzed tweets with AI sentiment
    analysis = data.get('analysis', {})
    analyzed_tweets = analysis.get('analyzed_tweets', [])
    
    eng_data = data.get('engagement_metrics', {})
    
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
    
    # Get AI-analyzed hashtag sentiment
    analysis = data.get('analysis', {})
    hashtag_data = analysis.get('hashtag_sentiment', {})
    
    if not hashtag_data:
        # Fallback to old format
        hashtag_data = data.get('hashtag_analysis', {})
    
    if not hashtag_data:
        st.info("No hashtag data available")
        return
    
    # Convert to DataFrame with sentiment
    hashtag_list = []
    for tag, data_dict in hashtag_data.items():
        hashtag_list.append({
            'Hashtag': f"#{tag}",
            'Count': data_dict.get('count', 0),
            'Positive': data_dict.get('positive', 0),
            'Negative': data_dict.get('negative', 0),
            'Neutral': data_dict.get('neutral', 0),
            'Avg_Engagement': data_dict.get('avg_engagement', 0)
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
    
    # Get AI-generated topics
    analysis = data.get('analysis', {})
    topic_data = analysis.get('topics', {})
    
    if not topic_data:
        # Fallback
        topic_data = data.get('topic_analysis', {})
    
    if not topic_data:
        st.info("No topic data available. Topics are extracted using machine learning from tweet text.")
        return
    
    st.markdown("**Method**: Latent Dirichlet Allocation (LDA) with TF-IDF vectorization")
    st.markdown(f"**Topics Identified**: {len(topic_data)}")
    
    # Display topics as expandable sections
    for topic_name, topic_info in topic_data.items():
        with st.expander(f"[TOPIC] {topic_name}", expanded=True):
            keywords = topic_info.get('keywords', [])
            all_keywords = topic_info.get('all_keywords', keywords)
            
            # Top 5 keywords
            st.markdown("**Key Terms**: " + " | ".join(f"`{kw}`" for kw in keywords))
            
            # All keywords
            if len(all_keywords) > len(keywords):
                st.markdown("**Related Terms**: " + ", ".join(all_keywords[5:]))

def display_posting_patterns(data):
    """Display posting pattern analysis"""
    st.header("[TIME] Posting Patterns")
    
    patterns = data.get('posting_patterns', {})
    
    if not patterns:
        st.info("No posting pattern data available")
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
    
    # Get AI audience analysis
    analysis = data.get('analysis', {})
    audience_data = analysis.get('audience_engagement', {})
    
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
    raw_data = data.get('raw_data', data)
    
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
            st.info("No follower data available")
    
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
            st.info("No following data available")

def display_public_opinion(data):
    """Display public opinion dashboard with AI sentiment"""
    st.header("[OPINION] Public Opinion Tracker (AI-Analyzed)")
    
    # Get AI sentiment summary
    analysis = data.get('analysis', {})
    sentiment_summary = analysis.get('sentiment_summary', {})
    
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

def main():
    # Title
    st.title("[LEBANON] Mark Daou - Political Intelligence Dashboard")
    st.markdown("**Advanced Sentiment Analysis, Campaign Tracking & Public Opinion Monitoring**")
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
    
    # Navigation
    st.sidebar.markdown("---")
    st.sidebar.header("Navigation")
    
    page = st.sidebar.radio(
        "Select View:",
        [
            "[CAMPAIGN] Campaign Overview",
            "[AI] Sentiment Analysis",
            "[OPINION] Public Opinion",
            "[STATS] Engagement Metrics",
            "[HASHTAGS] Hashtag Analysis",
            "[TOPICS] Topic Analysis",
            "[TIME] Posting Patterns",
            "[AUDIENCE] Audience Analysis",
            "[DATA] Raw Data"
        ]
    )
    
    # Display selected page
    if page == "[CAMPAIGN] Campaign Overview":
        display_campaign_overview(data)
    elif page == "[AI] Sentiment Analysis":
        display_sentiment_analysis(data)
    elif page == "[OPINION] Public Opinion":
        display_public_opinion(data)
    elif page == "[STATS] Engagement Metrics":
        display_engagement_metrics(data)
    elif page == "[HASHTAGS] Hashtag Analysis":
        display_hashtag_analysis(data)
    elif page == "[TOPICS] Topic Analysis":
        display_topic_analysis(data)
    elif page == "[TIME] Posting Patterns":
        display_posting_patterns(data)
    elif page == "[AUDIENCE] Audience Analysis":
        display_audience_analysis(data)
    elif page == "[DATA] Raw Data":
        st.header("[DATA] Raw Data")
        st.json(data)

if __name__ == "__main__":
    main()

