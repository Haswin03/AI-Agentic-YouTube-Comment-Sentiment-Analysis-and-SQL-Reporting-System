import os
import re
import html
import torch
import pandas as pd
import streamlit as st
import plotly.express as px
from datetime import datetime
from dotenv import load_dotenv
from googleapiclient.discovery import build
from transformers import pipeline
from sqlalchemy import create_engine, Column, String, Integer, Float, ForeignKey, DateTime, Text
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

# ==========================================
# 1. DATABASE SETUP & MODELS
# ==========================================
load_dotenv()
engine = create_engine(f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}")
Base = declarative_base()

class Channel(Base):
    __tablename__ = 'channels'
    channel_id = Column(String, primary_key=True)
    channel_title = Column(String, nullable=False)
    description = Column(Text)
    videos = relationship("Video", back_populates="channel")

class Video(Base):
    __tablename__ = 'videos'
    video_id = Column(String, primary_key=True)
    channel_id = Column(String, ForeignKey('channels.channel_id', ondelete="CASCADE"))
    video_title = Column(String, nullable=False)
    published_at = Column(DateTime)
    view_count = Column(Integer)
    duration_sec = Column(Integer)
    channel = relationship("Channel", back_populates="videos")
    comments = relationship("Comment", back_populates="video")

class Comment(Base):
    __tablename__ = 'comments'
    comment_id = Column(String, primary_key=True)
    video_id = Column(String, ForeignKey('videos.video_id', ondelete="CASCADE"))
    author_name = Column(String)
    comment_text = Column(Text, nullable=False)
    published_at = Column(DateTime)
    sentiment_label = Column(String)
    sentiment_score = Column(Float)
    video = relationship("Video", back_populates="comments")

# ==========================================
# 2. BACKEND LOGIC (THE BRAIN)
# ==========================================
def parse_yt_duration(duration_str):
    match = re.match('PT(\d+H)?(\d+M)?(\d+S)?', duration_str)
    if not match: return 0
    h = int(match.group(1)[:-1]) if match.group(1) else 0
    m = int(match.group(2)[:-1]) if match.group(2) else 0
    s = int(match.group(3)[:-1]) if match.group(3) else 0
    return h * 3600 + m * 60 + s

def clean_text(raw_text):
    if not isinstance(raw_text, str): return ""
    clean = html.unescape(raw_text)
    clean = re.sub(r'http\S+|www.\S+', '', clean)
    clean = re.sub(r'\s+', ' ', clean).strip()
    return clean[:1500]

def run_full_sync(handle):
    youtube = build('youtube', 'v3', developerKey=os.getenv("YOUTUBE_API_KEY"))
    device_id = 0 if torch.cuda.is_available() else -1
    
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        # Ensure tables exist; do NOT drop — we keep history across channels
        Base.metadata.create_all(engine)

        # 1. Resolve Channel
        username = handle.replace('@', '')
        res = youtube.search().list(q=username, type='channel', part='id,snippet', maxResults=1).execute()
        if not res.get('items'):
            return False, "Channel not found."

        ch_info = res['items'][0]
        channel_id = ch_info['id']['channelId']

        # If we've analyzed this channel before, wipe just its rows (FK cascade nukes its videos+comments)
        existing = session.query(Channel).filter_by(channel_id=channel_id).first()
        if existing:
            session.delete(existing)
            session.flush()

        chan = Channel(channel_id=channel_id, channel_title=ch_info['snippet']['title'])
        session.add(chan)

        # 2. Fetch Videos (Filtering Shorts)
        uploads_id = youtube.channels().list(part='contentDetails', id=chan.channel_id).execute()['items'][0]['contentDetails']['relatedPlaylists']['uploads']
        pl_items = youtube.playlistItems().list(part='contentDetails', playlistId=uploads_id, maxResults=50).execute()
        v_ids = [item['contentDetails']['videoId'] for item in pl_items['items']]
        
        v_req = youtube.videos().list(part='snippet,contentDetails,statistics', id=','.join(v_ids)).execute()
        
        # 3. Initialize AI (truncation=True is what stops the 512-token CUDA assert)
        MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        analyzer = pipeline(
            "sentiment-analysis",
            model=MODEL_NAME,
            tokenizer=MODEL_NAME,
            device=device_id,
            max_length=512,
            truncation=True,
        )

        count = 0
        for item in v_req['items']:
            duration = parse_yt_duration(item['contentDetails']['duration'])
            if duration <= 60:  # Skip Shorts
                continue

            v_obj = Video(
                video_id=item['id'], channel_id=chan.channel_id, video_title=item['snippet']['title'],
                published_at=datetime.strptime(item['snippet']['publishedAt'], '%Y-%m-%dT%H:%M:%SZ'),
                duration_sec=duration
            )
            session.add(v_obj)

            # 4. Fetch and Analyze Comments
            try:
                c_req = youtube.commentThreads().list(part="snippet", videoId=item['id'], maxResults=20, textFormat="plainText").execute()
            except Exception:
                c_req = {'items': []}  # Comments disabled on this video — keep going

            for c_item in c_req.get('items', []):
                snip = c_item['snippet']['topLevelComment']['snippet']
                cleaned_comment = clean_text(snip['textDisplay'])

                if not cleaned_comment:
                    continue

                # Per-comment guard: one weird Unicode payload must not nuke the batch
                try:
                    sent = analyzer(cleaned_comment)[0]
                    label = sent['label'].capitalize()
                    score = round(sent['score'], 4)
                except Exception:
                    label = 'Neutral'
                    score = 0.0

                comm_obj = Comment(
                    comment_id=c_item['id'], video_id=item['id'], author_name=snip['authorDisplayName'],
                    comment_text=cleaned_comment, published_at=datetime.strptime(snip['publishedAt'], '%Y-%m-%dT%H:%M:%SZ'),
                    sentiment_label=label, sentiment_score=score
                )
                session.add(comm_obj)

            count += 1
            if count == 10:  # Stop at exactly 10 long-form videos
                break
                    
        session.commit()
        return True, "Success"
        
    except Exception as e:
        session.rollback()
        return False, str(e)
    finally:
        session.close()

# ==========================================
# 3. STREAMLIT FRONTEND (THE FACE)
# ==========================================
st.set_page_config(page_title="YouTube Sentiment AI", page_icon="📊", layout="wide")
st.title("📊 YouTube Channel Sentiment Intelligence")
st.markdown("Automated insights powered by Hugging Face RoBERTa & PostgreSQL")

@st.cache_data
def get_channel_list():
    try:
        Base.metadata.create_all(engine)  # first-run safety so the table exists for the SELECT
        return pd.read_sql("SELECT channel_id, channel_title FROM channels ORDER BY channel_title ASC", engine)
    except Exception:
        return pd.DataFrame(columns=['channel_id', 'channel_title'])

@st.cache_data
def load_data(channel_id):
    if not channel_id:
        return None, None
    try:
        df_videos = pd.read_sql(
            "SELECT * FROM videos WHERE channel_id = %(cid)s ORDER BY published_at ASC",
            engine, params={'cid': channel_id}
        )
        df_comments = pd.read_sql(
            "SELECT c.* FROM comments c JOIN videos v ON c.video_id = v.video_id WHERE v.channel_id = %(cid)s",
            engine, params={'cid': channel_id}
        )
        if df_videos.empty or df_comments.empty:
            return None, None
        df_merged = pd.merge(df_comments, df_videos[['video_id', 'video_title', 'published_at']], on='video_id')
        return df_videos, df_merged
    except Exception:
        return None, None

# --- SIDEBAR: The Input Engine ---
selected_channel_id = None
selected_channel_title = None

with st.sidebar:
    st.header("📚 Saved Channels")
    channels_df = get_channel_list()

    if not channels_df.empty:
        selected_channel_title = st.selectbox(
            "Pick a previously analyzed channel:",
            channels_df['channel_title'].tolist(),
            key="channel_picker"
        )
        selected_channel_id = channels_df.loc[
            channels_df['channel_title'] == selected_channel_title, 'channel_id'
        ].iloc[0]
    else:
        st.info("No channels analyzed yet — add one below.")

    st.markdown("---")
    st.header("🔍 Analyze New Channel")
    target_handle = st.text_input("Enter YouTube Handle (e.g., @Google):", "@mkbhd")
    if st.button("Start Fresh Analysis"):
        with st.spinner(f"Scraping & Analyzing {target_handle}... This takes ~30 seconds."):
            success, msg = run_full_sync(target_handle)
            if success:
                st.cache_data.clear()  # invalidate channel list + dashboards
                st.success("Analysis Complete! Refreshing...")
                st.rerun()
            else:
                st.error(f"Failed: {msg}")

# --- DASHBOARD RENDERING ---
if selected_channel_title:
    st.subheader(f"Showing: {selected_channel_title}")
df_videos, df_main = load_data(selected_channel_id)

if df_main is not None and not df_main.empty:
    # Aggregation
    video_summary = df_main.groupby('video_title')['sentiment_label'].value_counts(normalize=True).unstack(fill_value=0) * 100
    video_summary = video_summary.round(2).reset_index()
    for col in ['Positive', 'Negative', 'Neutral']:
        if col not in video_summary.columns: video_summary[col] = 0.0

    channel_summary = pd.DataFrame([{
        'Total Videos Analyzed': len(df_videos),
        'Total Comments Analyzed': len(df_main),
        'Average Positive (%)': video_summary['Positive'].mean().round(2),
        'Average Negative (%)': video_summary['Negative'].mean().round(2),
        'Average Neutral (%)': video_summary['Neutral'].mean().round(2)
    }])

    # 1. Overview Section
    st.header("1. Channel-Wide Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Videos Analyzed", channel_summary['Total Videos Analyzed'][0])
    col2.metric("Overall Positivity", f"{channel_summary['Average Positive (%)'][0]}%")
    col3.metric("Overall Negativity", f"{channel_summary['Average Negative (%)'][0]}%")

    st.markdown("---")
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.subheader("Aggregate Sentiment Distribution")
        fig_pie = px.pie(df_main, names='sentiment_label', color='sentiment_label', 
                         color_discrete_map={'Positive':'#00CC96', 'Negative':'#EF553B', 'Neutral':'#636EFA'})
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_chart2:
        st.subheader("Sentiment Shift Over Time")
        trend_df = df_main.groupby('video_title', sort=False)['sentiment_label'].value_counts(normalize=True).unstack(fill_value=0) * 100
        trend_df = trend_df.reset_index()
        fig_line = px.line(trend_df, x='video_title', y=['Positive', 'Negative'], markers=True,
                           color_discrete_map={'Positive':'#00CC96', 'Negative':'#EF553B'})
        fig_line.update_layout(xaxis_title="Video Release Order", yaxis_title="Percentage (%)")
        fig_line.update_xaxes(showticklabels=False) 
        st.plotly_chart(fig_line, use_container_width=True)

    # 2. Video Analysis Section
    st.markdown("---")
    st.header("2. Video-Specific Analysis")
    selected_video = st.selectbox("Select a Video to Analyze:", df_videos['video_title'].tolist())
    vid_data = df_main[df_main['video_title'] == selected_video]

    v_col1, v_col2 = st.columns([1, 1])
    with v_col1:
        vid_pie = px.pie(vid_data, names='sentiment_label', color='sentiment_label', 
                         color_discrete_map={'Positive':'#00CC96', 'Negative':'#EF553B', 'Neutral':'#636EFA'})
        st.plotly_chart(vid_pie, use_container_width=True)

    with v_col2:
        st.subheader("Highest Confidence Insights")
        pos_comment = vid_data[vid_data['sentiment_label'] == 'Positive'].sort_values(by='sentiment_score', ascending=False)
        if not pos_comment.empty:
            st.info(f"**Top Positive:** \"{pos_comment.iloc[0]['comment_text']}\" (Score: {pos_comment.iloc[0]['sentiment_score']})")
            
        neg_comment = vid_data[vid_data['sentiment_label'] == 'Negative'].sort_values(by='sentiment_score', ascending=False)
        if not neg_comment.empty:
            st.error(f"**Top Negative:** \"{neg_comment.iloc[0]['comment_text']}\" (Score: {neg_comment.iloc[0]['sentiment_score']})")

    # 3. Export Section
    st.markdown("---")
    st.header("3. Export Reports")
    dl_col1, dl_col2 = st.columns(2)
    with dl_col1:
        st.download_button("📥 Download Video Report (CSV)", video_summary.to_csv(index=False).encode('utf-8'), "video_report.csv", "text/csv")
    with dl_col2:
        st.download_button("📥 Download Channel Report (CSV)", channel_summary.to_csv(index=False).encode('utf-8'), "channel_report.csv", "text/csv")

else:
    st.info("👈 Welcome! Enter a YouTube handle in the sidebar and click 'Start Fresh Analysis' to build the dashboard.")