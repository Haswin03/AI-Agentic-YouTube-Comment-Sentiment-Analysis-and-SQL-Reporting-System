import os
import datetime
import streamlit as st
import pandas as pd
import plotly.express as px
from sqlalchemy import create_engine, text
from database.schema import Base

from tools.input_router import route_input
from pipeline.reporter import generate_all_reports, ensure_export_dir

DATABASE_URL = "sqlite:///youtube_sentiment.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
Base.metadata.create_all(engine)

st.set_page_config(page_title="Agentic YouTube Sentiment Dashboard", layout="wide")

st.title("Agentic YouTube Comment Sentiment Analysis")
st.write("Welcome to the Final Project Dashboard. This tool uses a Hugging Face Agent to autonomously scrape YouTube data, store it in a PostgreSQL database, and run RoBERTa-based NLP sentiment analysis on user comments.")
st.divider()

st.subheader("1. Data Ingestion Hub")
st.write("Fill out ONE of the fields below to initiate the Agentic scraping pipeline.")

col1, col2 = st.columns(2)
with col1:
    channel_name = st.text_input("YouTube Channel Name", placeholder="@mkbhd")
    video_link = st.text_input("Single Video URL", placeholder="https://youtube.com/watch?v=id")
with col2:
    channel_link = st.text_input("Channel URL", placeholder="https://youtube.com/@mkbhd")
    playlist_link = st.text_input("Playlist URL", placeholder="https://youtube.com/playlist?list=id")

if st.button("Run Agentic Analysis", type="primary"):
    from pipeline.agent_orchestrator import run_agentic_pipeline
    
    route_data = route_input(channel_name, video_link, channel_link, playlist_link)
    if "error" in route_data:
        st.error(route_data["error"])
    else:
        st.info(f"Agent Instruction Generated: {route_data['action']} on target {route_data['value']}")
        with st.spinner("Executing agentic pipeline"):
            agent_result = run_agentic_pipeline(route_data)
        st.success("Agent Execution Cycle Concluded")
        with st.expander("View Agent Execution Logs", expanded=False):
            st.write(agent_result)

st.divider()

st.subheader("2. NLP Sentiment Processing Engine")
st.write("Process unscored comments in the database using the RoBERTa Machine Learning model.")

if st.button("Run Sentiment Analysis Model"):
    from pipeline.sentiment_ml import analyze_unscored_comments
    
    with st.spinner("Executing sentiment analysis"):
        ml_result = analyze_unscored_comments(batch_limit=500)
        st.success(ml_result)

st.divider()

st.subheader("3. Database Verification and Analytics")
st.write("Verify your ingested data and filter analytics by specific targets.")

try:
    with engine.connect() as conn:
        channels_query = text("SELECT channel_id, channel_title FROM channels;")
        channels_df = pd.read_sql(channels_query, conn)
        
        verification_query = text("SELECT v.video_title as Video, ch.channel_title as Channel, COUNT(c.comment_id) as Comments_Extracted FROM videos v LEFT JOIN comments c ON v.video_id = c.video_id LEFT JOIN channels ch ON v.channel_id = ch.channel_id GROUP BY v.video_title, ch.channel_title ORDER BY Comments_Extracted DESC;")
        verification_df = pd.read_sql(verification_query, conn)
        
        with st.expander("Data Inspector: See exactly what is in your database", expanded=False):
            if not verification_df.empty:
                st.dataframe(verification_df, use_container_width=True)
            else:
                st.write("Database is currently empty.")

        if not channels_df.empty:
            channel_options = ["All Data (Global)"] + channels_df['channel_title'].tolist()
            selected_filter = st.selectbox("Filter Analytics By Channel:", channel_options)
            
            if selected_filter == "All Data (Global)":
                sentiment_query = text("SELECT sentiment_label, COUNT(*) as count FROM comments WHERE sentiment_label IS NOT NULL GROUP BY sentiment_label;")
                chart_df = pd.read_sql(sentiment_query, conn)
            else:
                sentiment_query = text("SELECT c.sentiment_label, COUNT(*) as count FROM comments c JOIN videos v ON c.video_id = v.video_id JOIN channels ch ON v.channel_id = ch.channel_id WHERE c.sentiment_label IS NOT NULL AND ch.channel_title = :ch_title GROUP BY c.sentiment_label;")
                chart_df = pd.read_sql(sentiment_query, conn, params={"ch_title": selected_filter})

            if not chart_df.empty:
                dash_col1, dash_col2 = st.columns([1, 2])
                with dash_col1:
                    st.write("Key Summary Metrics")
                    total_comments = chart_df['count'].sum()
                    pos_row = chart_df[chart_df['sentiment_label'] == 'Positive']
                    neg_row = chart_df[chart_df['sentiment_label'] == 'Negative']
                    pos_count = pos_row['count'].values[0] if not pos_row.empty else 0
                    neg_count = neg_row['count'].values[0] if not neg_row.empty else 0
                    pos_pct = round((pos_count / total_comments) * 100, 1) if total_comments > 0 else 0
                    neg_pct = round((neg_count / total_comments) * 100, 1) if total_comments > 0 else 0
                    
                    st.metric(label="Total Scored Comments", value=f"{total_comments:,}")
                    st.metric(label="Positive Sentiment", value=f"{pos_pct}%", delta=f"{pos_count} comments")
                    st.metric(label="Negative Sentiment", value=f"{neg_pct}%", delta=f"-{neg_count} comments", delta_color="inverse")
                with dash_col2:
                    st.write(f"Sentiment Distribution for {selected_filter}")
                    fig = px.pie(chart_df, values='count', names='sentiment_label', color='sentiment_label', color_discrete_map={'Positive': '#2ecc71', 'Neutral': '#f1c40f', 'Negative': '#e74c3c'}, hole=0.45)
                    fig.update_traces(textposition='inside', textinfo='percent+label')
                    fig.update_layout(margin=dict(t=30, b=30, l=30, r=30), showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"No scored comments found for {selected_filter}. Run Step 2.")
        else:
            st.info("No channels found. Run the Agent in Step 1 to populate data.")
            
except Exception as db_err:
    st.error(f"Could not render analytics panel visualizers: {db_err}")

st.divider()

st.subheader("4. Export Compiled Datasets")

if st.button("Compile and Write Data to Files"):
    with st.spinner("Compiling exports"):
        generate_all_reports()
        st.success("Reports written to the workspace storage folder safely")

export_dir = ensure_export_dir()
col_a, col_b, col_c = st.columns(3)

time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M")

report_path = os.path.join(export_dir, "channel_sentiment_report.csv")
if os.path.exists(report_path):
    with open(report_path, "rb") as file:
        col_a.download_button(label="Download Channel Aggregations", data=file, file_name=f"channels_{time_str}.csv", mime="text/csv")

video_path = os.path.join(export_dir, "videos_summary.csv")
if os.path.exists(video_path):
    with open(video_path, "rb") as file:
        col_b.download_button(label="Download Video Metadata", data=file, file_name=f"videos_{time_str}.csv", mime="text/csv")

comments_path = os.path.join(export_dir, "comments_sentiment_details.csv")
if os.path.exists(comments_path):
    with open(comments_path, "rb") as file:
        col_c.download_button(label="Download Raw Annotated Comments", data=file, file_name=f"comments_{time_str}.csv", mime="text/csv")