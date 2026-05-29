import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(DATABASE_URL)

def ensure_export_dir():
    export_dir = os.path.join(os.getcwd(), "exports")
    if not os.path.exists(export_dir):
        os.makedirs(export_dir)
    return export_dir

def generate_videos_summary():
    query = "SELECT video_id, channel_id, video_title, published_at, view_count, like_count, comment_count FROM videos;"
    df = pd.read_sql(query, engine)
    
    if df.empty:
        return "No video data to export."
        
    export_path = os.path.join(ensure_export_dir(), "videos_summary.csv")
    df.to_csv(export_path, index=False)
    return f"Saved Videos Summary: {export_path}"

def generate_comments_details():
    query = """
        SELECT comment_id, video_id, author_name, comment_text, 
               like_count, sentiment_label, sentiment_score
        FROM comments
        ORDER BY video_id, sentiment_score DESC;
    """
    df = pd.read_sql(query, engine)
    
    if df.empty:
        return "No comment data to export."
        
    export_path = os.path.join(ensure_export_dir(), "comments_sentiment_details.csv")
    df.to_csv(export_path, index=False)
    return f"Saved Comments Details: {export_path}"

def generate_channel_sentiment_report():
    query = text("""
        SELECT 
            v.video_title,
            COUNT(c.comment_id) as total_analyzed_comments,
            ROUND(SUM(CASE WHEN c.sentiment_label = 'Positive' THEN 1 ELSE 0 END) * 100.0 / NULLIF(COUNT(c.comment_id), 0), 2) as positive_percentage,
            ROUND(SUM(CASE WHEN c.sentiment_label = 'Neutral' THEN 1 ELSE 0 END) * 100.0 / NULLIF(COUNT(c.comment_id), 0), 2) as neutral_percentage,
            ROUND(SUM(CASE WHEN c.sentiment_label = 'Negative' THEN 1 ELSE 0 END) * 100.0 / NULLIF(COUNT(c.comment_id), 0), 2) as negative_percentage,
            ROUND(CAST(AVG(c.sentiment_score) AS NUMERIC), 4) as avg_confidence_score
        FROM videos v
        LEFT JOIN comments c ON v.video_id = c.video_id
        GROUP BY v.video_id, v.video_title
        ORDER BY positive_percentage DESC;
    """)
    
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
        
    if df.empty or df['total_analyzed_comments'].sum() == 0:
        return "No sentiment data available to aggregate."
        
    export_path = os.path.join(ensure_export_dir(), "channel_sentiment_report.csv")
    df.to_csv(export_path, index=False)
    return f"Saved Channel Sentiment Report: {export_path}"

def generate_all_reports():
    """Master function to trigger all exports at once."""
    print(generate_videos_summary())
    print(generate_comments_details())
    print(generate_channel_sentiment_report())
    print("All reports successfully exported to the /exports folder!")

if __name__ == "__main__":
    generate_all_reports()