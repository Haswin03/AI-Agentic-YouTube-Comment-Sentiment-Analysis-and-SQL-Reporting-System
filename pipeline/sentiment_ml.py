import os
import re
import html
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from smolagents import tool

load_dotenv()
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(DATABASE_URL)

_sentiment_analyzer = None

def _get_analyzer():
    global _sentiment_analyzer
    if _sentiment_analyzer is None:
        import torch
        from transformers import pipeline
        
        device_id = 0 if torch.cuda.is_available() else -1
        MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        
        _sentiment_analyzer = pipeline(
            "sentiment-analysis", 
            model=MODEL_NAME, 
            tokenizer=MODEL_NAME, 
            device=device_id,
            max_length=512,      
            truncation=True
        )
    return _sentiment_analyzer

def clean_youtube_text(raw_text: str) -> str:
    if not isinstance(raw_text, str) or not raw_text.strip():
        return ""
    clean_text = html.unescape(raw_text)
    clean_text = re.sub(r'http\S+|www.\S+', '', clean_text)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    return clean_text[:1500]

@tool
def analyze_unscored_comments(batch_limit: int = 100) -> str:
    """Scans the database for comments without sentiment scores and analyzes them.
    
    Args:
        batch_limit: Maximum number of comments to process in this run.
    """
    analyzer = _get_analyzer()
    label_map = {
        'positive': 'Positive',
        'neutral': 'Neutral',
        'negative': 'Negative'
    }

    try:
        with engine.connect() as read_conn:
            fetch_query = text("SELECT comment_id, comment_text FROM comments WHERE sentiment_label IS NULL LIMIT :limit")
            result = read_conn.execute(fetch_query, {"limit": batch_limit}).mappings().all()
            
        if not result:
            return "No unscored comments found in the database. Everything is up to date."
            
        updates = []
        for row in result:
            raw_text = row['comment_text']
            cleaned_text = clean_youtube_text(raw_text)
            
            if not cleaned_text:
                updates.append({
                    "id": row['comment_id'], 
                    "label": "Neutral", 
                    "score": 0.0
                })
                continue
            
            try:
                ml_result = analyzer(cleaned_text)[0]
                raw_label = ml_result['label'].lower()
                
                updates.append({
                    "id": row['comment_id'],
                    "label": label_map.get(raw_label, 'Neutral'),
                    "score": round(ml_result['score'], 4)
                })
            except Exception:
                updates.append({"id": row['comment_id'], "label": "Neutral", "score": 0.0})

        with engine.begin() as write_conn:
            update_query = text("UPDATE comments SET sentiment_label = :label, sentiment_score = :score WHERE comment_id = :id")
            write_conn.execute(update_query, updates)
            
        return f"Successfully analyzed and updated {len(updates)} comments in the database."

    except Exception as e:
        return f"Database or execution error during sentiment analysis: {str(e)}"