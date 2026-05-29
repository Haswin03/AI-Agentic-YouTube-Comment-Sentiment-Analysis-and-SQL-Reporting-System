import os
from datetime import datetime
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from smolagents import tool
from database.schema import Channel, Video, Comment

load_dotenv()
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)

def _parse_iso_date(date_str: str):
    if not date_str: return None
    clean_str = date_str.replace('Z', '+00:00')
    try:
        return datetime.fromisoformat(clean_str)
    except Exception:
        return None

@tool
def save_channel_to_db(channel_data: dict) -> str:
    """Saves or updates YouTube channel metadata in the database.

    Args:
        channel_data: A dictionary containing channel details.
    """
    if "error" in channel_data:
        return f"Skipped: {channel_data['error']}"
    session = SessionLocal()
    try:
        channel_record = Channel(
            channel_id=channel_data['channel_id'],
            channel_title=channel_data.get('channel_title', 'Unknown'),
            description=channel_data.get('description', '')
        )
        session.merge(channel_record)
        session.commit()
        return f"Saved channel: {channel_data['channel_title']}"
    except Exception as e:
        session.rollback()
        return f"Error: {str(e)}"
    finally:
        session.close()

@tool
def save_videos_to_db(videos_data: list) -> str:
    """Saves a list of video metadata dictionaries to the database.

    Args:
        videos_data: A list of dictionaries containing video details.
    """
    if not videos_data: return "No video data."
    session = SessionLocal()
    try:
        for v in videos_data:
            video_record = Video(
                video_id=v['video_id'],
                channel_id=v['channel_id'],
                video_title=v['video_title'],
                published_at=_parse_iso_date(v.get('published_at')),
                view_count=v.get('view_count', 0),
                like_count=v.get('like_count', 0),
                comment_count=v.get('comment_count', 0)
            )
            session.merge(video_record)
        session.commit()
        return f"Saved {len(videos_data)} videos."
    except Exception as e:
        session.rollback()
        return f"Error: {str(e)}"
    finally:
        session.close()

@tool
def save_comments_to_db(comments_data: list) -> str:
    """Saves a list of video comments to the database.

    Args:
        comments_data: A list of comments dictionaries.
    """
    if not comments_data: return "No comment data."
    session = SessionLocal()
    try:
        for c in comments_data:
            comment_record = Comment(
                comment_id=c['comment_id'],
                video_id=c['video_id'],
                author_name=c.get('author_name', 'Unknown'),
                comment_text=c.get('comment_text', ''),
                like_count=c.get('like_count', 0),
                published_at=_parse_iso_date(c.get('published_at'))
            )
            session.merge(comment_record)
        session.commit()
        return f"Saved {len(comments_data)} comments."
    except Exception as e:
        session.rollback()
        return f"Error: {str(e)}"
    finally:
        session.close()