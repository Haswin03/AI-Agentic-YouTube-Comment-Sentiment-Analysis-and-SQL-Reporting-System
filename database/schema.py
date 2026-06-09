import os
from sqlalchemy import create_engine, Column, String, Integer, DateTime, Float, ForeignKey, Text
from sqlalchemy.orm import declarative_base, relationship

DATABASE_URL = "sqlite:///youtube_sentiment.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
Base = declarative_base()

class Channel(Base):
    __tablename__ = 'channels'
    
    channel_id = Column(String(100), primary_key=True)
    channel_title = Column(String(255), nullable=False)
    description = Column(String(2000))
    
    videos = relationship("Video", back_populates="channel", cascade="all, delete-orphan")

class Video(Base):
    __tablename__ = 'videos'
    
    video_id = Column(String(50), primary_key=True)
    channel_id = Column(String(100), ForeignKey('channels.channel_id', ondelete="CASCADE"), nullable=False)
    video_title = Column(String(255), nullable=False)
    published_at = Column(DateTime)
    view_count = Column(Integer, default=0)
    like_count = Column(Integer, default=0)
    comment_count = Column(Integer, default=0)
    
    channel = relationship("Channel", back_populates="videos")
    comments = relationship("Comment", back_populates="video", cascade="all, delete-orphan")

class Comment(Base):
    __tablename__ = 'comments'
    
    comment_id = Column(String(100), primary_key=True)
    video_id = Column(String(50), ForeignKey('videos.video_id', ondelete="CASCADE"), nullable=False)
    author_name = Column(String(255))
    comment_text = Column(Text, nullable=False)
    like_count = Column(Integer, default=0)
    published_at = Column(DateTime)
    sentiment_label = Column(String(20), nullable=True)
    sentiment_score = Column(Float, nullable=True)
    
    video = relationship("Video", back_populates="comments")

if __name__ == "__main__":
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    print("New tables and indices created successfully.")