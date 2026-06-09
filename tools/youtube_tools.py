import os
import re
from dotenv import load_dotenv
from googleapiclient.discovery import build

load_dotenv()
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

if not YOUTUBE_API_KEY:
    raise ValueError("YOUTUBE_API_KEY is missing from the environment variables.")

youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)

def _parse_yt_duration(duration_str: str) -> int:
    match = re.match(r'PT(\d+H)?(\d+M)?(\d+S)?', duration_str)
    if not match: return 0
    h = int(match.group(1)[:-1]) if match.group(1) else 0
    m = int(match.group(2)[:-1]) if match.group(2) else 0
    s = int(match.group(3)[:-1]) if match.group(3) else 0
    return h * 3600 + m * 60 + s

def fetch_channel_metadata(handle: str) -> dict:
    clean_handle = handle.replace('@', '')
    request = youtube.search().list(q=clean_handle, type='channel', part='id,snippet', maxResults=1)
    response = request.execute()
    if not response.get('items'):
        return {"error": f"Channel {handle} not found."}
    item = response['items'][0]
    return {
        'channel_id': item['id']['channelId'],
        'channel_title': item['snippet']['title'],
        'description': item['snippet']['description']
    }

def fetch_latest_videos(channel_id: str, limit: int = 10) -> list:
    ch_request = youtube.channels().list(part='contentDetails', id=channel_id)
    ch_response = ch_request.execute()
    if not ch_response.get('items'):
        return []
    uploads_playlist_id = ch_response['items'][0]['contentDetails']['relatedPlaylists']['uploads']
    
    playlist_req = youtube.playlistItems().list(part='contentDetails', playlistId=uploads_playlist_id, maxResults=50)
    playlist_res = playlist_req.execute()
    raw_video_ids = [item['contentDetails']['videoId'] for item in playlist_res.get('items', [])]
    if not raw_video_ids:
        return []

    stats_req = youtube.videos().list(part='snippet,contentDetails,statistics', id=','.join(raw_video_ids))
    stats_res = stats_req.execute()
    
    valid_videos = []
    for item in stats_res.get('items', []):
        duration_sec = _parse_yt_duration(item['contentDetails']['duration'])
        if duration_sec > 60:
            valid_videos.append({
                'video_id': item['id'],
                'channel_id': channel_id,
                'video_title': item['snippet']['title'],
                'published_at': item['snippet']['publishedAt'],
                'view_count': int(item['statistics'].get('viewCount', 0)),
                'like_count': int(item['statistics'].get('likeCount', 0)),
                'comment_count': int(item['statistics'].get('commentCount', 0))
            })
        if len(valid_videos) == limit:
            break
    return valid_videos

def fetch_comments(video_id: str, limit: int = 20) -> list:
    comments = []
    try:
        request = youtube.commentThreads().list(part="snippet", videoId=video_id, maxResults=limit, textFormat="plainText")
        response = request.execute()
        for item in response.get('items', []):
            snippet = item['snippet']['topLevelComment']['snippet']
            comments.append({
                'comment_id': item['id'],
                'video_id': video_id,
                'author_name': snippet.get('authorDisplayName', 'Unknown'),
                'comment_text': snippet.get('textDisplay', ''),
                'like_count': int(snippet.get('likeCount', 0)),
                'published_at': snippet.get('publishedAt')
            })
    except Exception as e:
        print(f"API Error fetching comments: {e}")
    return comments

def fetch_playlist_videos(playlist_id: str, limit: int = 10) -> list:
    try:
        playlist_req = youtube.playlistItems().list(part="snippet,contentDetails", playlistId=playlist_id, maxResults=limit)
        playlist_res = playlist_req.execute()
    except Exception:
        print(f"CRITICAL: Failed to fetch playlist {playlist_id}. Error: {str(e)}")
        return []

    raw_video_ids = [item["contentDetails"]["videoId"] for item in playlist_res.get("items", [])]
    if not raw_video_ids:
        return []

    stats_req = youtube.videos().list(part="snippet,contentDetails,statistics", id=",".join(raw_video_ids))
    stats_res = stats_req.execute()
    
    valid_videos = []
    for item in stats_res.get("items", []):
        valid_videos.append({
            "video_id": item["id"],
            "channel_id": item["snippet"]["channelId"],
            "video_title": item["snippet"]["title"],
            "published_at": item["snippet"]["publishedAt"],
            "view_count": int(item["statistics"].get("viewCount", 0)),
            "like_count": int(item["statistics"].get("likeCount", 0)),
            "comment_count": int(item["statistics"].get("commentCount", 0))
        })
    return valid_videos

def fetch_single_video_details(video_id: str) -> dict:
    try:
        req = youtube.videos().list(part="snippet,statistics", id=video_id)
        res = req.execute()
        if not res.get("items"): return {}
        
        item = res["items"][0]
        return {
            "video_id": item["id"],
            "channel_id": item["snippet"]["channelId"],
            "video_title": item["snippet"]["title"],
            "channel_title": item["snippet"]["channelTitle"],
            "published_at": item["snippet"]["publishedAt"],
            "view_count": int(item["statistics"].get("viewCount", 0)),
            "like_count": int(item["statistics"].get("likeCount", 0)),
            "comment_count": int(item["statistics"].get("commentCount", 0))
        }
    except Exception as e:
        print(f"API Error fetching video details: {e}")
        return {}
    
def fetch_video_details(video_id: str) -> dict:
    try:
        req = youtube.videos().list(part="snippet,statistics", id=video_id)
        res = req.execute()
        if not res.get("items"): return {}
        
        item = res["items"][0]
        return {
            "video_id": item["id"],
            "channel_id": item["snippet"]["channelId"],
            "video_title": item["snippet"]["title"],
            "channel_title": item["snippet"]["channelTitle"],
            "published_at": item["snippet"]["publishedAt"],
            "view_count": int(item["statistics"].get("viewCount", 0)),
            "like_count": int(item["statistics"].get("likeCount", 0)),
            "comment_count": int(item["statistics"].get("commentCount", 0))
        }
    except Exception as e:
        print(f"API Error fetching video details: {e}")
        return {}