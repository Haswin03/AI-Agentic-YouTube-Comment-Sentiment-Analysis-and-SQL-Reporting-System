import re
from typing import Optional

def extract_video_id(url: str) -> Optional[str]:
    if not url: return None
    
    # Regex looks for 'v=', 'youtu.be/', or 'embed/', then captures exactly 11 characters
    pattern = r"(?:v=|\/vi\/|youtu\.be\/|\/v\/|\/embed\/)([\w-]{11})"
    match = re.search(pattern, url)
    
    if match:
        return match.group(1)
    
    # If the user just pasted the 11-character ID directly by mistake, accept it
    if len(url.strip()) == 11 and re.match(r"^[\w-]+$", url.strip()):
        return url.strip()
        
    return None

def extract_playlist_id(url: str) -> Optional[str]:
    if not url: return None
    
    pattern = r"(?:list=)([\w-]+)"
    match = re.search(pattern, url)
    
    if match:
        return match.group(1)
    return None

def extract_channel_handle(url: str) -> Optional[str]:
    if not url: return None
    
    handle_match = re.search(r"(@[\w.-]+)", url)
    if handle_match:
        return handle_match.group(1)
        
    channel_id_match = re.search(r"channel\/([\w-]+)", url)
    if channel_id_match:
        return channel_id_match.group(1)
        
    custom_url_match = re.search(r"c\/([\w.-]+)", url)
    if custom_url_match:
        return custom_url_match.group(1)
        
    return None

def normalize_channel_name(name: str) -> Optional[str]:
    if not name: return None
    
    name = name.strip()
    if " " not in name and not name.startswith('@'):
        return f"@{name}"
        
    return name

def route_input(channel_name: str, video_link: str, channel_link: str, playlist_link: str) -> dict:
    # 1: Single Video
    if video_link:
        video_id = extract_video_id(video_link)
        if video_id:
            return {"action": "process_video", "value": video_id}
        else:
            return {"error": "Invalid Video URL."}
            
    # 2: Playlist
    elif playlist_link:
        playlist_id = extract_playlist_id(playlist_link)
        if playlist_id:
            return {"action": "process_playlist", "value": playlist_id}
        else:
            return {"error": "Invalid Playlist URL."}
            
    # 3: Channel Link
    elif channel_link:
        handle = extract_channel_handle(channel_link)
        if handle:
            return {"action": "process_channel", "value": handle}
        else:
            return {"error": "Invalid Channel URL."}
            
    # 4: Channel Name / Handle Search
    elif channel_name:
        clean_name = normalize_channel_name(channel_name)
        return {"action": "process_channel_search", "value": clean_name}
        
    # Failsafe
    return {"error": "No valid input provided. Please fill out one of the fields."}