import os
from dotenv import load_dotenv
from smolagents import CodeAgent, InferenceClientModel, Tool

from tools.youtube_tools import fetch_channel_metadata, fetch_latest_videos, fetch_comments, fetch_playlist_videos
from tools.db_tools import save_channel_to_db, save_videos_to_db, save_comments_to_db

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

class FetchChannelMetaTool(Tool):
    name = 'fetch_channel_metadata'
    description = 'Resolves a YouTube channel handle or name to its unique metadata dictionary.'
    inputs = {'handle': {'type': 'string', 'description': 'The YouTube channel handle string.'}}
    output_type = 'any'
    def forward(self, handle: str):
        return fetch_channel_metadata(handle)

class FetchLatestVidsTool(Tool):
    name = 'fetch_latest_videos'
    description = 'Fetches the most recent standard videos list for a channel.'
    inputs = {
        'channel_id': {'type': 'string', 'description': 'The unique YouTube Channel ID string.'},
        'limit': {'type': 'integer', 'description': 'The number of videos to return.', 'nullable': True}
    }
    output_type = 'any'
    def forward(self, channel_id: str, limit: int = 10):
        return fetch_latest_videos(channel_id, limit)

class FetchCommentsTool(Tool):
    name = 'fetch_comments'
    description = 'Extracts top-level comments list for a specific video ID.'
    inputs = {
        'video_id': {'type': 'string', 'description': 'The 11-character video ID string.'},
        'limit': {'type': 'integer', 'description': 'The maximum number of comments to fetch.', 'nullable': True}
    }
    output_type = 'any'
    def forward(self, video_id: str, limit: int = 20):
        return fetch_comments(video_id, limit)

class FetchPlaylistTool(Tool):
    name = 'fetch_playlist_videos'
    description = 'Extracts video metadata records list from a generic YouTube playlist ID.'
    inputs = {
        'playlist_id': {'type': 'string', 'description': 'The unique string identifier of the YouTube playlist.'},
        'limit': {'type': 'integer', 'description': 'Maximum number of video records to parse from the playlist.', 'nullable': True}
    }
    output_type = 'any'
    def forward(self, playlist_id: str, limit: int = 10):
        return fetch_playlist_videos(playlist_id, limit)

class SaveChannelTool(Tool):
    name = 'save_channel_to_db'
    description = 'Saves or updates YouTube channel metadata record in the database.'
    inputs = {'channel_data': {'type': 'any', 'description': 'The dictionary containing channel info.'}}
    output_type = 'any'
    def forward(self, channel_data: dict):
        return save_channel_to_db(channel_data)

class SaveVideosTool(Tool):
    name = 'save_videos_to_db'
    description = 'Saves a list of video metadata dictionaries to the database.'
    inputs = {'videos_list': {'type': 'any', 'description': 'The list containing video records.'}}
    output_type = 'any'
    def forward(self, videos_list: list):
        return save_videos_to_db(videos_list)

class SaveCommentsTool(Tool):
    name = 'save_comments_to_db'
    description = 'Saves a list of video comments dictionaries to the database.'
    inputs = {'comments_list': {'type': 'any', 'description': 'The list containing comment records.'}}
    output_type = 'any'
    def forward(self, comments_list: list):
        return save_comments_to_db(comments_list)

def get_agent():
    model = InferenceClientModel(model_id="Qwen/Qwen2.5-Coder-7B-Instruct", token=HF_TOKEN)    
    agent_tools = [
        FetchChannelMetaTool(),
        FetchLatestVidsTool(),
        FetchCommentsTool(),
        FetchPlaylistTool(),
        SaveChannelTool(),
        SaveVideosTool(),
        SaveCommentsTool()
    ]
    
    return CodeAgent(tools=agent_tools, model=model, additional_authorized_imports=['time', 'json'], max_steps=12)

def run_agentic_pipeline(route_data: dict) -> str:
    if "error" in route_data:
        return f"Pipeline Halted: {route_data['error']}"

    action = route_data.get("action")
    value = route_data.get("value")
    
    agent = get_agent()

    if action in ["process_channel", "process_channel_search"]:
        prompt = (
            f"Process channel data operations for target handle '{value}' across tools in order: "
            f"fetch_channel_metadata, save_channel_to_db, then fetch_latest_videos with limit=10, "
            f"save_videos_to_db, then fetch_comments for each video with limit=20, and save_comments_to_db. "
            f"Return a string operation count summary."
        )
        
    elif action == "process_video":
        prompt = (
            f"Process a standalone video link for target ID '{value}' across tools in order: "
            f"call save_channel_to_db with a channel dictionary built using channel_id='UNKNOWN' and channel_title='Standalone Video Ingestion', "
            f"call save_videos_to_db with a list containing a single video dictionary for video_id='{value}' and channel_id='UNKNOWN', "
            f"call fetch_comments for video_id='{value}' with limit=50, and pass the results to save_comments_to_db. "
            f"Return a string confirmation."
        )
        
    elif action == "process_playlist":
        prompt = (
            f"Process a playlist link for target ID '{value}' across tools in order: "
            f"call save_channel_to_db with a channel dictionary built using channel_id='PLAYLIST_DATA' and channel_title='Playlist Ingestion Stream', "
            f"call fetch_playlist_videos for playlist_id='{value}' with limit=10, overwrite each video channel_id to 'PLAYLIST_DATA', "
            f"pass the videos list to save_videos_to_db, then call fetch_comments for each video with limit=20, and pass results to save_comments_to_db. "
            f"Return a string compilation count summary."
        )
    else:
        return f"Unknown action pipeline routing rule encountered: {action}"

    try:
        return str(agent.run(prompt))
    except Exception as e:
        return f"Agent Orchestration Error: {str(e)}"