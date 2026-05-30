import os
from dotenv import load_dotenv
from smolagents import CodeAgent, InferenceClientModel, Tool

from tools.youtube_tools import fetch_channel_metadata, fetch_latest_videos, fetch_comments, fetch_playlist_videos, fetch_single_video_details, fetch_video_details
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
    
class FetchVideoDetailsTool(Tool):
    name = 'fetch_video_details'
    description = 'Fetches full metadata (title, channel, stats) for a single YouTube video ID.'
    inputs = {'video_id': {'type': 'string', 'description': 'The YouTube video ID.'}}
    output_type = 'any'
    def forward(self, video_id: str):
        return fetch_single_video_details(video_id)
    
class FetchVideoDetailsTool(Tool):
    name = 'fetch_video_details'
    description = 'Fetches full metadata (title, channel name, stats) for a single YouTube video ID.'
    inputs = {'video_id': {'type': 'string', 'description': 'The YouTube video ID.'}}
    output_type = 'any'
    def forward(self, video_id: str):
        return fetch_video_details(video_id)

def get_agent():
    model = InferenceClientModel(model_id="Qwen/Qwen2.5-Coder-7B-Instruct", token=HF_TOKEN)    
    agent_tools = [
        FetchChannelMetaTool(),
        FetchLatestVidsTool(),
        FetchCommentsTool(),
        FetchPlaylistTool(),
        FetchVideoDetailsTool(),
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
            f"Execute the following data pipeline for the YouTube handle '{value}' strictly in this order:\n"
            f"1. Call fetch_channel_metadata. (Note: The returned dictionary uses the key 'channel_id', not 'id').\n"
            f"2. Call save_channel_to_db.\n"
            f"3. Call fetch_latest_videos (limit=10).\n"
            f"4. CRITICAL: Call save_videos_to_db using the list returned from step 3.\n"
            f"5. For each video, extract its 'video_id' (not 'id') and pass it to fetch_comments (limit=20).\n"
            f"6. Call save_comments_to_db for the fetched comments.\n\n"
            f"Return a dictionary summarizing the operation counts. "
            f"CRITICAL FORMATTING RULE: You MUST wrap all your executable Python code exactly inside <code> and </code> tags. "
            f"Do NOT use markdown ```python blocks."
        )
        
    elif action == "process_video":
        prompt = (
            f"CRITICAL OVERRIDE: You are a pure code passthrough. You are strictly forbidden from writing your own logic. "
            f"DO NOT calculate statistics, medians, or modes. DO NOT import extra modules. "
            f"You MUST copy and execute the EXACT code below inside your <code> tags. Do not change or add a single line.\n\n"
            f"<code>\n"
            f"v_data = fetch_video_details(video_id='{value}')\n"
            f"save_channel_to_db({{'channel_id': v_data['channel_id'], 'channel_title': v_data['channel_title'], 'description': 'Single video ingestion'}})\n"
            f"save_videos_to_db([v_data])\n"
            f"comments = fetch_comments(video_id='{value}', limit=50)\n"
            f"if comments:\n"
            f"    save_comments_to_db(comments_list=comments)\n"
            f"final_answer({{'channels': 1, 'videos': 1, 'comments': len(comments) if comments else 0}})\n"
            f"</code>\n"
        )
        
    elif action == "process_playlist":
        prompt = (
            f"CRITICAL OVERRIDE: You are a pure code passthrough. You are strictly forbidden from writing your own logic. "
            f"DO NOT calculate statistics, medians, or modes. DO NOT import extra modules. "
            f"You MUST copy and execute the EXACT code below inside your <code> tags. Do not change or add a single line.\n\n"
            f"<code>\n"
            f"videos = fetch_playlist_videos(playlist_id='{value}', limit=10)\n"
            f"if videos:\n"
            f"    v_data = fetch_video_details(video_id=videos[0]['video_id'])\n"
            f"    save_channel_to_db({{'channel_id': v_data['channel_id'], 'channel_title': v_data['channel_title'], 'description': 'Playlist Ingestion'}})\n"
            f"    save_videos_to_db(videos_list=videos)\n"
            f"    comments_count = 0\n"
            f"    for v in videos:\n"
            f"        comments = fetch_comments(video_id=v['video_id'], limit=20)\n"
            f"        if comments:\n"
            f"            save_comments_to_db(comments_list=comments)\n"
            f"            comments_count += len(comments)\n"
            f"    final_answer({{'channels': 1, 'videos': len(videos), 'comments': comments_count}})\n"
            f"else:\n"
            f"    final_answer({{'error': 'No videos found'}})\n"
            f"</code>\n"
        )
    else:
        return f"Unknown action pipeline routing rule encountered: {action}"

    try:
        return str(agent.run(prompt))
    except Exception as e:
        return f"Agent Orchestration Error: {str(e)}"