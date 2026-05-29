# AI Agentic YouTube Comment Sentiment Analysis System

**🔴 Live Streamlit Dashboard:** [Insert Your Streamlit Cloud Link Here]

## 📖 Project Overview
This project is an advanced, fully autonomous **Agentic ETL (Extract, Transform, Load) Pipeline** designed to ingest YouTube data, ensure relational database integrity, and perform deep-learning sentiment analysis. 

Instead of relying on hardcoded scraping scripts, this architecture utilizes a **Hugging Face CodeAgent** powered by the `Qwen2.5-Coder` Large Language Model. The Agent acts as an orchestrator, dynamically selecting the appropriate Python tools to extract data from YouTube Channels, Playlists, or Single Videos, safely loading them into a PostgreSQL database, and triggering a RoBERTa-based Machine Learning model to evaluate public sentiment.

## ✨ Core Features
* **Agentic Orchestration:** The system dynamically routes instructions based on user input, chaining tools together to handle complex multi-step data extractions autonomously.
* **Relational Data Integrity:** Implements defensive database writing. It automatically constructs "dummy" parent records (e.g., for standalone videos) to satisfy strict Foreign Key constraints in PostgreSQL without dropping data.
* **Lazy-Loaded Machine Learning:** The 500MB RoBERTa Transformer model is lazy-loaded into memory only upon explicit user request, ensuring the UI boots instantly (under 1 second) even on resource-constrained cloud servers.
* **Cloud-Ready SQL Reporting:** Features a robust, dynamically loaded SQLAlchemy engine that compiles relational data into analytical CSV reports without exposing connection strings to the global thread.

---

## 📂 Project Architecture & File Directory

The codebase is highly modular, separating the User Interface, AI Orchestration, External Tools, and Database layers.

### 1. The Presentation Layer
* **`app.py`**
  * **Role:** The main Streamlit application and User Interface. 
  * **Function:** Acts as the entry point for the user. It captures input URLs, displays execution logs from the Agent, visualizes the sentiment data using Plotly pie charts, and provides secure download buttons for the exported reports.

### 2. The AI & Processing Pipeline (`/pipeline`)
* **`pipeline/agent_orchestrator.py`**
  * **Role:** The "Brain" of the operation.
  * **Function:** Initializes the `smolagents` CodeAgent. It defines custom classes inheriting from `Tool` to bypass framework limitations, providing the LLM with a strict set of actions. It contains the prompt engineering required to teach the Agent how to bypass SQL Foreign Key constraints when handling edge cases like single-video links or generic playlists.
* **`pipeline/sentiment_ml.py`**
  * **Role:** The Machine Learning Inspector.
  * **Function:** Houses the Hugging Face `transformers` pipeline using the `cardiffnlp/twitter-roberta-base-sentiment-latest` model. It scans the database for unscored comments, cleans the raw HTML text, analyzes the sentiment (Positive/Neutral/Negative), and updates the database records.
* **`pipeline/reporter.py`**
  * **Role:** The Data Export Engine.
  * **Function:** Safely lazy-loads a SQLAlchemy database connection to extract data from the PostgreSQL tables. It joins relational data to create comprehensive analytical summaries and writes them to `.csv` files.

### 3. The Agent's Toolkit (`/tools`)
* **`tools/input_router.py`**
  * **Role:** The Traffic Cop.
  * **Function:** Uses Regex pattern matching to classify the user's input URL (e.g., Channel Handle vs. Playlist ID vs. Video ID) and passes the classified intent to the Agent Orchestrator.
* **`tools/youtube_tools.py`**
  * **Role:** The Data Harvesters.
  * **Function:** Contains the Python functions that interact directly with the Google YouTube Data v3 API. Features specific tools to resolve channel handles, fetch uploaded playlists, extract video metadata, and pull top-level comments.
* **`tools/db_tools.py`**
  * **Role:** The Database Loaders.
  * **Function:** Accepts the raw dictionaries returned by the YouTube tools and uses SQLAlchemy's `session.merge()` to safely insert or update records in PostgreSQL without creating duplicates.

### 4. Database Infrastructure (`/database`)
* **`database/schema.py`**
  * **Role:** The Blueprint.
  * **Function:** Uses SQLAlchemy ORM to define the exact structural tables of the database (`channels`, `videos`, `comments`). It enforces data integrity through Primary Keys and Foreign Key relationships.

### 5. Workspace & Environment
* **`/exports`**: A dynamically generated directory where the system saves the finalized `.csv` reports.
* **`.env`**: (Git-ignored) Secure vault for API keys and database credentials.
* **`requirements.txt`**: Cloud-optimized dependency list ensuring Linux/Streamlit Cloud compatibility.

---

## ⚙️ Installation & Local Setup

### Prerequisites
* Python 3.10+
* PostgreSQL installed and running locally.
* Google Cloud Console account (for YouTube Data API Key).
* Hugging Face account (for Inference API Token).

### Step-by-Step Guide
1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
   cd YOUR_REPO_NAME
