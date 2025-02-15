from fastapi import FastAPI, HTTPException
import os
import aiofiles
import json
import sqlite3
import requests
import markdown
import duckdb
import git
import shutil
from PIL import Image
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")
USER_EMAIL = os.getenv("USER_EMAIL")

app = FastAPI()

# Security Constraints
DATA_DIR = "/data"

def ensure_secure_path(path: str):
    if not path.startswith(DATA_DIR):
        raise HTTPException(status_code=400, detail="Access to this path is restricted")

def ensure_no_deletion(task_desc: str):
    if "delete" in task_desc.lower():
        raise HTTPException(status_code=400, detail="Deletion is not allowed")

# B3: Fetch data from an API and save it
def execute_b3_task(api_url: str = "https://jsonplaceholder.typicode.com/todos/1",
                    output_file: str = f"{DATA_DIR}/api-data.json"):
    response = requests.get(api_url)
    with open(output_file, "w") as f:
        f.write(response.text)
    return {"status": "success", "file_saved": output_file}

# B4: Clone a Git repo and make a commit
def execute_b4_task(task_description: str):
    """Extracts repo details using AI Proxy and makes a commit."""
    url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {AIPROXY_TOKEN}", "Content-Type": "application/json"}
    data = {
        "model": "gpt-4-turbo",
        "messages": [{"role": "user", "content": task_description}]
    }
    response = requests.post(url, json=data, headers=headers).json()
    
    repo_url = response.get("repo_url")
    commit_message = response.get("commit_message")
    if not repo_url or not commit_message:
        raise ValueError("Invalid task description: Missing repo details.")
    
    repo_name = repo_url.split("/")[-1].replace(".git", "")
    local_path = os.path.join(DATA_DIR, repo_name)

    if not os.path.exists(local_path):
        git.Repo.clone_from(repo_url, local_path)
    
    repo = git.Repo(local_path)
    repo.git.add(A=True)
    repo.index.commit(commit_message)
    repo.remote(name="origin").push()

    return {"status": "success", "message": f"Committed & pushed: {commit_message}"}

def execute_b5_task(db_path: str, query: str):
    ensure_secure_path(db_path)
    try:
        # Connect to the database
        conn = sqlite3.connect(db_path) if db_path.endswith(".db") else duckdb.connect(db_path)
        cursor = conn.cursor()

        # Execute the query
        cursor.execute(query)
        columns = [desc[0] for desc in cursor.description]  # Get column names
        result = [dict(zip(columns, row)) for row in cursor.fetchall()]  # Convert to list of dicts

    except sqlite3.Error as e:
        result = {"error": str(e)}
    finally:
        conn.close()

    return result


# B6: Extract data from a website
def execute_b6_task(url: str, output_file: str):
    ensure_secure_path(output_file)
    response = requests.get(url)
    if response.status_code != 200:
        raise HTTPException(status_code=400, detail="Failed to fetch webpage")
    soup = BeautifulSoup(response.text, "html.parser")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(soup.get_text())

# B7: Compress or resize an image
def execute_b7_task(input_file: str, output_file: str, size: tuple):
    ensure_secure_path(input_file)
    ensure_secure_path(output_file)
    img = Image.open(input_file)
    img = img.resize(size, Image.ANTIALIAS)
    img.save(output_file, quality=80)

# B8: Transcribe audio using AI Proxy
def execute_b8_task(audio_file: str, output_file: str):
    ensure_secure_path(audio_file)
    ensure_secure_path(output_file)
    url = "https://aiproxy.sanand.workers.dev/openai/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {AIPROXY_TOKEN}"}
    files = {"file": open(audio_file, "rb")}
    data = {"model": "whisper-1"}
    response = requests.post(url, headers=headers, files=files, data=data).json()
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(response.get("text", ""))

# B9: Convert Markdown to HTML
def execute_b9_task(input_file: str, output_file: str):
    ensure_secure_path(input_file)
    ensure_secure_path(output_file)
    with open(input_file, "r", encoding="utf-8") as f:
        md_content = f.read()
    html_content = markdown.markdown(md_content)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_content)

# B10: Filter CSV and return JSON
def execute_b10_task(csv_file: str, column: str, value: str):
    ensure_secure_path(csv_file)
    import pandas as pd
    df = pd.read_csv(csv_file)
    filtered_df = df[df[column] == value]
    return filtered_df.to_json(orient="records")

# Task Mapping
task_mapping = {
    "b3": execute_b3_task,
    "b4": execute_b4_task,
    "b5": execute_b5_task,
    "b6": execute_b6_task,
    "b7": execute_b7_task,
    "b8": execute_b8_task,
    "b9": execute_b9_task,
    "b10": execute_b10_task
}
from fastapi import Request, HTTPException
import sqlite3
import duckdb

def ensure_secure_path(path: str):
    if not path.startswith('/data'):
        raise HTTPException(status_code=403, detail="Access outside /data is forbidden")

def execute_b5_task(db_path: str, query: str):
    ensure_secure_path(db_path)
    conn = sqlite3.connect(db_path) if db_path.endswith(".db") else duckdb.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(query)
    result = cursor.fetchall()
    conn.close()
    return result

@app.post("/run")
async def run_task(task: str, request: Request):
    # Get the plain text from the request body
    task_description = (await request.body()).decode('utf-8').strip()

    # Check if task is in the range b1 to b10
    if task in [f"b{i}" for i in range(1, 11)]:
        # For b4, task_description is mandatory
        if task == "b4" and not task_description:
            raise HTTPException(status_code=400, detail="task_description is required for task b4")

        # For b5, a SQL query is mandatory
        if task == "b5":
            if not task_description:
                raise HTTPException(status_code=400, detail="SQL query is required for task b5")
            # Example database path, change as needed
            db_path = "/data/example.db"
            result = execute_b5_task(db_path, task_description)
            return {"result": result}
        
        # Return a message for all other tasks
        return {"message": f"Running task {task} with description: {task_description or 'No description provided'}"}

    else:
        raise HTTPException(status_code=400, detail=f"Unknown task: {task}")





@app.get("/read")
async def read_file(file_path: str):
    ensure_secure_path(file_path)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=400, detail=f"File not found: {file_path}")
    async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
        content = await f.read()
    return {"content": content}

