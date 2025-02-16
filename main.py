from fastapi import FastAPI, HTTPException
import subprocess
import json
import os
import aiofiles
import sqlite3
import datetime
import numpy as np
from dateutil import parser
from openai import OpenAI
from dotenv import load_dotenv
import urllib.request
import pytesseract
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Load environment variables
import os
AIPROXY_TOKEN = os.environ["AIPROXY_TOKEN"]

USER_EMAIL = os.environ["USER_EMAIL"]

app = FastAPI()

# Task A1: Install `uv` and Run `datagen.py`
def execute_a1_task():
    try:
        script_url = "https://raw.githubusercontent.com/sanand0/tools-in-data-science-public/tds-2025-01/project-1/datagen.py"
        script_path = os.path.join(os.getcwd(), "datagen.py")
        urllib.request.urlretrieve(script_url, script_path)
        subprocess.run(["pip", "install", "uv"], check=True)
        if not USER_EMAIL:
            raise ValueError("USER_EMAIL is not set")
        python_path = os.path.join(os.getcwd(), "virie", "Scripts", "python.exe")  
        result = subprocess.run([python_path, script_path, USER_EMAIL], capture_output=True, text=True)

        if result.returncode != 0:
            raise HTTPException(status_code=400, detail=f"Error executing task A1: {result.stderr}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Unexpected error: {str(e)}")

# Task A2: Format a Markdown File Using Prettier
def execute_a2_task():
    try:
        subprocess.run(["C:/Program Files/nodejs/npx.cmd", "prettier", "--write", "C:/data/format.md"], check=True)
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=400, detail=f"Error executing task A2: {str(e)}")

# Task A3: Handle Multiple Date Formats and Extract Wednesdays
def execute_a3_task():
    input_file = "C:/data/dates.txt"
    output_file = "C:/data/dates-wednesdays.txt"

    if not os.path.exists(input_file):
        raise HTTPException(status_code=400, detail=f"File not found: {input_file}")

    wednesday_count = 0
    with open(input_file, "r") as f:
        for line in f:
            date_str = line.strip()
            if not date_str:
                continue
            try:
                parsed_date = parser.parse(date_str)
                if parsed_date.weekday() == 2:
                    wednesday_count += 1
            except ValueError:
                continue

    with open(output_file, "w") as f:
        f.write(str(wednesday_count))

# Task A4: Sort Contacts JSON by Last Name
def execute_a4_task():
    input_file = "C:/data/contacts.json"
    output_file = "C:/data/contacts-sorted.json"

    if not os.path.exists(input_file):
        raise HTTPException(status_code=400, detail=f"File not found: {input_file}")

    with open(input_file, "r") as f:
        contacts = json.load(f)

    contacts.sort(key=lambda x: (x["last_name"], x["first_name"]))

    with open(output_file, "w") as f:
        json.dump(contacts, f, indent=2)

# Task A5: Extract Recent Log Entries
def execute_a5_task():
    log_dir = "C:/data/logs"
    output_file = "C:/data/logs-recent.txt"

    if not os.path.exists(log_dir):
        raise HTTPException(status_code=400, detail=f"Directory not found: {log_dir}")

    log_files = sorted(
        [f for f in os.listdir(log_dir) if f.endswith(".log")],
        key=lambda f: os.path.getmtime(os.path.join(log_dir, f)),
        reverse=True
    )

    recent_logs = []
    for log_file in log_files[:10]:
        with open(os.path.join(log_dir, log_file), "r") as f:
            recent_logs.append(f.readline().strip())

    with open(output_file, "w") as f:
        f.write("\n".join(recent_logs))

# Task A6: Generate Index for Markdown Files
def execute_a6_task():
    try:
        index = {}
        docs_path = "C:/data/docs/"

        for root, _, files in os.walk(docs_path):
            for file in files:
                if file.endswith(".md"):
                    file_path = os.path.join(root, file)
                    with open(file_path, "r", encoding="utf-8") as f:
                        for line in f:
                            if line.startswith("# "):
                                relative_path = os.path.relpath(file_path, docs_path)
                                index[relative_path] = line.strip("# ").strip()
                                break

        with open("C:/data/docs/index.json", "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error executing task A6: {str(e)}")

# Task A7: Extract Sender's Email from a Text File
def execute_a7_task():
    input_file = "C:/data/email.txt"
    output_file = "C:/data/email-sender.txt"

    if not os.path.exists(input_file):
        raise HTTPException(status_code=400, detail=f"File not found: {input_file}")

    with open(input_file, "r") as f:
        email_content = f.read()

    client = OpenAI(api_key=AIPROXY_TOKEN, base_url="https://aiproxy.sanand.workers.dev/openai/v1")
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": f"Extract the sender's email address from: {email_content}"}]
    )
    email_address = response.choices[0].message.content.strip()

    with open(output_file, "w") as f:
        f.write(email_address)

import os
import openai
import base64

def execute_a8_task():
    image_path = "C:/data/credit_card.png"

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"File not found: {image_path}")

    client = openai.Client(api_key=AIPROXY_TOKEN, base_url="https://aiproxy.sanand.workers.dev/openai/v1")

    # Convert image to Base64
    with open(image_path, "rb") as img_file:
        image_base64 = base64.b64encode(img_file.read()).decode("utf-8")

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Extract all readable text from this image."},
                {"role": "user", "content": [
                    {"type": "text", "text": "Extract the credit card number and expiry date from this image."},
                    {"type": "image_url", "image_url": f"data:image/png;base64,{image_base64}"}
                ]}
            ]
        )

        extracted_text = response.choices[0].message.content.strip()

        output_file = "C:/data/image-text.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(extracted_text)

        return {"message": "Text extracted successfully", "output": extracted_text}

    except openai.OpenAIError as e:
        return {"error": f"OpenAI API error: {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}







# Task A9: Find Most Similar Comments
def execute_a9_task():
    input_file = "C:/data/comments.txt"
    output_file = "C:/data/comments-similar.txt"
    
    if not os.path.exists(input_file):
        raise HTTPException(status_code=400, detail=f"File not found: {input_file}")
    
    with open(input_file, "r") as f:
        comments = [line.strip() for line in f.readlines() if line.strip()]
    
    if len(comments) < 2:
        raise HTTPException(status_code=400, detail="Not enough comments to compare")
    
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(comments)
    similarity_matrix = cosine_similarity(embeddings)
    
    max_sim = -1
    best_pair = (None, None)
    for i in range(len(comments)):
        for j in range(i + 1, len(comments)):
            if similarity_matrix[i][j] > max_sim:
                max_sim = similarity_matrix[i][j]
                best_pair = (comments[i], comments[j])
    
    with open(output_file, "w") as f:
        f.write("\n".join(best_pair))
# Task A10: Calculate Total Sales of 'Gold' Tickets
def execute_a10_task():
    db_path = "C:/data/ticket-sales.db"
    output_file = "C:/data/ticket-sales-gold.txt"
    
    if not os.path.exists(db_path):
        raise HTTPException(status_code=400, detail=f"Database not found: {db_path}")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT SUM(units * price) FROM tickets WHERE type = 'Gold'")
    total_sales = cursor.fetchone()[0] or 0
    conn.close()
    
    with open(output_file, "w") as f:
        f.write(str(total_sales))



# Task Mapping
task_mapping = {
    "a1": execute_a1_task,
    "a2": execute_a2_task,
    "a3": execute_a3_task,
    "a4": execute_a4_task,
    "a5": execute_a5_task,
    "a6": execute_a6_task,
    "a7": execute_a7_task,
    "a8": execute_a8_task,
    "a9": execute_a9_task,
    "a10": execute_a10_task
}

# API Endpoints
@app.post("/run")
async def run_task(task: str):
    task_lower = task.lower()
    if task_lower in task_mapping:
        task_mapping[task_lower]()
        return {"status": "success"}
    raise HTTPException(status_code=400, detail="Unknown task")

@app.get("/read")
async def read_file(file_path: str):
    if not os.path.exists(file_path):
        raise HTTPException(status_code=400, detail=f"File not found: {file_path}")
    async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
        content = await f.read()
    return {"content": content}


from fastapi import FastAPI
from main_B import bp  # Import the router from main_B

app = FastAPI()
app.include_router(bp)  # Register the APIRouter

@app.post('/run')
def run_task():
    return {"message": "Task executed from main!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

