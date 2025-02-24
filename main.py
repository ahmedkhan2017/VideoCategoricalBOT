# -*- coding: utf-8 -*-
"""
Final backend code for Video Categorical Bot.
Each category uses its own directories.
The uploaded video’s filename is made unique by appending a timestamp.
The chat endpoint (/chat/{category}) uses only the latest video transcript of that category.
"""

import os
import base64
from datetime import datetime
from threading import Thread

import torch
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

try:
    from moviepy.editor import VideoFileClip
except ImportError:
    from moviepy import VideoFileClip

import speech_recognition as sr
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# Load environment variables from .env file (if running locally)
from dotenv import load_dotenv
load_dotenv()

# Get API Key from Environment Variables
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set. Please set it in Railway project settings or .env file.")

print("Loaded API Key:", OPENAI_API_KEY[:4] + "****")

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Initialize FastAPI app
app = FastAPI()

# Enable CORS (for frontend compatibility)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize LangChain Chat Model
llm = ChatOpenAI(
    temperature=0,
    model="meta-llama/llama-3-8b-instruct:free",
    openai_api_key=OPENAI_API_KEY,
    base_url=os.environ.get("BASE_URL", "https://openrouter.ai/api/v1")
)

# Define categories
CATEGORIES = ["Tuesday", "Thursday", "Wednesday", "Advance Option"]

# Create directories for each category
for category in CATEGORIES:
    for subdir in ["videos", "audio", "text"]:
        os.makedirs(os.path.join(subdir, category), exist_ok=True)

def get_unique_filename(original_filename: str) -> str:
    """Generates a unique filename by appending a timestamp."""
    name, ext = os.path.splitext(original_filename)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    return f"{name}_{timestamp}{ext}"

def extract_text_from_video(video_path: str, category: str) -> str:
    """Extracts audio from a video and transcribes it."""
    try:
        video_clip = VideoFileClip(video_path)
        audio_filename = os.path.basename(video_path).replace(".mp4", ".wav")
        audio_path = os.path.join("audio", category, audio_filename)
        video_clip.audio.write_audiofile(audio_path, logger=None)

        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data)
        except (sr.UnknownValueError, sr.RequestError):
            text = ""

        text_filename = os.path.basename(video_path).replace(".mp4", ".txt")
        text_path = os.path.join("text", category, text_filename)
        with open(text_path, "w", encoding="utf-8") as text_file:
            text_file.write(text)

        return text
    except Exception as e:
        print(f"Error processing video {video_path}: {str(e)}")
        return ""

@app.post("/upload-video/{category}")
async def upload_video(
    category: str,
    file: UploadFile = File(...),
    label: str = Form(..., examples=["Weekly Meeting"]),
    description: str = Form(..., examples=["Team meeting about project updates"]),
    date: str = Form(..., examples=["2023-10-05"])
):
    if category not in CATEGORIES:
        raise HTTPException(status_code=400, detail=f"Invalid category. Choose from {CATEGORIES}")
    try:
        unique_filename = get_unique_filename(file.filename)
        video_path = os.path.join("videos", category, unique_filename)
        with open(video_path, "wb") as buffer:
            buffer.write(await file.read())

        metadata_path = video_path.replace(".mp4", "_metadata.txt")
        with open(metadata_path, "w", encoding="utf-8") as metadata_file:
            metadata_file.write(f"Label: {label}\nDescription: {description}\nDate: {date}\n")

        transcript = extract_text_from_video(video_path, category)
        with open(metadata_path, "a", encoding="utf-8") as metadata_file:
            metadata_file.write(f"Transcript: {transcript}\n")

        return JSONResponse(content={"filename": unique_filename, "label": label, "description": description, "date": date, "transcript": transcript})
    except Exception as e:
        print(f"Error processing video: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")

@app.get("/list-categories")
async def list_categories():
    return JSONResponse(content={"categories": CATEGORIES})

def start_fastapi():
    """Start FastAPI server on port 8080 (required for Railway)."""
    port = int(os.environ.get("PORT", 8080))  # Always use port 8080 for Railway
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    start_fastapi()
