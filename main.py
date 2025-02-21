# -*- coding: utf-8 -*-
"""
Final backend code for Video Categorical Bot
Is code mein har category ke liye alag directories use ho rahi hain.
Uploaded video ka file name ab unique banane ke liye timestamp add kiya gaya hai.
Chat endpoint (/chat/{category}) ab sirf us category ke latest video transcript
ko use karega response generate karne ke liye.
"""

import os
import base64
from datetime import datetime
from threading import Thread

import torch
import uvicorn
import nest_asyncio
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pyngrok import ngrok

try:
    from moviepy.editor import VideoFileClip
except ImportError:
    from moviepy import VideoFileClip

import speech_recognition as sr
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Debug: Check if API key is loaded (remove or comment out in production)
print("Loaded API Key:", os.environ.get("OPENAI_API_KEY"))

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Apply nest_asyncio to allow running async tasks (useful for Colab/local)
nest_asyncio.apply()

# Initialize FastAPI app
app = FastAPI()

# Enable CORS (adjust allowed origins for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize LangChain components (using OpenRouter as base_url)
llm = ChatOpenAI(
    temperature=0,
    model="meta-llama/llama-3-8b-instruct:free",
    openai_api_key=os.environ.get("OPENAI_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

# Define allowed categories
CATEGORIES = ["Tuesday", "Thursday", "Wednesday", "Advance Option"]

# Create directories for each category to store videos, audio, and text/transcripts
for category in CATEGORIES:
    os.makedirs(os.path.join("videos", category), exist_ok=True)
    os.makedirs(os.path.join("audio", category), exist_ok=True)
    os.makedirs(os.path.join("text", category), exist_ok=True)

def get_unique_filename(original_filename: str) -> str:
    """
    Generates a unique filename by appending the current timestamp
    to the original filename (before the extension).
    """
    name, ext = os.path.splitext(original_filename)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    return f"{name}_{timestamp}{ext}"

# Function to extract text from a video using SpeechRecognition
def extract_text_from_video(video_path: str, category: str) -> str:
    try:
        # Extract audio from the video
        video_clip = VideoFileClip(video_path)
        audio_filename = os.path.basename(video_path).replace(".mp4", ".wav")
        audio_path = os.path.join("audio", category, audio_filename)
        video_clip.audio.write_audiofile(audio_path, logger=None)

        # Transcribe the audio using SpeechRecognition
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data)
        except sr.UnknownValueError:
            text = ""
        except sr.RequestError:
            text = ""
        
        # Save the transcript to a text file in the corresponding category folder
        text_filename = os.path.basename(video_path).replace(".mp4", ".txt")
        text_path = os.path.join("text", category, text_filename)
        with open(text_path, "w", encoding="utf-8") as text_file:
            text_file.write(text)

        return text
    except Exception as e:
        print(f"Error processing video {video_path}: {str(e)}")
        return ""

# Endpoint to upload video using direct file upload
@app.post("/upload-video/{category}")
async def upload_video(
    category: str,
    file: UploadFile = File(...),
    label: str = Form(..., example="Weekly Meeting"),
    description: str = Form(..., example="Team meeting about project updates"),
    date: str = Form(..., example="2023-10-05")
):
    if category not in CATEGORIES:
        raise HTTPException(status_code=400, detail=f"Invalid category. Choose from {CATEGORIES}")
    try:
        # Generate a unique filename for the video
        unique_filename = get_unique_filename(file.filename)
        video_path = os.path.join("videos", category, unique_filename)
        with open(video_path, "wb") as buffer:
            buffer.write(await file.read())

        # Create metadata file in the text folder of the same category
        metadata_filename = unique_filename.replace(".mp4", "_metadata.txt")
        metadata_path = os.path.join("text", category, metadata_filename)
        with open(metadata_path, "w", encoding="utf-8") as metadata_file:
            metadata_file.write(f"Label: {label}\n")
            metadata_file.write(f"Description: {description}\n")
            metadata_file.write(f"Date: {date}\n")

        # Extract transcript from the video and save it
        transcript = extract_text_from_video(video_path, category)

        with open(metadata_path, "a", encoding="utf-8") as metadata_file:
            metadata_file.write(f"Transcript: {transcript}\n")

        return JSONResponse(content={
            "filename": unique_filename,
            "label": label,
            "description": description,
            "date": date,
            "transcript": transcript
        })
    except Exception as e:
        print(f"Error processing video: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")

# Endpoint to upload video using Base64 encoding
@app.post("/upload-video-base64/{category}")
async def upload_video_base64(category: str, file_data: dict):
    if category not in CATEGORIES:
        raise HTTPException(status_code=400, detail=f"Invalid category. Choose from {CATEGORIES}")
    try:
        file_name = file_data.get("file_name")
        data_url = file_data.get("data_url")
        label = file_data.get("label")
        description = file_data.get("description")
        date = file_data.get("date")

        if not data_url or not label or not description or not date:
            raise HTTPException(status_code=400, detail="All fields ('data_url', 'label', 'description', 'date') are required.")

        if not data_url.startswith("data:video/mp4;base64,"):
            raise HTTPException(status_code=400, detail="Invalid Data URL format. Expected 'data:video/mp4;base64,' prefix.")

        base64_content = data_url.split("data:video/mp4;base64,")[1]
        file_bytes = base64.b64decode(base64_content)

        unique_filename = get_unique_filename(file_name)
        video_path = os.path.join("videos", category, unique_filename)
        with open(video_path, "wb") as buffer:
            buffer.write(file_bytes)

        metadata_filename = unique_filename.replace(".mp4", "_metadata.txt")
        metadata_path = os.path.join("text", category, metadata_filename)
        with open(metadata_path, "w", encoding="utf-8") as metadata_file:
            metadata_file.write(f"Label: {label}\n")
            metadata_file.write(f"Description: {description}\n")
            metadata_file.write(f"Date: {date}\n")

        transcript = extract_text_from_video(video_path, category)

        with open(metadata_path, "a", encoding="utf-8") as metadata_file:
            metadata_file.write(f"Transcript: {transcript}\n")

        return JSONResponse(content={
            "filename": unique_filename,
            "label": label,
            "description": description,
            "date": date,
            "message": "Video uploaded successfully.",
            "transcript": transcript
        })
    except Exception as e:
        print(f"Error processing video: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")

# Endpoint for chat based on the latest transcript of a specific category
@app.get("/chat/{category}")
async def chat_with_bot(category: str, query: str):
    if category not in CATEGORIES:
        raise HTTPException(status_code=400, detail=f"Invalid category. Choose from {CATEGORIES}")
    try:
        # List transcript files in the specific category folder
        transcript_dir = os.path.join("text", category)
        transcript_files = [f for f in os.listdir(transcript_dir) if f.endswith(".txt")]
        if not transcript_files:
            raise HTTPException(status_code=404, detail=f"No transcripts found for category: {category}")

        # Use only the latest transcript file (based on modification time)
        latest_file = max(transcript_files, key=lambda f: os.path.getmtime(os.path.join(transcript_dir, f)))
        transcript_path = os.path.join(transcript_dir, latest_file)
        with open(transcript_path, "r", encoding="utf-8") as f:
            transcript = f.read()

        # Build the prompt using only the latest transcript
        template = f"""You are an AI assistant specialized in {category} topics.
Context from the transcript:
{{context_data}}
Based on the above, answer the following question: {query}
Provide a detailed and relevant response:"""
        prompt = PromptTemplate.from_template(template)
        chain = LLMChain(llm=llm, prompt=prompt)
        try:
            response = chain.run({'context_data': transcript})
        except Exception as e:
            print(f"Error generating AI response: {str(e)}")
            response = "Error generating AI response, please try again later."
        return JSONResponse(content={"response": response})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

# Endpoint for chat based on a specific video's transcript within its category
@app.get("/chat-video/{category}/{filename}")
async def chat_with_video(category: str, filename: str, query: str):
    if category not in CATEGORIES:
        raise HTTPException(status_code=400, detail=f"Invalid category. Choose from {CATEGORIES}")
    transcript_path = os.path.join("text", category, filename.replace(".mp4", ".txt"))
    if not os.path.exists(transcript_path):
        raise HTTPException(status_code=404, detail=f"No transcript found for {filename} in category {category}.")
    try:
        with open(transcript_path, "r", encoding="utf-8") as f:
            transcript = f.read()
        template = f"""You are an AI assistant specialized in {category} topics.
Context from the transcript:
{{context_data}}
Based on the above, answer the following question: {query}
Provide a detailed and relevant response:"""
        prompt = PromptTemplate.from_template(template)
        chain = LLMChain(llm=llm, prompt=prompt)
        try:
            response = chain.run({'context_data': transcript})
        except Exception as e:
            print(f"Error generating AI response: {str(e)}")
            response = "Error generating AI response, please try again later."
        return JSONResponse(content={"response": response})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

# Endpoint to delete a video and its associated files (only within the specified category)
@app.get("/delete-video/{category}/{filename}")
async def delete_video(category: str, filename: str):
    if category not in CATEGORIES:
        raise HTTPException(status_code=400, detail=f"Invalid category. Choose from {CATEGORIES}")
    try:
        video_path = os.path.join("videos", category, filename)
        audio_path = os.path.join("audio", category, filename.replace(".mp4", ".wav"))
        text_path = os.path.join("text", category, filename.replace(".mp4", ".txt"))
        metadata_path = os.path.join("text", category, filename.replace(".mp4", "_metadata.txt"))
        deleted_files = []
        for path, label in [(video_path, "video"), (audio_path, "audio"), (text_path, "text"), (metadata_path, "metadata")]:
            if os.path.exists(path):
                os.remove(path)
                deleted_files.append(label)
        if not deleted_files:
            raise HTTPException(status_code=404, detail=f"No files found for {filename} in category {category}.")
        return JSONResponse(content={"message": f"Deleted {', '.join(deleted_files)} for {filename} in category {category}."})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting files: {str(e)}")

# Endpoint to list all videos in a specific category along with their metadata
@app.get("/list-videos/{category}")
async def list_videos(category: str):
    if category not in CATEGORIES:
        raise HTTPException(status_code=400, detail=f"Invalid category. Choose from {CATEGORIES}")
    try:
        video_dir = os.path.join("videos", category)
        video_files = os.listdir(video_dir)
        videos_with_metadata = []
        for video_file in video_files:
            metadata_filename = video_file.replace(".mp4", "_metadata.txt")
            metadata_path = os.path.join("text", category, metadata_filename)
            metadata = {}
            if os.path.exists(metadata_path):
                with open(metadata_path, "r", encoding="utf-8") as metadata_file:
                    for line in metadata_file.readlines():
                        if ": " in line:
                            key, value = line.strip().split(": ", 1)
                            metadata[key.lower()] = value
            videos_with_metadata.append({
                "filename": video_file,
                "label": metadata.get("label", ""),
                "description": metadata.get("description", ""),
                "date": metadata.get("date", "")
            })
        return JSONResponse(content={"videos": videos_with_metadata})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing videos: {str(e)}")

# Endpoint to retrieve the transcript of a specific video from its category
@app.get("/get-transcript/{category}/{filename}")
async def get_transcript(category: str, filename: str):
    if category not in CATEGORIES:
        raise HTTPException(status_code=400, detail=f"Invalid category. Choose from {CATEGORIES}")
    try:
        text_path = os.path.join("text", category, filename.replace(".mp4", ".txt"))
        if not os.path.exists(text_path):
            raise HTTPException(status_code=404, detail=f"No transcript found for {filename} in category {category}.")
        with open(text_path, "r", encoding="utf-8") as f:
            transcript = f.read()
        return JSONResponse(content={"transcript": transcript})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving transcript: {str(e)}")

# Endpoint to list all available categories
@app.get("/list-categories")
async def list_categories():
    return JSONResponse(content={"categories": CATEGORIES})

# Function to start the FastAPI app with a dynamic port
def start_fastapi():
    port = int(os.environ.get("PORT", 8530))
    uvicorn.run(app, host="0.0.0.0", port=port)

# Expose the app using NGROK (for Colab or external access) with a dynamic port
def expose_with_ngrok():
    NGROK_AUTH_TOKEN = os.environ.get("NGROK_AUTH_TOKEN", "2sToKHXPVpV45CWcvpIu7o0Xzf7_2cJoBvwQQj7UmyEJ3z2jG")
    ngrok.set_auth_token(NGROK_AUTH_TOKEN)
    port = int(os.environ.get("PORT", 8530))
    public_url = ngrok.connect(port)
    print(f"Public URL: {public_url}")

# Start FastAPI in a background thread and expose it via NGROK
thread = Thread(target=start_fastapi)
thread.start()
expose_with_ngrok()
