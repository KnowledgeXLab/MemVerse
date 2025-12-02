import os
import io
import base64
from pathlib import Path
from PIL import Image
import subprocess
import requests
from MemoryKB.User_Conversation import process_image as pi
import os

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def extract_frames(video_path: Path, fps: int = 1, temp_dir: Path = None):
    """
    Extract frames from video at given fps.
    Returns a list of frame file paths.
    """
    if temp_dir is None:
        temp_dir = video_path.parent / "frames_temp"
    temp_dir.mkdir(exist_ok=True)

    cmd = [
        "ffmpeg",
        "-i", str(video_path),
        "-vf", f"fps={fps}",
        str(temp_dir / "frame_%05d.jpg")
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    return sorted(temp_dir.glob("frame_*.jpg"))

def summarize_video_captions(captions: list) -> str:
    """
    Summarize a list of captions using GPT-4o.
    """
    if not captions:
        return "[No frames captions to summarize]"

    prompt_text = "Summarize the following frame descriptions into a coherent summary of the video:\n\n"
    prompt_text += "\n".join(captions)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }
    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "user", "content": prompt_text}
        ]
    }

    response = requests.post(f"{OPENAI_BASE_URL}/chat/completions", headers=headers, json=data)
    if response.status_code == 200:
        resp_json = response.json()
        return resp_json['choices'][0]['message']['content'].strip()
    else:
        return f"[Failed to summarize video captions, status code: {response.status_code}]"

def process_video(video_path: Path, fps: int = 1, prompt: str = None) -> str:
    """
    Process video: extract frames, run image captioning, aggregate captions, summarize video.
    """
    if prompt is None:
        prompt = """
        Please provide a detailed visual description of this image. 
        Include key objects, their spatial relationships, 
        notable visual features, and any observable actions or events.
        Respond in clear, structured English paragraphs.
        """

    frame_files = extract_frames(video_path, fps=fps)
    if not frame_files:
        return f"[No frames extracted from {video_path.name}]"

    captions = []
    for frame in frame_files:
        caption = pi.process_image(frame, prompt=prompt)
        captions.append(caption)

    # Summarize all frame captions into one video description
    video_caption = summarize_video_captions(captions)
    return video_caption