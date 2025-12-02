import os
from pathlib import Path
from openai import OpenAI

def process_audio(file_path: Path) -> str:
    """
    Convert audio file to text using OpenAI Whisper API.
    Reads API key and Base URL from environment variables:
      OPENAI_API_KEY
      OPENAI_BASE_URL
    """
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")

    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set.")

    client = OpenAI(
        api_key=api_key,
        **({"base_url": base_url} if base_url else {})
    )

    with open(file_path, "rb") as f:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=f
        )

    return transcript.text