import base64
import io
from pathlib import Path
from PIL import Image
import requests
import logging

# ===== Helper functions =====
def load_image(image_path: Path) -> Image.Image:
    return Image.open(image_path).convert('RGB')

def image_to_base64(image: Image.Image) -> str:
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def generate_image_caption(image: Image.Image, prompt: str, api_url: str, api_key: str) -> str:
    """
    Calls GPT-4o API to generate a visual description of the image.
    """
    base64_image = image_to_base64(image)
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    data = {
        "model": "gpt-4o",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]
        }]
    }

    try:
        response = requests.post(api_url, headers=headers, json=data)
        if response.status_code == 200:
            result_json = response.json()
            return result_json['choices'][0]['message']['content'].strip()
        else:
            logging.error(f"Image caption API failed: {response.status_code} {response.text}")
            return f"[Failed to generate caption for {image_path.name}]"
    except Exception as e:
        logging.error(f"Error calling image caption API: {str(e)}")
        return f"[Error generating caption: {str(e)}]"

# ===== Modal processing functions =====
def process_text(text: str) -> str:
    """Process text input. Returns the text as-is."""
    return text

def process_image(file_path: Path, api_url: str, api_key: str) -> str:
    """Process image input. Calls GPT-4o to generate a caption."""
    prompt = """
    Please provide a detailed visual description of this image. 
    Include key objects, their spatial relationships, 
    notable visual features, and any observable actions or events.
    Respond in clear, structured English paragraphs.
    """
    image = load_image(file_path)
    return generate_image_caption(image, prompt, api_url, api_key)