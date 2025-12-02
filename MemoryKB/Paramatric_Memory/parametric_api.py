import torch
import uvicorn
import threading
import time
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
MODEL_ROOT = "ckpt_parametric_memory"
SEQ_LEN = 1024
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_CHECK_INTERVAL = 3600  # Check for model updates every hour

# ------------------------------------------------------------------------------
# Global Model Variables
# ------------------------------------------------------------------------------
app = FastAPI(title="Parametric Memory Model API", version="1.0")
model = None
tokenizer = None
current_model_version = None
current_model_path = None
model_lock = threading.Lock()  # Prevent race conditions during model loading

# ------------------------------------------------------------------------------
# Model Loading Functions
# ------------------------------------------------------------------------------
def find_latest_model():
    """Automatically find the latest checkpoint_best_* directory"""
    if not os.path.exists(MODEL_ROOT):
        return None
    
    # Get all directories starting with checkpoint_best_
    model_dirs = []
    for item in os.listdir(MODEL_ROOT):
        item_path = os.path.join(MODEL_ROOT, item)
        if os.path.isdir(item_path) and item.startswith("checkpoint_best_"):
            model_dirs.append(item_path)
    
    if not model_dirs:
        return None
    
    # Sort by modification time (newest first)
    model_dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return model_dirs[0]

def get_model_version(model_path):
    """Extract version from model path"""
    dir_name = os.path.basename(model_path)
    if dir_name.startswith("checkpoint_best_"):
        return dir_name.replace("checkpoint_best_", "")
    return dir_name

def load_model(model_path):
    """Load model and tokenizer from specified path"""
    print(f"[Model Loader] Loading model from: {model_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    model.eval()
    
    # Get model version from path
    version = get_model_version(model_path)
    
    return model, tokenizer, version

def update_model():
    """Check for new model version and update if needed"""
    global model, tokenizer, current_model_version, current_model_path
    
    try:
        # Find latest model directory
        latest_model_path = find_latest_model()
        
        if not latest_model_path:
            print(f"[Model Updater] No model directories found in {MODEL_ROOT}")
            return
        
        # Get version of latest model
        latest_version = get_model_version(latest_model_path)
        
        # Skip if version/path is the same
        if latest_model_path == current_model_path and latest_version == current_model_version:
            return
        
        # Load new model with lock
        with model_lock:
            print(f"[Model Updater] Detected new model: {latest_model_path} (version: {latest_version})")
            new_model, new_tokenizer, new_version = load_model(latest_model_path)
            
            # Update global variables
            model = new_model
            tokenizer = new_tokenizer
            current_model_version = new_version
            current_model_path = latest_model_path
            
        print(f"[Model Updater] Successfully updated to version: {new_version}")
        
    except Exception as e:
        print(f"[Model Updater] Error updating model: {str(e)}")

def model_update_worker():
    """Background worker for periodic model updates"""
    while True:
        update_model()
        time.sleep(MODEL_CHECK_INTERVAL)

# ------------------------------------------------------------------------------
# API Request/Response Models
# ------------------------------------------------------------------------------
class QueryRequest(BaseModel):
    query: str
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9

# ------------------------------------------------------------------------------
# API Endpoints
# ------------------------------------------------------------------------------
@app.post("/generate", summary="Generate response from query")
async def generate(request: QueryRequest):
    """Generate response using the latest model"""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Build prompt
        prompt = f"Question: {request.query}\nPlease provide a concise and relevant analysis or context to answer this question.\nAnalysis:"
        
        # Tokenize input
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=SEQ_LEN - request.max_new_tokens,
            padding=True
        ).to(DEVICE)
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        # Extract generated content
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_content = full_response[len(prompt):].strip()
        
        return {
            "query": request.query,
            "response": generated_content,
            "model_version": current_model_version,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.get("/health", summary="Health check endpoint")
async def health():
    """Check service health and model status"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "current_model_version": current_model_version,
        "current_model_path": current_model_path,
        "timestamp": datetime.now().isoformat(),
        "device": str(DEVICE)
    }

@app.get("/model_info", summary="Get current model information")
async def model_info():
    """Get details about the currently loaded model"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Get all available models
    available_models = []
    if os.path.exists(MODEL_ROOT):
        for item in os.listdir(MODEL_ROOT):
            item_path = os.path.join(MODEL_ROOT, item)
            if os.path.isdir(item_path) and item.startswith("checkpoint_best_"):
                available_models.append({
                    "path": item_path,
                    "version": get_model_version(item_path),
                    "modified_time": datetime.fromtimestamp(os.path.getmtime(item_path)).isoformat()
                })
    
    return {
        "current_model_version": current_model_version,
        "current_model_path": current_model_path,
        "tokenizer_vocab_size": len(tokenizer),
        "device": str(DEVICE),
        "last_updated": datetime.now().isoformat(),
        "available_models": available_models
    }

@app.get("/refresh_model", summary="Manually trigger model refresh")
async def refresh_model():
    """Manually check for new models and update"""
    try:
        update_model()
        return {
            "status": "success",
            "message": "Model refresh completed",
            "current_model_version": current_model_version,
            "current_model_path": current_model_path,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model refresh failed: {str(e)}")

# ------------------------------------------------------------------------------
# Initialize Service
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # Load initial model
    print(f"[API Service] Starting service...")
    
    # Find and load latest model
    latest_model_path = find_latest_model()
    if latest_model_path:
        model, tokenizer, current_model_version = load_model(latest_model_path)
        current_model_path = latest_model_path
        print(f"[API Service] Loaded initial model version: {current_model_version}")
        print(f"[API Service] Model path: {current_model_path}")
    else:
        print(f"[API Service] No initial model found in {MODEL_ROOT}. Waiting for training to complete...")
    
    # Start background model update worker
    update_thread = threading.Thread(target=model_update_worker, daemon=True)
    update_thread.start()
    print(f"[API Service] Model update worker started (check interval: {MODEL_CHECK_INTERVAL}s)")
    
    # Start API server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        workers=1
    )