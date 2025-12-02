# 🚀 Parametric Memory Training & Inference
This document describes the complete workflow for training, deploying, and using the parametric memory module in MemVerse. The system supports scheduled training (every 24 hours by default) and automatic model updates for the inference API, enabling seamless continuous optimization.

## 📋 Table of Contents

## 1. Training Parametric Memory

The training script supports scheduled retraining every 24 hours, with versioned model saving and automatic latest model tracking.

### 1.1 Prepare Training Data

The training data must be a JSON file with the following format (each entry contains a query and its corresponding retrieved content):

```json

[
  {
    "id": 1,
    "query": "What is photosynthesis?",
    "retrieved": "Photosynthesis is the process by which green plants, algae, and some bacteria convert light energy into chemical energy stored in glucose. This process occurs in chloroplasts and requires sunlight, water, and carbon dioxide, producing oxygen as a byproduct."
  },
  {
    "id": 2,
    "query": "Explain Newton's second law of motion.",
    "retrieved": "Newton's second law of motion states that the acceleration of an object is directly proportional to the net force acting on it and inversely proportional to its mass. Mathematically, it is expressed as F = ma, where F is the net force, m is the mass of the object, and a is the acceleration."
  }
]
```

Save this file (e.g., `train_data.json`) in the `param_memory/` directory.

### 1.2 Launch Scheduled Training

Use the following command to start the training process (supports GPU acceleration by default):

```bash

python parametric_train.py \
  --data_path param_memory/train_data.json \
  --seq_len 1024 \
  --batch_size 4 \
  --gradient_accumulation_steps 4 \
  --epochs 5 \
  --lr 2e-6 \
  --weight_decay 0.01 \
  --max_grad_norm 1.0 \
  --fp16 \
  --num_workers 4 \
  --out_dir ./ckpt_parametric_memory \
  --interval_hours 24
```

### 1.3 Key Training Features

- **Scheduled Retraining**: Automatically retrains every 24 hours (configurable via `--interval_hours`)

- **Versioned Model Saving**: Saves models with timestamp-based names (e.g., `checkpoint_best_20251202_115436`) to avoid overwriting

- **Best Model Tracking**: Automatically records the best-performing model during training

- **Mixed Precision Training**: Enables FP16 training via `--fp16` to save GPU memory

- **Reproducibility**: Fixes random seeds to ensure consistent training results

### 1.4 Training Parameter Explanation

|Flag|Type|Default Value|Description|
|---|---|---|---|
|--data_path|string|param_memory/example_data.json|Path to the training JSON file|
|--seq_len|integer|1024|Maximum token length for input (query) + output (retrieved content)|
|--batch_size|integer|4|Batch size per GPU (adjust based on GPU memory)|
|--gradient_accumulation_steps|integer|4|Number of steps to accumulate gradients (increases effective batch size)|
|--epochs|integer|5|Number of training epochs per cycle|
|--lr|float|2e-6|Learning rate (recommended: 1e-6 ~ 5e-6 for 7B models)|
|--weight_decay|float|0.01|Weight decay coefficient for regularization|
|--max_grad_norm|float|1.0|Maximum gradient norm for clipping (prevents gradient explosion)|
|--fp16|flag|False|Enable mixed-precision training (FP16)|
|--num_workers|integer|4|Number of threads for data loading|
|--out_dir|string|./ckpt_parametric_memory|Directory to save model checkpoints|
|--interval_hours|float|24|Interval between training cycles (in hours)|
GPU Memory Requirement: At least 16GB VRAM is recommended for training the Qwen2.5-7B model with the default parameters. If you encounter OOM (Out of Memory) errors, reduce `--batch_size` or increase `--gradient_accumulation_steps`.

## 2. Deploying Inference API

The API service supports automatic latest model discovery, background model updates, and zero-downtime deployment.

### 2.1 Start the API Server

Launch the API service with the following command:

```bash

python parametric_api.py
```

### 2.2 Key API Features

- **Automatic Model Discovery**: Scans the `./ckpt_parametric_memory` directory for the latest trained model (no manual softlink required)

- **Background Model Updates**: Checks for new models every hour (configurable via `MODEL_CHECK_INTERVAL`) and loads them without restarting the service

- **Thread-Safe Loading**: Uses a lock to ensure model updates do not disrupt ongoing inference requests

- **Health Monitoring**: Provides dedicated endpoints for service health checks and model information

- **Fallback Handling**: Gracefully handles cases where no initial model exists (waits for training to complete)

### 2.3 Verify Deployment

After starting the API, verify its status using the following commands:

```bash

# Check service health
curl http://localhost:8001/health

# Get detailed model information
curl http://localhost:8001/model_info
```

Successful health check response:

```json

{
  "status": "healthy",
  "model_loaded": true,
  "current_model_version": "20251202_115436",
  "current_model_path": "./ckpt_parametric_memory/checkpoint_best_20251202_115436",
  "timestamp": "2025-12-02T12:00:00.000000",
  "device": "cuda"
}
```

## 3. Using the API

The API provides a `/generate` endpoint to generate parametric memory content for a given query.

### 3.1 Generate Response

Send a POST request to the `/generate` endpoint (example using curl):

```bash

curl -X POST "http://localhost:8001/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is parametric memory in MemVerse?",
    "max_new_tokens": 256,
    "temperature": 0.7,
    "top_p": 0.9
  }'
```

### 3.2 Response Example

```json

{
  "query": "What is parametric memory in MemVerse?",
  "response": "Parametric memory in MemVerse refers to the module that stores task-specific knowledge within the model's parameters. It is trained to generate relevant analysis or context for a given query based on pre-retrieved content, enabling the system to provide accurate and concise responses without explicit external knowledge bases. This module is continuously updated through scheduled retraining to adapt to new data.",
  "model_version": "20251202_115436",
  "timestamp": "2025-12-02T12:05:30.123456"
}
```

### 3.3 API Endpoints

|Endpoint|Method|Description|Request Parameters|
|---|---|---|---|
|/generate|POST|Generate parametric memory response for a query|query (required), max_new_tokens, temperature, top_p|
|/health|GET|Check service health and model loading status|None|
|/model_info|GET|Get detailed information about the current model and available versions|None|
|/refresh_model|GET|Manually trigger a model refresh (bypasses the hourly check)|None|
### 3.4 Request Parameter Details

|Parameter|Type|Default Value|Description|
|---|---|---|---|
|query|string|-|User's input question/query (required)|
|max_new_tokens|integer|256|Maximum number of tokens to generate (controls response length)|
|temperature|float|0.7|Generation randomness (0 = deterministic, 1 = more creative)|
|top_p|float|0.9|Nucleus sampling parameter (controls diversity of generation)|
## 4. System Architecture

### 4.1 Training Pipeline

1. Data Preparation: Load and preprocess the JSON training data

2. Scheduled Trigger: Start training every 24 hours (configurable)

3. Model Training: Fine-tune Qwen2.5-7B with mixed precision support

4. Versioned Saving: Save the best model with a timestamp-based name

5. Wait for Next Cycle: Sleep until the next training interval

### 4.2 Deployment Pipeline

1. Initialization: Scan for the latest model and load it (if exists)

2. API Serving: Start the FastAPI server to handle inference requests

3. Background Update: Check for new models every hour in a separate thread

4. Zero-Downtime Update: Load new models without interrupting ongoing requests

5. Monitoring: Provide health and model info endpoints for observability

## 5. Advanced Usage

### 5.1 Custom Training Interval

Adjust the training interval (e.g., every 12 hours):

```bash

python parametric_train.py --interval_hours 12 [other parameters]
```

### 5.2 Manual Model Refresh

Force the API to check for new models immediately (useful after manual training):

```bash

curl http://localhost:8001/refresh_model
```

### 5.3 Modify API Configuration

Adjust API settings (e.g., port, model check interval) by modifying the `parametric_api.py` file:

```python

# Configuration section in parametric_api.py
MODEL_ROOT = "ckpt_parametric_memory"  # Model checkpoint directory
SEQ_LEN = 1024  # Same as training
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_CHECK_INTERVAL = 3600  # Model check interval (seconds, default: 1 hour)
PORT = 8001  # API port
```

### 5.4 Rollback to Previous Model

To use an older model version, modify the `find_latest_model()` function in `parametric_api.py` to return the desired model path, or temporarily replace the latest model path in the code.

## 6. Troubleshooting

### 6.1 Common Issues

- **API: Model not loaded (503 error)**Cause: Training has not completed yet, or no models exist in `./ckpt_parametric_memory`

- Solution: Start the training script first and wait for the first training cycle to complete

**Training: GPU Out of Memory (OOM)**Cause: Batch size is too large for the available GPU memory

Solution: Reduce `--batch_size` (e.g., to 2) or enable `--fp16`

**API: Address already in use (98 error)**Cause: Port 8001 is occupied by another process

Solution: Modify the `PORT` in `parametric_api.py` (e.g., to 8002) and restart the API

**Training: Data file not found**Cause: Incorrect `--data_path` parameter

Solution: Verify the path to the training JSON file and correct the `--data_path` value

### 6.2 Logging

- Training logs: Printed to the console (redirect to a file with `> training.log 2>&1`)

- API logs: Printed to the console (redirect to a file with `> api.log 2>&1`)

### 6.3 Environment Requirements

```bash

# Install required packages
pip install torch transformers fastapi uvicorn pydantic tqdm
```

## 📝 Notes

- For production deployments, add authentication (e.g., API keys) and rate limiting to the API service.

- Ensure the training and API services have read/write permissions for the `./ckpt_parametric_memory` directory.

- To stop the scheduled training or API service, press `Ctrl + C` in the terminal.
