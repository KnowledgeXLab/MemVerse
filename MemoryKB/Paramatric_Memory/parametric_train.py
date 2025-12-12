import argparse
import json
import os
import shutil
import random
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    get_scheduler
)
try:
    from torch.optim import AdamW
except ImportError:
    from transformers import AdamW
from tqdm import tqdm  
from torch.cuda.amp import autocast, GradScaler  
from datetime import datetime

# ------------------------------------------------------------------------------
# Dataset Class for Parametric Memory Training
# ------------------------------------------------------------------------------
class ParametricMemoryDataset(Dataset):
    """
    Dataset for parametric memory training: Input=query, Output=retrieved content.
    Ensures fixed-length tensors for batch processing with max-length padding.
    """
    def __init__(self, data_path, tokenizer, seq_len=1024):
        # Load raw JSON data (array of {"id": int, "query": str, "retrieved": str})
        with open(data_path, "r", encoding="utf-8") as f:
            self.raw_data = json.load(f)
        
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        
        # Set pad_token if not exists (Qwen models don't have default pad_token)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            print(f"[Dataset Info] Set pad_token to eos_token (model has no default pad_token)")

    def __len__(self):
        """Return total number of training samples"""
        return len(self.raw_data)

    def __getitem__(self, idx):
        """Return tokenized sample with fixed length"""
        item = self.raw_data[idx]
        query_text = item["query"]
        retrieved_text = item["retrieved"]
        
        # Prompt template (matches training objective: query → retrieved analysis)
        prompt = f"Question: {query_text}\nPlease provide a concise and relevant analysis or context to answer this question.\nAnalysis:"
        
        # Tokenize with fixed-length padding (critical for batch stacking)
        encoding = self.tokenizer(
            prompt,
            text_target=retrieved_text,
            truncation=True,
            max_length=self.seq_len,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Flatten tensors for DataLoader batching
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': encoding['labels'].flatten()
        }

# ------------------------------------------------------------------------------
# Global Configurations
# ------------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

# Set random seeds for reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    print(f"[Global Info] Set random seeds (SEED={SEED})")

# ------------------------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------------------------
def safe_remove_dir(dir_path):
    """Safely remove directory to clean up old checkpoints"""
    if os.path.exists(dir_path):
        try:
            shutil.rmtree(dir_path)
            print(f"[Utility] Removed old directory: {dir_path}")
        except Exception as e:
            print(f"[Warning] Failed to delete {dir_path}. Error: {str(e)}")

def get_model_version():
    """Generate timestamp-based model version (e.g., 20251203_1430)"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

# ------------------------------------------------------------------------------
# Training Function
# ------------------------------------------------------------------------------
def train(args):
    """Main training function with versioned model saving"""
    # 1. Load Model and Tokenizer
    print("\n" + "="*80)
    print(f"[Step 1/6] Loading Model: Qwen/Qwen2.5-7B")
    print("="*80)
    
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-7B",
        trust_remote_code=True,
        padding_side="right"  
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"[Model Info] Set pad_token to eos_token")

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B",
        dtype=torch.bfloat16,       
        device_map="auto",         
        trust_remote_code=True,
        low_cpu_mem_usage=True     
    )

    model.gradient_checkpointing_enable()
    print(f"[Model Info] Gradient checkpointing enabled (saves VRAM)")

    # 2. Load Dataset
    print("\n" + "="*80)
    print(f"[Step 2/6] Loading Dataset")
    print("="*80)
    train_dataset = ParametricMemoryDataset(
        args.data_path,
        tokenizer,
        seq_len=args.seq_len
    )
    print(f"[Dataset Info] Training samples: {len(train_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,               
        num_workers=args.num_workers,
        pin_memory=True             
    )

    # 3. Initialize Optimizer and Scheduler
    print("\n" + "="*80)
    print(f"[Step 3/6] Initializing Training Components")
    print("="*80)
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,                 
        weight_decay=args.weight_decay,  
        betas=(0.9, 0.95)           
    )

    num_training_steps = args.epochs * len(train_loader) // args.gradient_accumulation_steps
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps
    )

    scaler = GradScaler() if args.fp16 else None

    # 4. Training Loop
    print("\n" + "="*80)
    print(f"[Step 4/6] Starting Training")
    print("="*80)
    best_loss = float('inf')
    total_training_time = 0

    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()

        train_pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{args.epochs}",
            unit="batch"
        )

        for batch_idx, batch in enumerate(train_pbar):
            # Move batch to device
            batch = {k: v.to(DEVICE, non_blocking=True) for k, v in batch.items()}

            # Forward pass with mixed precision
            with autocast(enabled=args.fp16):
                outputs = model(**batch)
                loss = outputs.loss / args.gradient_accumulation_steps

            # Backward pass
            if args.fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            step_loss = loss.item() * args.gradient_accumulation_steps
            train_loss += step_loss

            # Optimizer step after accumulation
            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()

                lr_scheduler.step()
                optimizer.zero_grad()

            # Update progress bar
            train_pbar.set_postfix({
                "loss": f"{step_loss:.6f}",
                "avg_loss": f"{train_loss/(batch_idx+1):.6f}",
                "lr": f"{lr_scheduler.get_last_lr()[0]:.8f}"
            })

        train_pbar.close()
        avg_train_loss = train_loss / len(train_loader)
        epoch_time = time.time() - epoch_start_time
        total_training_time += epoch_time

        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Average Loss: {avg_train_loss:.6f}")
        print(f"Epoch Time: {epoch_time/60:.2f}min")

        # Save best model
        if avg_train_loss < best_loss:
            best_loss = avg_train_loss
            model_version = get_model_version()
            best_model_dir = os.path.join(args.out_dir, f"checkpoint_best_{model_version}")
            latest_dir = os.path.join(args.out_dir, "model_latest")

            # Save new best model
            model.save_pretrained(best_model_dir, safe_serialization=True)
            tokenizer.save_pretrained(best_model_dir)

            # Update latest model link
            if os.path.exists(latest_dir):
                if os.path.islink(latest_dir):
                    os.unlink(latest_dir)
                else:
                    safe_remove_dir(latest_dir)
            os.symlink(best_model_dir, latest_dir)
            print(f"[Checkpoint] Saved best model: {best_model_dir}")
            print(f"[Checkpoint] Updated latest model link: {latest_dir}")

    # 5. Save final model
    print("\n" + "="*80)
    print(f"[Step 5/6] Training Completed")
    print("="*80)
    final_model_dir = os.path.join(args.out_dir, f"final_model_{get_model_version()}")
    model.save_pretrained(final_model_dir, safe_serialization=True)
    tokenizer.save_pretrained(final_model_dir)
    print(f"[Final Save] Model saved to: {final_model_dir}")

# ------------------------------------------------------------------------------
# Main Function with Scheduled Training
# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Scheduled Training for Parametric Memory Model")
    
    # Data parameters
    parser.add_argument("--data_path", type=str, 
        default="MemVerse/MemoryKB/Paramatric_Memory/paramemory.json",
        help="Path to training JSON data"
    )
    
    # Model parameters
    parser.add_argument("--seq_len", type=int, default=1024)
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--fp16", action="store_true")
    
    # Other parameters
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--out_dir", type=str, default="./ckpt_parametric_memory")
    parser.add_argument("--interval_hours", type=float, default=0.25, help="Training interval (hours)")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)

    # Scheduled training loop
    interval_seconds = args.interval_hours * 3600
    print(f"[Scheduler] Starting scheduled training (interval: {args.interval_hours}h)")
    print(f"[Scheduler] Next training will start at: {datetime.now() + timedelta(hours=args.interval_hours)}")

    while True:
        # Run training
        print("\n" + "="*80)
        print(f"[Scheduler] Starting new training cycle at {datetime.now()}")
        print("="*80)
        try:
            train(args)
        except Exception as e:
            print(f"[Error] Training failed: {str(e)}")
        
        # Wait for next interval
        print(f"\n[Scheduler] Waiting {args.interval_hours} hours for next training...")
        time.sleep(interval_seconds)

if __name__ == "__main__":
    from datetime import timedelta
    main()