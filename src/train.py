import time
import math
import os
import argparse
import traceback
import glob
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from model import ChimeraModel
from tokenizer import ChimeraTokenizer
from dataset import ChimeraDatasetStreamer
from logger import ChimeraLogger

class CheckpointManager:
    """Manages keeping only the Top-K checkpoints to save local disk space (e.g. max 5 files)."""
    def __init__(self, check_dir, max_keep=5):
        self.check_dir = check_dir
        self.max_keep = max_keep
        os.makedirs(self.check_dir, exist_ok=True)
        
    def save(self, model, step):
        ckpt_path = os.path.join(self.check_dir, f"chimera_step_{step}.pt")
        torch.save(model.state_dict(), ckpt_path)
        
        # Enforce max_keep
        files = glob.glob(os.path.join(self.check_dir, "chimera_step_*.pt"))
        if len(files) > self.max_keep:
            files.sort(key=os.path.getmtime)
            os.remove(files[0]) # Delete oldest

def train_loop(duration_hours: float):
    print(f"[System] Initiating Max-Level Training Rig for {duration_hours} hour(s)...")
    
    tokenizer = ChimeraTokenizer()
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    checkpoints_dir = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
    
    streamer = ChimeraDatasetStreamer(data_dir=data_dir, tokenizer=tokenizer, max_seq_length=512)
    model = ChimeraModel()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Advanced Hyperparameters
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=1000, T_mult=2)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.special_tokens["<PAD>"])
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())
    
    ckpt_manager = CheckpointManager(checkpoints_dir, max_keep=3)
    
    start_time = time.time()
    max_duration_seconds = duration_hours * 3600
    step = 0
    running_loss = 0.0
    accumulation_steps = 4 # Gradient Accumulation for larger effective batch size without OOM
    
    try:
        model.train()
        optimizer.zero_grad()
        
        for chunk in streamer.stream_curriculum():
            if time.time() - start_time > max_duration_seconds:
                print(f"\n[System] Reached requested training duration of {duration_hours} hour(s). Auto-pausing.")
                break
                
            x_seq = torch.tensor([chunk[:-1]], dtype=torch.long).to(device)
            y_seq = torch.tensor([chunk[1:]], dtype=torch.long).to(device)
            
            # Mixed Precision Forward Pass (autocast accelerates BitNet logic seamlessly on modern GPUs)
            with autocast(enabled=torch.cuda.is_available()):
                logits = model(x_seq)
                loss = criterion(logits.view(-1, logits.size(-1)), y_seq.view(-1))
                loss = loss / accumulation_steps
                
            loss_val = loss.item() * accumulation_steps
            
            # Autonomous Monitor: Loss Spike & NaN interception
            if math.isnan(loss_val) or loss_val > 20.0:
                err_msg = f"Training destabilized! Loss spiked to {loss_val}."
                print(f"[Error] {err_msg}")
                ChimeraLogger.log_failed_math("Training Loss Calculation", str(loss_val), "< 20.0 (Valid float)")
                print("[System] Pausing loop to prevent corrupted weights.")
                break
                
            scaler.scale(loss).backward()
            
            if (step + 1) % accumulation_steps == 0:
                # Gradient Clipping to prevent explosion
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
            
            running_loss += loss_val
            step += 1
            
            if step % 20 == 0:
                curr_lr = optimizer.param_groups[0]['lr']
                print(f"Step {step:05d} | Loss: {loss_val:.4f} | LR: {curr_lr:.2e} | Time: {(time.time() - start_time)/60:.1f} min")
                
            if step % 100 == 0:
                ckpt_manager.save(model, step)

        final_ckpt = os.path.join(checkpoints_dir, "chimera_latest.pt")
        torch.save(model.state_dict(), final_ckpt)
        print("[System] Training cycle ended gracefully. Latest checkpoint saved.")
        
    except Exception as e:
        tb = traceback.format_exc()
        ChimeraLogger.log_failed_code(type(e).__name__, str(e), tb)
        print("\n[CRITICAL ERROR] Pipeline crashed. Traceback routed to logs/failed_code.json.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration_hours", type=float, required=True, help="Hours to run")
    args = parser.parse_args()
    train_loop(args.duration_hours)
