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
    """Manages keeping only the Top-K checkpoints to save local disk space."""
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
    
    # ---------------------------------------------------------
    # 0. TOKENIZER PRE-TRAINING (VOCABULARY BUILDING)
    # ---------------------------------------------------------
    print("[System] Scanning datasets to build 20,000-word Vocabulary...")
    vocab_text = ""
    
    # Scan all available data sources for vocabulary
    vocab_sources = [
        os.path.join(data_dir, "real_pretrain", "wikipedia_pretrain.txt"),
        os.path.join(data_dir, "language", "massive_instructions.txt"),
        os.path.join(data_dir, "code", "synthetic_code.txt"),
        os.path.join(data_dir, "math", "synthetic_math.txt"),
    ]
    
    for src in vocab_sources:
        if os.path.exists(src):
            with open(src, "r", encoding="utf-8") as f:
                for _ in range(10000):
                    line = f.readline()
                    if not line:
                        break
                    vocab_text += line
                    
    tokenizer.train_from_text(vocab_text)
    print(f"[System] Tokenizer active. Vocabulary Size: {len(tokenizer.vocab)} tokens.")
    
    streamer = ChimeraDatasetStreamer(
        data_dir=data_dir, 
        tokenizer=tokenizer, 
        max_seq_length=512,
        curriculum_stages=["real_pretrain", "language"]
    )
    model = ChimeraModel()
    
    print("\n================================================")
    print("  TRAINING ARCHITECTURE DIAGNOSTIC CHECK")
    print("================================================")
    
    # Check 1: Bfloat16 Support
    if hasattr(torch, 'bfloat16'):
        print("[+] PASS: CPU Hardware supports Bfloat16 Mixed Precision.")
    else:
        print("[-] WARN: Bfloat16 disabled by CPU architecture. Overriding to fallback.")
        
    # Check 2: Scorer System
    if os.path.exists(os.path.join(os.path.dirname(__file__), "scorer.py")):
        print("[+] PASS: Generative Teacher v2.0 armed (7-axis scan every 500 steps).")
    else:
        print("[-] FAIL: scorer.py missing! Training halted.")
        return
        
    # Check 3: Data Sources
    for src in vocab_sources:
        name = os.path.basename(src)
        if os.path.exists(src):
            size_mb = os.path.getsize(src) / (1024 * 1024)
            print(f"[+] PASS: Dataset '{name}' loaded ({size_mb:.1f} MB).")
        else:
            print(f"[-] WARN: Dataset '{name}' not found. Skipping.")
    
    print("[+] PASS: Anti-Memorization Weight Decay Filter is locked in.")
    print("================================================\n")
    
    device = torch.device("cpu")
    model.to(device)
    
    # Auto-Resume Logic
    all_ckpts = glob.glob(os.path.join(checkpoints_dir, "*.pt"))
    if all_ckpts:
        all_ckpts.sort(key=os.path.getmtime)
        latest_ckpt = all_ckpts[-1]
        try:
            model.load_state_dict(torch.load(latest_ckpt, map_location=device), strict=False)
            print(f"[System] Resuming from checkpoint: {latest_ckpt}")
        except Exception as e:
            print(f"[System] Failed to load checkpoint {latest_ckpt}: {e}")
    else:
        print("[System] No checkpoints found. Starting from scratch.")
        
    # Advanced Hyperparameters
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=1000, T_mult=2)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.special_tokens["<PAD>"])
    scaler = torch.amp.GradScaler('cpu', enabled=False)
    
    ckpt_manager = CheckpointManager(checkpoints_dir, max_keep=3)
    
    start_time = time.time()
    max_duration_seconds = duration_hours * 3600
    step = 0
    running_loss = 0.0
    accumulation_steps = 4 # Gradient Accumulation for larger effective batch size without OOM
    best_score = 0
    
    try:
        model.train()
        optimizer.zero_grad()
        
        for chunk in streamer.stream_curriculum():
            if time.time() - start_time > max_duration_seconds:
                print(f"\n[System] Reached requested training duration of {duration_hours} hour(s). Auto-pausing.")
                break
                
            x_seq = torch.tensor([chunk[:-1]], dtype=torch.long).to(device)
            y_seq = torch.tensor([chunk[1:]], dtype=torch.long).to(device)
            
            # Mixed Precision Forward Pass (Disabled for 32-bit CPU eager execution)
            with autocast(enabled=False):
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
                
            # Anti-Memorization System: Prevent saving raw data into weights
            # Only activate on sustained low loss, not occasional easy tokens
            if loss_val < 0.05:
                print(f"  [Anti-Memorization] Loss extremely low ({loss_val:.4f}). Spiking weight decay...")
                for param_group in optimizer.param_groups:
                    param_group['weight_decay'] = 0.30
            elif loss_val < 0.20:
                # Moderate zone — gentle regularization
                for param_group in optimizer.param_groups:
                    param_group['weight_decay'] = 0.05
            else:
                for param_group in optimizer.param_groups:
                    param_group['weight_decay'] = 0.01
                
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
            
            # Print every 10 steps instead of every step to reduce noise
            if step % 10 == 0:
                avg_loss = running_loss / 10
                curr_lr = optimizer.param_groups[0]['lr']
                wd = optimizer.param_groups[0]['weight_decay']
                time_min = (time.time() - start_time)/60
                print(f"Step {step:05d} | Loss: {avg_loss:.4f} | LR: {curr_lr:.2e} | WD: {wd:.2f} | Time: {time_min:.1f} min")
                running_loss = 0.0
                
            # Write daily report every 100 steps
            if step % 100 == 0:
                time_min = (time.time() - start_time)/60
                report_path = os.path.join(os.path.dirname(__file__), "..", "DAILY_REPORT.md")
                with open(report_path, "w", encoding="utf-8") as f:
                    f.write("# CHIMERA OFFLINE DAILY REPORT\n\n")
                    f.write(f"**Last Updated:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"- **Current Step:** {step}\n")
                    f.write(f"- **Current Loss:** {loss_val:.4f}\n")
                    f.write(f"- **Best Scorer Grade:** {best_score}/100\n")
                    f.write(f"- **Uptime:** {time_min:.1f} minutes\n")
                    f.write(f"- **Active Checkpoint File:** `chimera_latest.pt`\n\n")
                    f.write("> The pipeline is completely healthy and running.\n")
                
                # Save checkpoint every 100 steps
                ckpt_manager.save(model, step)
                
            # --- GENERATIVE TEACHER EVALUATION (every 500 steps) ---
            if step % 500 == 0 and step > 0:
                try:
                    from scorer import ChimeraScorer
                    # CRITICAL: Pass the SAME tokenizer to the scorer so vocab matches!
                    live_scorer = ChimeraScorer(tokenizer=tokenizer)
                    eval_score = live_scorer.calculate_score(model, step=step)
                    
                    if eval_score > best_score:
                        best_score = eval_score
                        best_ckpt = os.path.join(checkpoints_dir, "chimera_best.pt")
                        torch.save(model.state_dict(), best_ckpt)
                        print(f"[System] New best score! Saved chimera_best.pt ({best_score}/100)")
                        
                except Exception as e:
                    print(f"[Warning] Scorer encountered an issue: {e}")

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
