import time
import math
import os
import argparse
import traceback
import torch
import torch.nn as nn
from model import ChimeraModel
from tokenizer import ChimeraTokenizer
from dataset import ChimeraDatasetStreamer
from logger import ChimeraLogger

def train_loop(duration_hours: float):
    print(f"[System] Initiating Chimera Training Loop for {duration_hours} hour(s)...")
    
    # Initialize Core Components
    tokenizer = ChimeraTokenizer()
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    checkpoints_dir = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    streamer = ChimeraDatasetStreamer(data_dir=data_dir, tokenizer=tokenizer, max_seq_length=512)
    model = ChimeraModel()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()
    
    start_time = time.time()
    max_duration_seconds = duration_hours * 3600
    
    step = 0
    running_loss = 0.0
    
    try:
        model.train()
        # Curriculum: Language -> Code -> Math
        for chunk in streamer.stream_curriculum():
            if time.time() - start_time > max_duration_seconds:
                print(f"\n[System] Reached requested training duration of {duration_hours} hour(s). Auto-pausing.")
                break
                
            # Prepare inputs
            # chunk is List[int]
            x_seq = torch.tensor([chunk[:-1]], dtype=torch.long).to(device)
            y_seq = torch.tensor([chunk[1:]], dtype=torch.long).to(device)
            
            optimizer.zero_grad()
            logits = model(x_seq)
            
            # Loss calculation
            # logits: (Batch, SeqLen, Vocab) -> view for CrossEntropy
            loss = criterion(logits.view(-1, logits.size(-1)), y_seq.view(-1))
            
            # Autonomous Monitor: Loss Spike & NaN checks
            loss_val = loss.item()
            if math.isnan(loss_val) or loss_val > 20.0:
                err_msg = f"Training destabilized! Loss spiked to {loss_val}."
                print(f"[Error] {err_msg}")
                ChimeraLogger.log_failed_math(expression="Training Loss Calculation", incorrect_output=str(loss_val), expected_output="< 20.0 (Valid float)")
                # Autonomous pause
                print("[System] Pausing loop to prevent checkpoint corruption.")
                break
                
            loss.backward()
            optimizer.step()
            
            running_loss += loss_val
            step += 1
            
            if step % 10 == 0:
                print(f"Step {step} | Loss: {loss_val:.4f} | Time Elapsed: {(time.time() - start_time)/60:.1f} min")
                
            # Periodic Checkpointing (Simulated every 100 steps for now)
            if step % 100 == 0:
                ckpt_path = os.path.join(checkpoints_dir, f"chimera_step_{step}.pt")
                torch.save(model.state_dict(), ckpt_path)
                print(f"[Checkpoint] Saved weights to {ckpt_path} (Size: ~10MB)")

        # Final save
        final_ckpt = os.path.join(checkpoints_dir, "chimera_latest.pt")
        torch.save(model.state_dict(), final_ckpt)
        print("[System] Training cycle ended gracefully. Latest checkpoint saved.")
        
    except Exception as e:
        tb = traceback.format_exc()
        ChimeraLogger.log_failed_code(type(e).__name__, str(e), tb)
        print("\n[CRITICAL ERROR] Pipeline crashed. Traceback routed to logs/failed_code.json.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chimera Offline Agentic Training Loop")
    parser.add_argument("--duration_hours", type=float, required=True, help="Number of hours to train for this session")
    args = parser.parse_args()
    
    train_loop(args.duration_hours)
