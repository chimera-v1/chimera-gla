import torch
import traceback
import json
import os
from model import ChimeraModel

def log_error(err_type: str, msg: str, tb: str):
    log_path = os.path.join(os.path.dirname(__file__), "..", "logs", "failed_code.json")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    error_data = {
        "type": err_type,
        "message": msg,
        "traceback": tb
    }
    
    logs = []
    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as f:
            try:
                logs = json.load(f)
            except json.JSONDecodeError:
                pass
                
    logs.append(error_data)
    
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2)

def test_model_architecture():
    try:
        # Initialize the 45M backbone
        model = ChimeraModel(vocab_size=20000, dim=512, num_heads=8, num_layers=6)
        
        # Mock input sequence (Batch Size = 2, Sequence Length = 64)
        x = torch.randint(0, 20000, (2, 64))
        
        # Test standard forward pass (incorporates Recurrent Loops, BitNet, MoD, MLA)
        logits = model(x, use_ssm=False)
        
        # Logits should be shape [Batch, Sequence_Length, Vocab_Size]
        assert logits.shape == (2, 64, 20000), f"Standard forward pass failed. Shape was {logits.shape}"
        
        # Test SSM Mode bypass
        logits_ssm = model(x, use_ssm=True)
        assert logits_ssm.shape == (2, 64, 20000), f"SSM mode failed. Shape was {logits_ssm.shape}"
        
        # Test reasoning tag insertion utility
        tagged_seq = model.insert_reasoning_tags([100, 200, 300])
        assert tagged_seq[0] == 4 and tagged_seq[-1] == 5, "Reasoning tags not correctly wrapped."
        
        # Memory Check calculation
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total Framework Parameters: {total_params:,}")
        
        print("Model architecture verification complete. All pathways executed successfully.")
        
    except Exception as e:
        tb = traceback.format_exc()
        print("Model test failed, logging to failed_code.json...")
        log_error("AssertionError" if isinstance(e, AssertionError) else type(e).__name__, str(e), tb)
        raise

if __name__ == "__main__":
    test_model_architecture()
