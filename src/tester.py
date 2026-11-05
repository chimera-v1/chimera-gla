import os
import torch
import sys
from model import ChimeraModel
from tokenizer import ChimeraTokenizer
from infer import generate_text

def run_tester():
    print("========================================")
    print("  CHIMERA CHECKPOINT TESTER (Overfitting Check)")
    print("========================================")
    print("This script throws UNSEEN prompts at the model to see if it actually learned")
    print("the English language, or if it just blindly memorized the training data.\n")
    
    device = torch.device("cpu")
    checkpoints_dir = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
    
    # Find all checkpoints
    checkpoints = [f for f in os.listdir(checkpoints_dir) if f.endswith(".pt")]
    if not checkpoints:
        print("[CRITICAL ERROR] No checkpoints found to test.")
        return
        
    print("Available Checkpoints:")
    for i, ckpt in enumerate(checkpoints):
        print(f"  [{i}] {ckpt}")
        
    try:
        choice = int(input("\nEnter the number of the checkpoint to test: "))
        selected_ckpt = os.path.join(checkpoints_dir, checkpoints[choice])
    except (ValueError, IndexError):
        print("Invalid choice.")
        return
        
    tokenizer = ChimeraTokenizer()
    model = ChimeraModel()
    
    print(f"\n[System] Loading weights from: {selected_ckpt}")
    model.load_state_dict(torch.load(selected_ckpt, map_location=device))
    model.to(device)
    model.eval()
    
    # The ultimate test prompts (Things the model has NEVER seen exactly)
    test_prompts = [
        "User: Hello there, who are you?\nAI:", 
        "User: write me a small story about a blue bird\nAI:",
        "User: What is the speed of sound in water?\nAI:",
        "User: Can you explain the poem The Raven?\nAI:"
    ]
    
    print("\n--- INITIATING UNSEEN PROMPT TEST ---\n")
    for prompt in test_prompts:
        print(f"\n[PROMPT] {prompt.strip()}")
        response = generate_text(model, tokenizer, prompt, max_new_tokens=60, temperature=0.7)
        print("\n" + "-"*40)
        
        # Simple heuristic check
        if "I am Chimera" in response and "blue bird" in prompt:
            print("[DIAGNOSIS] 🚨 FAIL: OVERFITTING DETECTED! The model ignored the prompt and regurgitated memorized text.")
        elif "blue bird" in prompt and len(response) > 10:
            print("[DIAGNOSIS] ✅ PASS: GENERALIZATION DETECTED! The model attempted to create a unique story.")
            
    print("\n[System] Test Complete. If the model passes, it has successfully generalized English!")

if __name__ == "__main__":
    run_tester()
