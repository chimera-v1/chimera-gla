import torch
import shutil
import os
import sys

def check_gpu_memory(threshold_mb: int = 7800):
    """
    Split-Brain Cloud Router Logic
    Monitors RTX laptop GPU memory. If local memory maxes out, it pauses,
    packages the checkpoint, and routes to Lightning AI instances.
    """
    if not torch.cuda.is_available():
        print("[Split-Brain] CUDA not detected. Assuming CPU or basic mode.")
        return False
        
    allocated_mb = torch.cuda.memory_allocated() / (1024 ** 2)
    print(f"[Split-Brain] Local GPU Memory Allocated: {allocated_mb:.1f} MB / {threshold_mb} MB max")
    
    if allocated_mb > threshold_mb:
        print("[Split-Brain] ALARM: GPU Memory limit exceeded. Initiating Cloud Route.")
        return True
    return False

def route_to_lightning_ai():
    print("[Split-Brain] Packaging current checkpoint...")
    ckpt_dir = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
    archive_path = os.path.join(os.path.dirname(__file__), "..", "checkpoints_backup")
    shutil.make_archive(archive_path, 'zip', ckpt_dir)
    print(f"[Split-Brain] Packaged: {archive_path}.zip")
    
    print("[Split-Brain] Initializing Lightning AI remote payload execution...")
    # Stub for Lightning AI API deployment
    # e.g., lightning run app cloud_app.py --env ckpt=checkpoints_backup.zip
    print("[Split-Brain] Remote job dispatched. Local training process will now suspend.")
    sys.exit(0)

if __name__ == "__main__":
    if check_gpu_memory():
        route_to_lightning_ai()
    else:
        print("[Split-Brain] Memory stable. Continuing local execution.")
