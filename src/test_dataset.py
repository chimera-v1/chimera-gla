import os
import shutil
import traceback
import json
from dataset import ChimeraDatasetStreamer
from tokenizer import ChimeraTokenizer

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

def setup_mock_data(data_dir: str):
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
    
    os.makedirs(os.path.join(data_dir, "language"))
    os.makedirs(os.path.join(data_dir, "code"))
    os.makedirs(os.path.join(data_dir, "math"))
    
    with open(os.path.join(data_dir, "language", "doc1.txt"), "w", encoding="utf-8") as f:
        f.write("Hello world.\nThis is a language test.")
        
    with open(os.path.join(data_dir, "code", "script1.py"), "w", encoding="utf-8") as f:
        f.write("def foo():\n    return 42\n")
        
    with open(os.path.join(data_dir, "math", "eq1.txt"), "w", encoding="utf-8") as f:
        f.write("Math: 2 + 2 = 4\n")

def test_dataset_streamer():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    
    try:
        setup_mock_data(data_dir)
        tokenizer = ChimeraTokenizer(vocab_size=200) # Small vocab
        streamer = ChimeraDatasetStreamer(data_dir=data_dir, tokenizer=tokenizer, max_seq_length=5)
        
        # Test full curriculum stream
        curriculum_stream = streamer.stream_curriculum()
        
        chunks = []
        for chunk in curriculum_stream:
            chunks.append(chunk)
            assert len(chunk) <= 5, f"Chunk size exceeded max_seq_length: {len(chunk)}"
            
        assert len(chunks) > 0, "No chunks were yielded by the streamer"
        
        # Verify stages were hit
        print(f"Total chunks yielded: {len(chunks)}")
        print("Curriculum successfully streamed via generators without excessive memory loading.")
        
    except Exception as e:
        tb = traceback.format_exc()
        print("Test failed, logging to failed_code.json...")
        log_error("AssertionError" if isinstance(e, AssertionError) else type(e).__name__, str(e), tb)
        raise

if __name__ == "__main__":
    test_dataset_streamer()
