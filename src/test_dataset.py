import os
import shutil
import traceback
from dataset import ChimeraDatasetStreamer
from tokenizer import ChimeraTokenizer
from logger import ChimeraLogger

def setup_mock_data(data_dir: str):
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir, ignore_errors=True)
    
    os.makedirs(os.path.join(data_dir, "language"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "code"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "math"), exist_ok=True)
    
    with open(os.path.join(data_dir, "language", "doc1.txt"), "w", encoding="utf-8") as f:
        f.write("Hello world.\nThis is a language test.")
        
    with open(os.path.join(data_dir, "code", "script1.py"), "w", encoding="utf-8") as f:
        f.write("def foo():\n    return 42\n")
        
    with open(os.path.join(data_dir, "math", "eq1.txt"), "w", encoding="utf-8") as f:
        f.write("Math: 2 + 2 = 4\n")

def test_dataset_streamer():
    mock_data_dir = os.path.join(os.path.dirname(__file__), "..", "temp_mock_data")
    
    try:
        setup_mock_data(mock_data_dir)
        tokenizer = ChimeraTokenizer(vocab_size=200) # Small vocab
        streamer = ChimeraDatasetStreamer(data_dir=mock_data_dir, tokenizer=tokenizer, max_seq_length=5)
        
        # Test curriculum stream
        curriculum_stream = streamer.stream_curriculum()
        
        chunks = []
        for i, chunk in enumerate(curriculum_stream):
            chunks.append(chunk)
            assert len(chunk) <= 6, f"Chunk size exceeded max_seq_length+1: {len(chunk)}"
            if i >= 10:  # Sample 10 chunks safely
                break
            
        assert len(chunks) > 0, "No chunks were yielded by the streamer"
        
        print(f"Total chunks tested: {len(chunks)}")
        print("Curriculum successfully streamed via generators without excessive memory loading.")
        
    except Exception as e:
        tb = traceback.format_exc()
        print("Test failed, logging to failed_code.json...")
        ChimeraLogger.log_failed_code("AssertionError" if isinstance(e, AssertionError) else type(e).__name__, str(e), tb)
        raise
    finally:
        if os.path.exists(mock_data_dir):
            shutil.rmtree(mock_data_dir, ignore_errors=True)

if __name__ == "__main__":
    test_dataset_streamer()
