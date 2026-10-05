import traceback
import json
import os
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

def test_tokenizer():
    try:
        tok = ChimeraTokenizer(vocab_size=20000)
        
        # Test 1: Isolated digits
        text = "Math: 1984 + 42"
        tok.train_from_text(text)
        encoded = tok.encode(text)
        
        # "1", "9", "8", "4" should be separate
        decoded = tok.decode(encoded)
        
        # Checking digit IDs
        assert tok.vocab["1"] == 11, "Digit 1 not at expected vocab ID"
        assert tok.vocab["9"] == 19, "Digit 9 not at expected vocab ID"
        
        # Ensure digits were split
        tokens_str = [tok.inverse_vocab.get(i) for i in encoded]
        assert "1" in tokens_str and "9" in tokens_str and "8" in tokens_str and "4" in tokens_str, "Digits were not isolated"
        
        # Test 2: Indentation
        code_text = "def foo():\n    return 42"
        tok.train_from_text(code_text)
        encoded_code = tok.encode(code_text)
        tokens_code = [tok.inverse_vocab.get(i) for i in encoded_code]
        assert "<INDENT_4>" in tokens_code, "4-space indent not captured as token"
        
        # Test 3: Protected tags
        reasoning_text = "<REASONING_START> I am thinking <REASONING_END> Final Answer"
        tok.train_from_text(reasoning_text)
        encoded_reasoning = tok.encode(reasoning_text)
        tokens_reasoning = [tok.inverse_vocab.get(i) for i in encoded_reasoning]
        assert "<REASONING_START>" in tokens_reasoning, "Reasoning tag start failed"
        assert "<REASONING_END>" in tokens_reasoning, "Reasoning tag end failed"
        
        print("All tests passed successfully.")
        
    except Exception as e:
        tb = traceback.format_exc()
        print("Test failed, logging to failed_code.json...")
        log_error("AssertionError" if isinstance(e, AssertionError) else type(e).__name__, str(e), tb)
        raise

if __name__ == "__main__":
    test_tokenizer()
