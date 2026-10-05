import traceback
from tokenizer import ChimeraTokenizer
from logger import ChimeraLogger

def test_tokenizer():
    try:
        tok = ChimeraTokenizer(vocab_size=20000)
        
        # Test 1: Isolated digits
        text = "Math: 1984 + 42"
        tok.train_from_text(text)
        encoded = tok.encode(text)
        
        # "1", "9", "8", "4" should be separate
        decoded = tok.decode(encoded)
        
        # Checking digit existence
        assert "1" in tok.vocab, "Digit 1 not in vocab"
        assert "9" in tok.vocab, "Digit 9 not in vocab"
        
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
        reasoning_text = "<think> I am thinking </think> Final Answer"
        tok.train_from_text(reasoning_text)
        encoded_reasoning = tok.encode(reasoning_text)
        tokens_reasoning = [tok.inverse_vocab.get(i) for i in encoded_reasoning]
        assert "<think>" in tokens_reasoning, "Reasoning tag start failed"
        assert "</think>" in tokens_reasoning, "Reasoning tag end failed"
        
        print("All tests passed successfully.")
        
    except Exception as e:
        tb = traceback.format_exc()
        print("Test failed, logging to failed_code.json...")
        ChimeraLogger.log_failed_code("AssertionError" if isinstance(e, AssertionError) else type(e).__name__, str(e), tb)
        raise

if __name__ == "__main__":
    test_tokenizer()
