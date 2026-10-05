import re

class ChimeraTokenizer:
    """
    Max-Upgraded Chimera Tokenizer.
    Custom Byte-Pair Encoding (BPE) stub prioritizing the 20,000 vocab limit.
    Enforces strict digit isolation to prevent math hallucination and natively parses
    Python indentation loops.
    """
    def __init__(self, vocab_size=20000):
        self.vocab_size = vocab_size
        
        # Reserved Control & Structural Tokens (Indices 0-9)
        self.special_tokens = {
            "<PAD>": 0,
            "<UNK>": 1,
            "<EOS>": 2,
            "<BOS>": 3,
            "<think>": 4,
            "</think>": 5,
            "<INDENT_2>": 6,
            "<INDENT_4>": 7,
            "<INDENT_8>": 8
        }
        
        self.vocab = self.special_tokens.copy()
        
        # Seed ASCII characters to prevent unknown character dropping
        for i in range(256):
            char = chr(i)
            if char not in self.vocab:
                self.vocab[char] = len(self.vocab)
                
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
                
    def train_from_text(self, text: str):
        """Simulated BPE training pass to build vocab dynamically up to vocab_size."""
        text = self._isolate_digits(text)
        text = self._format_indents(text)
        for word in text.split():
            if word not in self.vocab and len(self.vocab) < self.vocab_size:
                idx = len(self.vocab)
                self.vocab[word] = idx
                self.inverse_vocab[idx] = word
                
    def _isolate_digits(self, text: str) -> str:
        """
        Forces structural separation of digits (e.g., '123' -> '1 2 3')
        Ensures the model learns fundamental arithmetic via place values, not memorized blobs.
        """
        return re.sub(r'(\d)', r' \1 ', text)

    def _format_indents(self, text: str) -> str:
        """
        Compresses Pythonic whitespace blocks into single, semantic tokens to save sequence length.
        """
        text = text.replace("        ", " <INDENT_8> ")
        text = text.replace("    ", " <INDENT_4> ")
        text = text.replace("  ", " <INDENT_2> ")
        return text

    def encode(self, text: str) -> list:
        if not text:
            return []
            
        text = self._isolate_digits(text)
        text = self._format_indents(text)
        
        tokens = []
        words = text.split()
        
        for word in words:
            if word in self.special_tokens:
                tokens.append(self.vocab[word])
            else:
                # Basic sub-word fallback
                for char in word:
                    tokens.append(self.vocab.get(char, self.vocab["<UNK>"]))
        return tokens

    def decode(self, tokens: list) -> str:
        output = ""
        for t in tokens:
            if t in self.inverse_vocab:
                token_str = self.inverse_vocab[t]
                if token_str == "<INDENT_2>":
                    output += "  "
                elif token_str == "<INDENT_4>":
                    output += "    "
                elif token_str == "<INDENT_8>":
                    output += "        "
                elif token_str in ["<think>", "</think>", "<PAD>", "<UNK>", "<EOS>", "<BOS>"]:
                    pass # Ignore rendering structural reasoning tags in final output text
                else:
                    output += token_str
        return output.strip()
