import os
import json
from typing import List, Dict

class ChimeraTokenizer:
    """
    Chimera Tokenizer: Custom 20,000-word vocabulary tokenizer.
    Features:
    - Sub-200 MB memory footprint.
    - Isolated digit handling (character by character) to prevent arithmetic breakdown.
    - Code indentation tracking (captures 2-space, 4-space, 8-space, and tabs as single tokens).
    - Protected Reasoning Tags to isolate internal thinking.
    """
    
    SPECIAL_TOKENS = {
        "<PAD>": 0,
        "<UNK>": 1,
        "<BOS>": 2,
        "<EOS>": 3,
        "<REASONING_START>": 4,
        "<REASONING_END>": 5,
        "<INDENT_2>": 6,
        "<INDENT_4>": 7,
        "<INDENT_8>": 8,
        "<TAB>": 9
    }
    
    def __init__(self, vocab_size: int = 20000):
        self.vocab_size = vocab_size
        self.vocab: Dict[str, int] = self.SPECIAL_TOKENS.copy()
        self.inverse_vocab: Dict[int, str] = {v: k for k, v in self.vocab.items()}
        self.current_id = len(self.vocab)
        
        # Pre-populate single character digits to enforce isolated digit handling
        for digit in "0123456789":
            self._add_token(digit)
            
    def _add_token(self, token: str) -> int:
        if token not in self.vocab and len(self.vocab) < self.vocab_size:
            self.vocab[token] = self.current_id
            self.inverse_vocab[self.current_id] = token
            self.current_id += 1
        return self.vocab.get(token, self.SPECIAL_TOKENS["<UNK>"])

    def train_from_text(self, text: str):
        """
        Trains the tokenizer up to the 20k word limit based on local text data.
        """
        tokens = self._pre_tokenize(text)
        for t in tokens:
            if len(self.vocab) >= self.vocab_size:
                break
            self._add_token(t)

    def _pre_tokenize(self, text: str) -> List[str]:
        """
        Pre-tokenization step that handles code indentation, isolates digits,
        and splits by words/punctuation.
        """
        # Replace indents first
        text = text.replace("        ", " <INDENT_8> ")
        text = text.replace("    ", " <INDENT_4> ")
        text = text.replace("  ", " <INDENT_2> ")
        text = text.replace("\t", " <TAB> ")
        
        # Protect reasoning tags
        text = text.replace("<REASONING_START>", " <REASONING_START> ")
        text = text.replace("<REASONING_END>", " <REASONING_END> ")

        raw_tokens = text.split()
        final_tokens = []
        
        for token in raw_tokens:
            if token in self.SPECIAL_TOKENS:
                final_tokens.append(token)
            else:
                # Isolate digits: any digit becomes its own token
                current_word = ""
                for char in token:
                    if char.isdigit():
                        if current_word:
                            final_tokens.append(current_word)
                            current_word = ""
                        final_tokens.append(char)
                    elif not char.isalnum():
                        # split punctuation
                        if current_word:
                            final_tokens.append(current_word)
                            current_word = ""
                        final_tokens.append(char)
                    else:
                        current_word += char
                if current_word:
                    final_tokens.append(current_word)

        return final_tokens

    def encode(self, text: str) -> List[int]:
        tokens = self._pre_tokenize(text)
        return [self.vocab.get(t, self.SPECIAL_TOKENS["<UNK>"]) for t in tokens]

    def decode(self, token_ids: List[int]) -> str:
        tokens = [self.inverse_vocab.get(tid, "<UNK>") for tid in token_ids]
        
        result = []
        for t in tokens:
            if t == "<INDENT_8>":
                result.append("        ")
            elif t == "<INDENT_4>":
                result.append("    ")
            elif t == "<INDENT_2>":
                result.append("  ")
            elif t == "<TAB>":
                result.append("\t")
            elif t == "<REASONING_START>":
                result.append("<REASONING_START>")
            elif t == "<REASONING_END>":
                result.append("<REASONING_END>")
            elif t.isalnum():
                result.append(f" {t}" if result and result[-1] not in ("        ", "    ", "  ", "\t", "<REASONING_START>") else t)
            else:
                result.append(t)
                
        # Basic cleanup for spaces
        out_text = "".join(result).strip()
        return out_text

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"vocab": self.vocab}, f, indent=2)

    def load(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Vocabulary file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.vocab = data.get("vocab", {})
            self.inverse_vocab = {v: k for k, v in self.vocab.items()}
            self.current_id = len(self.vocab)
