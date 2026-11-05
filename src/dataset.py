import os
import gc
from typing import Generator, List

class ChimeraDatasetStreamer:
    """
    Max-Upgraded Dataset Streamer.
    Strict Curriculum Sequencing (Language -> Code -> Math).
    Uses heavy generator-based stream architecture yielding tightly packed chunks
    to enforce the Sub-200 MB RAM Promise. Keeps maximum active memory buffer ~5-10 MB.
    """
    def __init__(self, data_dir: str, tokenizer, max_seq_length: int = 512, curriculum_stages: List[str] = None):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.curriculum = curriculum_stages if curriculum_stages is not None else ["real_pretrain", "language", "code", "math"]

    def _stream_file(self, file_path: str) -> Generator[str, None, None]:
        """Reads extremely large files line-by-line via memory-safe IO generator."""
        if not os.path.exists(file_path):
            return
            
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                yield line.strip()

    def stream_curriculum(self) -> Generator[List[int], None, None]:
        """
        Flows continuously through the domains.
        Packs strings tightly into fixed sequence length tensors.
        Triggers garbage collection (gc.collect()) aggressively.
        """
        import itertools
        for domain in itertools.cycle(self.curriculum):
            domain_path = os.path.join(self.data_dir, domain)
            if not os.path.exists(domain_path):
                continue
                
            for filename in os.listdir(domain_path):
                file_path = os.path.join(domain_path, filename)
                
                buffer = []
                for line in self._stream_file(file_path):
                    if not line:
                        continue
                        
                    tokens = self.tokenizer.encode(line)
                    buffer.extend(tokens)
                    
                    # Yield chunks of precisely `max_seq_length + 1` for next-token prediction
                    while len(buffer) >= self.max_seq_length + 1:
                        chunk = buffer[:self.max_seq_length + 1]
                        buffer = buffer[self.max_seq_length + 1:]
                        yield chunk
                
                # If there are leftovers, pad them to exact length and yield
                if len(buffer) > 0:
                    pad_length = (self.max_seq_length + 1) - len(buffer)
                    chunk = buffer + [0] * pad_length
                    yield chunk

                # Force memory cleanup after every single file processed
                buffer.clear()
                gc.collect()
