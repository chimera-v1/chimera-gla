import os
from typing import Generator, List
from tokenizer import ChimeraTokenizer

class ChimeraDatasetStreamer:
    """
    Chimera Dataset Streamer
    Features:
    - Zero network dependencies: strictly reads from local `data/` directory.
    - Generator-based streaming to ensure sub-200 MB RAM limit.
    - Phased Curriculum: Language -> Code -> Math.
    """
    def __init__(self, data_dir: str, tokenizer: ChimeraTokenizer, max_seq_length: int = 1024):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.curriculum_phases = ["language", "code", "math"]
        
    def _stream_file(self, file_path: str) -> Generator[List[int], None, None]:
        """
        Streams a single file line by line to maintain a tiny memory footprint.
        Yields encoded token lists.
        """
        if not os.path.exists(file_path):
            return
            
        with open(file_path, "r", encoding="utf-8") as f:
            buffer = []
            for line in f:
                encoded_line = self.tokenizer.encode(line)
                buffer.extend(encoded_line)
                
                # Yield when buffer reaches max_seq_length
                while len(buffer) >= self.max_seq_length:
                    yield buffer[:self.max_seq_length]
                    buffer = buffer[self.max_seq_length:]
                    
            # Yield any remaining tokens
            if buffer:
                yield buffer

    def stream_phase(self, phase_name: str) -> Generator[List[int], None, None]:
        """
        Streams all data files within a specific curriculum phase directory.
        """
        if phase_name not in self.curriculum_phases:
            raise ValueError(f"Unknown phase: {phase_name}")
            
        phase_dir = os.path.join(self.data_dir, phase_name)
        if not os.path.exists(phase_dir):
            # Create the directory structure if it doesn't exist for scaffolding
            os.makedirs(phase_dir, exist_ok=True)
            return
            
        for filename in os.listdir(phase_dir):
            file_path = os.path.join(phase_dir, filename)
            if os.path.isfile(file_path):
                yield from self._stream_file(file_path)

    def stream_curriculum(self) -> Generator[List[int], None, None]:
        """
        Master curriculum generator: Streams Language, then Code, then Math.
        """
        for phase in self.curriculum_phases:
            yield from self.stream_phase(phase)
