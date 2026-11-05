import torch
import torch.nn.functional as F
import json
import os

class ChimeraVectorDB:
    def __init__(self, db_path="data/knowledge_graph.json"):
        self.db_path = db_path
        self.memory_bank = []
        self.embeddings = None
        self.vocab = {} # Simple word-to-index mapping for basic TF-IDF style embedding
        
        # Initialize default facts if db doesn't exist
        if not os.path.exists(db_path):
            self._initialize_default_knowledge()
        else:
            self.load_db()
            
    def _initialize_default_knowledge(self):
        default_facts = [
            "Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to create oxygen and energy in the form of sugar.",
            "The speed of light in a vacuum is approximately 299,792 kilometers per second.",
            "Water boils at 100 degrees Celsius at sea level.",
            "The capital of France is Paris, known for the Eiffel Tower.",
            "Artificial Intelligence (AI) is the simulation of human intelligence processes by machines, especially computer systems.",
            "A blue bird represents happiness and joy in many cultures across the world."
        ]
        
        # Build vocabulary
        idx = 0
        for fact in default_facts:
            for word in fact.lower().split():
                if word not in self.vocab:
                    self.vocab[word] = idx
                    idx += 1
                    
        self.memory_bank = default_facts
        self.embeddings = self._embed_batch(default_facts)
        self.save_db()

    def _embed_text(self, text):
        """Ultra-fast local embedding generation without internet."""
        vec = torch.zeros(len(self.vocab))
        words = text.lower().split()
        for word in words:
            if word in self.vocab:
                vec[self.vocab[word]] += 1.0
        # L2 Normalize
        if vec.sum() > 0:
            vec = F.normalize(vec, p=2, dim=0)
        return vec

    def _embed_batch(self, texts):
        vecs = [self._embed_text(t) for t in texts]
        return torch.stack(vecs)

    def search(self, query, top_k=1):
        """Retrieves the most relevant memory from the external brain."""
        if self.embeddings is None or len(self.memory_bank) == 0:
            return ""
            
        query_vec = self._embed_text(query)
        if query_vec.sum() == 0:
            return "" # Unknown words
            
        # Cosine Similarity
        similarities = torch.matmul(self.embeddings, query_vec)
        best_idx = torch.argmax(similarities).item()
        
        if similarities[best_idx] > 0.1: # Confidence threshold
            return self.memory_bank[best_idx]
        return ""

    def add_memory(self, text):
        """Dynamically learn new facts and store them permanently."""
        # Update vocab
        new_words = False
        for word in text.lower().split():
            if word not in self.vocab:
                self.vocab[word] = len(self.vocab)
                new_words = True
                
        if new_words:
            # Recompute all embeddings if vocab expands
            self.embeddings = self._embed_batch(self.memory_bank + [text])
        else:
            new_vec = self._embed_text(text).unsqueeze(0)
            self.embeddings = torch.cat([self.embeddings, new_vec], dim=0)
            
        self.memory_bank.append(text)
        self.save_db()

    def save_db(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        data = {
            "vocab": self.vocab,
            "memory_bank": self.memory_bank
        }
        with open(self.db_path, "w") as f:
            json.dump(data, f)
            
    def load_db(self):
        with open(self.db_path, "r") as f:
            data = json.load(f)
        self.vocab = data["vocab"]
        self.memory_bank = data["memory_bank"]
        self.embeddings = self._embed_batch(self.memory_bank)
