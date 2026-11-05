import os
import time
import torch
import torch.nn.functional as F
from model import ChimeraModel
from tokenizer import ChimeraTokenizer
from vector_db import ChimeraVectorDB

# ============================================================
#  CHIMERA INFERENCE ENGINE v3.0
#  Features: Speculative Decoding, KV-Cache, Brain (Vector DB),
#            Context Window Manager, Persona System
# ============================================================

DEFAULT_PERSONA = "You are Chimera, a highly intelligent offline AI assistant. You are helpful, concise, and knowledgeable."

def build_tokenizer():
    """Builds the tokenizer vocabulary identically to how training does it."""
    tokenizer = ChimeraTokenizer()
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    
    # Scan all available text files to build vocab (same order as train.py)
    vocab_sources = [
        os.path.join(data_dir, "real_pretrain", "wikipedia_pretrain.txt"),
        os.path.join(data_dir, "language", "massive_instructions.txt"),
    ]
    
    vocab_text = ""
    for src in vocab_sources:
        if os.path.exists(src):
            with open(src, "r", encoding="utf-8") as f:
                for _ in range(10000):
                    line = f.readline()
                    if not line:
                        break
                    vocab_text += line
                    
    if vocab_text:
        tokenizer.train_from_text(vocab_text)
        
    return tokenizer


class ContextWindowManager:
    """
    Manages the chat history so it never overflows the model's 512-token context window.
    
    Strategy: Keep the system prompt + last N turns that fit within the budget.
    When history grows too long, older turns are compressed into a single-line summary
    and prepended as a "[Previous Context]" block, preserving continuity without overflow.
    """
    def __init__(self, tokenizer, max_tokens=480, persona=""):
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens  # Leave 32 tokens headroom for scratchpad
        self.persona = persona
        self.turns = []  # List of (role, text) tuples
        self.compressed_summary = ""  # Summary of old turns that were evicted
        
    def add_turn(self, role, text):
        """Add a new turn to the conversation."""
        self.turns.append((role, text))
        
    def build_prompt(self):
        """
        Builds the prompt string that fits within the token budget.
        Keeps recent turns and compresses old ones.
        """
        # Build the fixed prefix
        prefix = ""
        if self.persona:
            prefix += f"[System: {self.persona}]\n"
        if self.compressed_summary:
            prefix += f"[Previous Context: {self.compressed_summary}]\n"
            
        prefix_tokens = len(self.tokenizer.encode(prefix))
        remaining_budget = self.max_tokens - prefix_tokens
        
        # Walk backwards through turns, adding as many as fit
        selected_turns = []
        total_tokens = 0
        
        for role, text in reversed(self.turns):
            turn_str = f"{role}: {text}\n"
            turn_tokens = len(self.tokenizer.encode(turn_str))
            
            if total_tokens + turn_tokens > remaining_budget:
                break
            selected_turns.insert(0, (role, text))
            total_tokens += turn_tokens
            
        # If we had to drop turns, compress them into a summary
        kept_count = len(selected_turns)
        dropped_turns = self.turns[:len(self.turns) - kept_count]
        
        if dropped_turns:
            # Build a compressed summary of what was discussed
            topics = []
            for role, text in dropped_turns:
                if role == "User":
                    # Extract the key topic from the user's message
                    words = text.split()[:6]  # First 6 words
                    topics.append(" ".join(words))
            if topics:
                self.compressed_summary = "Previously discussed: " + "; ".join(topics[-3:])  # Keep last 3 topics
            
            # Evict the old turns permanently
            self.turns = self.turns[len(dropped_turns):]
        
        # Assemble final prompt
        prompt = prefix
        for role, text in selected_turns:
            prompt += f"{role}: {text}\n"
            
        return prompt


def generate_text(model, tokenizer, prompt, max_new_tokens=300, temperature=0.7, top_k=30, db=None):
    device = torch.device("cpu")
    model.eval()
    
    # 1. RAG Knowledge Injection (The External Brain)
    injected_prompt = prompt
    if db is not None:
        # Extract only the latest user query for the brain search
        user_query = prompt.split("User:")[-1].strip().rstrip("AI:").strip()
        if user_query:
            retrieved_fact = db.search(user_query)
            if retrieved_fact:
                print(f"\n[Brain] Fact injected: '{retrieved_fact[:80]}...'")
                injected_prompt = f"[System Context: {retrieved_fact}]\n{prompt}"
            
    # Encode prompt
    input_ids = tokenizer.encode(injected_prompt)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    
    generated_tokens = []
    
    print("AI: ", end="", flush=True)
    gen_start = time.time()
    
    with torch.no_grad():
        kv_cache = {}
        start_pos = 0
        current_tensor = input_tensor
        is_reasoning = False
        
        # --- PREFILL PHASE: Process the entire prompt through full attention ---
        prefill_logits = model(current_tensor, kv_cache=kv_cache, start_pos=0)
        start_pos = current_tensor.size(1)
        
        # Sample the first token from the prefill output
        first_logits = prefill_logits[0, -1, :]
        first_logits = first_logits / temperature
        top_k_logits, top_k_indices = torch.topk(first_logits, top_k)
        sparse = torch.full_like(first_logits, float('-inf'))
        sparse.scatter_(0, top_k_indices, top_k_logits)
        probs = F.softmax(sparse, dim=-1)
        first_token = torch.multinomial(probs, num_samples=1).item()
        
        generated_tokens.append(first_token)
        _print_token(tokenizer, first_token, is_reasoning)
        
        current_tensor = torch.tensor([[first_token]], dtype=torch.long, device=device)
        
        while len(generated_tokens) < max_new_tokens:
            
            # --- SPECULATIVE DECODING: DRAFT PHASE (Fast SSM) ---
            draft_tokens = []
            d_id = current_tensor[0, -1].item()
            draft_steps = 3
            
            for di in range(draft_steps):
                d_tensor = torch.tensor([[d_id]], dtype=torch.long, device=device)
                d_logits = model(d_tensor, use_ssm=True, start_pos=start_pos + di)
                d_id = torch.argmax(d_logits[0, -1, :]).item()
                draft_tokens.append(d_id)
                
            # --- SPECULATIVE DECODING: VERIFY PHASE (Full Attention) ---
            verify_ids = [current_tensor[0, -1].item()] + draft_tokens
            verify_tensor = torch.tensor([verify_ids], dtype=torch.long, device=device)
            
            # Single parallel forward pass through full attention
            v_logits = model(verify_tensor, kv_cache=kv_cache, start_pos=start_pos)
            
            accepted_tokens = []
            
            for i in range(len(verify_ids)):
                next_logits = v_logits[0, i, :].clone()
                
                # Repetition Penalty
                for token_id in set(generated_tokens + accepted_tokens):
                    if token_id < next_logits.size(0):
                        if next_logits[token_id] < 0:
                            next_logits[token_id] *= 1.2
                        else:
                            next_logits[token_id] /= 1.2
                        
                next_logits = next_logits / temperature
                
                # Top-K Sampling
                top_k_logits, top_k_indices = torch.topk(next_logits, min(top_k, next_logits.size(0)))
                sparse_logits = torch.full_like(next_logits, float('-inf'))
                sparse_logits.scatter_(0, top_k_indices, top_k_logits)
                probs = F.softmax(sparse_logits, dim=-1)
                true_next = torch.multinomial(probs, num_samples=1).item()
                
                if i < len(draft_tokens):
                    if true_next == draft_tokens[i]:
                        accepted_tokens.append(true_next)
                    else:
                        accepted_tokens.append(true_next)
                        break
                else:
                    accepted_tokens.append(true_next)
                    
            # --- KV CACHE ROLLBACK ---
            target_cache_len = start_pos + len(accepted_tokens)
            for layer_key in kv_cache:
                k_cache, v_cache = kv_cache[layer_key]
                kv_cache[layer_key] = (
                    k_cache[:, :, :target_cache_len, :],
                    v_cache[:, :, :target_cache_len, :]
                )
                
            # --- COMMIT ACCEPTED TOKENS ---
            should_stop = False
            for t in accepted_tokens:
                generated_tokens.append(t)
                char = tokenizer.decode([t])
                
                if "<REASONING_START>" in char:
                    is_reasoning = True
                    continue
                if "<REASONING_END>" in char:
                    is_reasoning = False
                    continue
                    
                if not is_reasoning:
                    print(char, end="", flush=True)
                    
                if "\nUser:" in char or "\nUser" in char:
                    should_stop = True
                    break
                    
            if should_stop:
                break
                
            # Advance position
            start_pos += len(accepted_tokens)
            current_tensor = torch.tensor([[accepted_tokens[-1]]], dtype=torch.long, device=device)
            
    elapsed = time.time() - gen_start
    tokens_per_sec = len(generated_tokens) / max(elapsed, 0.001)
    print(f"\n[{len(generated_tokens)} tokens | {tokens_per_sec:.1f} tok/s]\n")
    return tokenizer.decode(generated_tokens)


def _print_token(tokenizer, token_id, is_reasoning):
    """Helper to print a single token, respecting reasoning tag visibility."""
    char = tokenizer.decode([token_id])
    if "<REASONING_START>" in char or "<REASONING_END>" in char:
        return
    if not is_reasoning:
        print(char, end="", flush=True)


def main():
    print("=" * 50)
    print("  CHIMERA INFERENCE ENGINE v3.0")
    print("  Spec-Decode | KV-Cache | Brain | Persona")
    print("=" * 50)
    
    device = torch.device("cpu")
    checkpoints_dir = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
    latest_ckpt = os.path.join(checkpoints_dir, "chimera_latest.pt")
    
    # Build tokenizer with the same vocabulary as training
    print("[System] Building tokenizer vocabulary...")
    tokenizer = build_tokenizer()
    print(f"[System] Tokenizer ready. Vocabulary: {len(tokenizer.vocab)} tokens.")
    
    model = ChimeraModel()
    
    if os.path.exists(latest_ckpt):
        print(f"[System] Loading weights from: {latest_ckpt}")
        checkpoint = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(checkpoint, strict=False)
    else:
        print("[CRITICAL ERROR] No trained weights found in checkpoints directory.")
        return
        
    model.to(device)
    model.eval()
    
    print("[System] Booting Vector DB (External Brain)...")
    db = ChimeraVectorDB()
    
    # Initialize persona and context manager
    persona = DEFAULT_PERSONA
    ctx_manager = ContextWindowManager(tokenizer, max_tokens=480, persona=persona)
    
    print("[System] Model ready. Type 'exit' to quit.")
    print("[System] Commands: /persona <text> to change personality\n")
    
    while True:
        try:
            user_input = input("User: ")
            
            if user_input.lower() in ["exit", "quit"]:
                break
            if not user_input.strip():
                continue
                
            # --- COMMAND: /persona ---
            if user_input.strip().startswith("/persona"):
                new_persona = user_input.strip()[len("/persona"):].strip()
                if new_persona:
                    persona = new_persona
                    ctx_manager.persona = persona
                    print(f"[System] Persona updated to: '{persona}'\n")
                else:
                    print(f"[System] Current persona: '{persona}'")
                    print("[System] Usage: /persona You are a pirate who speaks in riddles\n")
                continue
                
            # Add user turn to context manager
            ctx_manager.add_turn("User", user_input)
            
            # Build the prompt with intelligent window management
            prompt = ctx_manager.build_prompt() + "AI:"
            
            # Generate response
            response = generate_text(model, tokenizer, prompt, db=db)
            
            # Clean the response and add AI turn
            clean_response = response.strip()
            ctx_manager.add_turn("AI", clean_response)
            
        except KeyboardInterrupt:
            print("\n[System] Shutting down gracefully.")
            break

if __name__ == "__main__":
    main()
