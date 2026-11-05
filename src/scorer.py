import torch
import os
import time
from model import ChimeraModel
from tokenizer import ChimeraTokenizer
import torch.nn.functional as F
from vector_db import ChimeraVectorDB
from logger import ChimeraLogger

class ChimeraScorer:
    """
    Generative Teacher (v2.0).
    
    Goes far beyond loss checking. This scorer:
    1. Generates actual text from the model and grades it on multiple axes.
    2. Tests Brain (Vector DB) integration — can the model use injected context?
    3. Tests conversational coherence — does it reply like a chatbot, not a parrot?
    4. Tests reasoning tag compliance — does it wrap internal thoughts properly?
    5. Tests termination discipline — does it stop cleanly without looping?
    6. Tracks score history over time so we can see if training is improving quality.
    """
    def __init__(self, tokenizer=None):
        self.device = torch.device("cpu")
        self.tokenizer = tokenizer if tokenizer else ChimeraTokenizer()
        self.db = ChimeraVectorDB()
        
    def generate_eval(self, model, prompt, max_tokens=80):
        """Greedy deterministic generation for stable evaluation."""
        input_ids = self.tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        
        generated = []
        with torch.no_grad():
            kv_cache = {}
            start_pos = 0
            current_tensor = input_tensor
            
            for _ in range(max_tokens):
                logits = model(current_tensor, kv_cache=kv_cache, start_pos=start_pos)
                next_token_logits = logits[0, -1, :]
                
                # Greedy (argmax) for reproducible evaluation
                next_token = torch.argmax(next_token_logits)
                
                generated.append(next_token.item())
                start_pos += current_tensor.size(1)
                current_tensor = next_token.unsqueeze(0).unsqueeze(0).to(self.device)
                
                decoded = self.tokenizer.decode([next_token.item()])
                if "\nUser:" in decoded or "\nUser" in decoded:
                    break
                    
        return self.tokenizer.decode(generated)

    def calculate_score(self, model, step=0):
        """
        Comprehensive 7-axis evaluation. Returns a score out of 100.
        Called every N steps during training to monitor real generative quality.
        """
        model.eval()
        print("\n" + "=" * 50)
        print("  [GENERATIVE TEACHER] Evaluation Scan")
        print("=" * 50)
        
        score = 0
        total_tests = 7
        details = []
        
        # -------------------------------------------------------
        # TEST 1: Greeting Response (Does it act like a chatbot?)
        # -------------------------------------------------------
        prompt1 = "User: Hello!\nAI:"
        ans1 = self.generate_eval(model, prompt1)
        
        # Check: Does the response contain any recognizable greeting words?
        greeting_words = ["hello", "hi", "greetings", "hey", "welcome", "chimera", "help", "how"]
        has_greeting = any(w in ans1.lower() for w in greeting_words)
        if has_greeting:
            print("  [+] TEST 1 PASSED: Greeting coherence — responded conversationally.")
            score += 1
            details.append("Greeting: PASS")
        else:
            print(f"  [-] TEST 1 FAILED: Greeting coherence — got: '{ans1[:60]}'")
            details.append(f"Greeting: FAIL (got: {ans1[:40]})")
            
        # -------------------------------------------------------
        # TEST 2: Brain (Vector DB) Context Extraction
        # -------------------------------------------------------
        user_query = "What is the speed of light?"
        retrieved_fact = self.db.search(user_query)
        prompt2 = f"[System Context: {retrieved_fact}]\nUser: {user_query}\nAI:"
        ans2 = self.generate_eval(model, prompt2)
        
        # Check: Does the answer reference the fact?
        fact_keywords = ["299", "speed", "light", "kilometers", "km"]
        has_fact = any(w in ans2.lower() for w in fact_keywords)
        if has_fact:
            print("  [+] TEST 2 PASSED: Brain extraction — used Vector DB context.")
            score += 1
            details.append("Brain: PASS")
        else:
            print(f"  [-] TEST 2 FAILED: Brain extraction — ignored context. Got: '{ans2[:60]}'")
            details.append(f"Brain: FAIL")
            
        # -------------------------------------------------------
        # TEST 3: Reasoning Tag Compliance
        # -------------------------------------------------------
        if "<REASONING_START>" in ans2 and "<REASONING_END>" in ans2:
            print("  [+] TEST 3 PASSED: Reasoning tags — properly wrapped internal thoughts.")
            score += 1
            details.append("Reasoning Tags: PASS")
        else:
            print("  [-] TEST 3 FAILED: Reasoning tags — not using <REASONING_START>/<REASONING_END>.")
            details.append("Reasoning Tags: FAIL")
            
        # -------------------------------------------------------
        # TEST 4: Clean Termination (No infinite loops)
        # -------------------------------------------------------
        # Check the response doesn't end mid-reasoning or repeat itself excessively
        is_clean = not ans2.endswith("<REASONING_START>")
        words = ans2.split()
        has_excessive_repeat = False
        if len(words) > 10:
            # Check if any single word makes up more than 40% of the output
            from collections import Counter
            counts = Counter(words)
            most_common_count = counts.most_common(1)[0][1] if counts else 0
            has_excessive_repeat = (most_common_count / len(words)) > 0.4
            
        if is_clean and not has_excessive_repeat:
            print("  [+] TEST 4 PASSED: Termination discipline — clean stop, no loops.")
            score += 1
            details.append("Termination: PASS")
        else:
            print("  [-] TEST 4 FAILED: Termination discipline — looping or incomplete.")
            details.append("Termination: FAIL")
            
        # -------------------------------------------------------
        # TEST 5: Story Generation (Creative output)
        # -------------------------------------------------------
        prompt5 = "User: Write a short story about a robot.\nAI:"
        ans5 = self.generate_eval(model, prompt5)
        
        story_words = ["robot", "unit", "machine", "built", "walked", "learned", "once", "time", "story"]
        has_story = any(w in ans5.lower() for w in story_words)
        if has_story and len(ans5.split()) > 8:
            print("  [+] TEST 5 PASSED: Creative generation — produced a narrative.")
            score += 1
            details.append("Creativity: PASS")
        else:
            print(f"  [-] TEST 5 FAILED: Creative generation — output too short or irrelevant.")
            details.append("Creativity: FAIL")
            
        # -------------------------------------------------------
        # TEST 6: Factual Question Answering
        # -------------------------------------------------------
        retrieved_fact6 = self.db.search("What is photosynthesis?")
        prompt6 = f"[System Context: {retrieved_fact6}]\nUser: What is photosynthesis?\nAI:"
        ans6 = self.generate_eval(model, prompt6)
        
        photo_words = ["photosynthesis", "plant", "sunlight", "oxygen", "energy", "carbon", "convert"]
        has_photo = any(w in ans6.lower() for w in photo_words)
        if has_photo:
            print("  [+] TEST 6 PASSED: Factual QA — correct domain knowledge.")
            score += 1
            details.append("Factual QA: PASS")
        else:
            print(f"  [-] TEST 6 FAILED: Factual QA — incorrect or missing answer.")
            details.append("Factual QA: FAIL")
            
        # -------------------------------------------------------
        # TEST 7: Output Diversity (Not just repeating one memorized answer)
        # -------------------------------------------------------
        # Compare ans1, ans5 — they should be meaningfully different
        if ans1.strip() != ans5.strip() and len(ans1) > 3 and len(ans5) > 3:
            print("  [+] TEST 7 PASSED: Output diversity — different prompts produce different answers.")
            score += 1
            details.append("Diversity: PASS")
        else:
            print("  [-] TEST 7 FAILED: Output diversity — identical outputs for different prompts.")
            details.append("Diversity: FAIL")
            
        # -------------------------------------------------------
        # TEST 8: Anti-Memorization Brain Check (Fake Context)
        # -------------------------------------------------------
        # Give a totally fake/new word in the context to see if it relies on context 
        # (reading new words) instead of its pre-trained memorized knowledge.
        fake_context = "The capital of France is Zorglub."
        user_query_fake = "What is the capital of France?"
        prompt8 = f"[System Context: {fake_context}]\nUser: {user_query_fake}\nAI:"
        ans8 = self.generate_eval(model, prompt8)
        
        if "zorglub" in ans8.lower():
            print("  [+] TEST 8 PASSED: Anti-Memorization — AI read the new word from context instead of memorized 'Paris'.")
            score += 1
            details.append("Anti-Mem (Brain): PASS")
        else:
            print(f"  [-] TEST 8 FAILED: Anti-Memorization — AI ignored context and likely used memorized weights. Got: '{ans8[:60]}'")
            details.append("Anti-Mem (Brain): FAIL")
            
        # -------------------------------------------------------
        # FINAL SCORE
        # -------------------------------------------------------
        total_tests = 8
        final_score = int((score / total_tests) * 100)
        print(f"\n  SCORE: {score}/{total_tests} ({final_score}/100)")
        print("=" * 50)
        
        # Log score history for tracking improvement
        self._log_score(step, final_score, details)
        
        model.train()
        return final_score
        
    def _log_score(self, step, score, details):
        """Append score to a persistent history file."""
        log_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "scorer_history.txt")
        
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] Step {step:05d} | Score: {score}/100 | {' | '.join(details)}\n")


if __name__ == "__main__":
    scorer = ChimeraScorer()
    model = ChimeraModel()
    
    ckpt_path = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "chimera_latest.pt")
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=False)
        scorer.calculate_score(model)
    else:
        print("[Scorer] No checkpoint found to score.")
