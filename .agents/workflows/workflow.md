---
description: # CHIMERA DEVELOPMENT WORKFLOW  This document outlines the standard 6-stage development lifecycle for implementing features, running training cycles, and verifying constraints[cite: 31, 32].
---


---

### Stage 1: Task Scoping & Memory Ingestion
1. **Rule Parsing:** Check `CHIMERA_RULES.md` to ensure design decisions comply with the Sub-200 MB RAM Promise and 100% Offline Mode[cite: 31, 32].
2. **Ledger Alignment:** Reference `docs/FEATURE_LEDGER.md` to confirm parameter thresholds, quantization boundaries, or module interfaces across Domains 1–6[cite: 31].
3. **Sprint Identification:** Confirm the target module within `PROGRESS.md` (e.g., Year 1, Months 1–2 foundation milestones)[cite: 31].

---

### Stage 2: Architectural Artifact & Blueprinting
1. **Artifact Generation:** Before modifying code, draft a technical design artifact specifying data structures, memory footprints, and dependency limits[cite: 31].
2. **Memory Profiling Estimate:** Ensure runtime allocations fit within the 200–500 MB RAM budget[cite: 31, 32].
3. **Graph-to-Code Mapping:** Establish structural references to prevent broken links across components[cite: 31].

---

### Stage 3: Implementation & Directory Routing
1. **Core Logic Development:** Write code strictly within defined module scopes[cite: 32]:
   * `src/tokenizer.py`: Implement the 20,000-word vocabulary with isolated digit handling and indentation tracking[cite: 31, 32].
   * `src/dataset.py`: Construct data streaming pipelines sequenced from natural language, to code, and then to mathematics[cite: 31, 32].
2. **Quantization & Building Blocks:** Restrict model weights to BitNet ternary values (-1, 0, +1) and configure RoPE, SwiGLU, and 6-block recurrent loops[cite: 31].
3. **Tag Isolation:** Implement Protected Reasoning Tags to isolate internal thinking steps from user-facing responses[cite: 31].

---

### Stage 4: Sandboxed Testing & Verification
1. **Isolated Execution:** Execute test suites inside the Local Code Sandbox without external network connections[cite: 31].
2. **Symbolic Branch Pruning (Symb-RATT):** Route any mathematical calculations through local code checkers; discard any reasoning paths that fail execution[cite: 31].
3. **Multi-Critic Evaluation (M-RAT):** Run generator-critic validation routines to detect syntax or logic flaws prior to committing changes[cite: 31].

---

### Stage 5: Autonomous Failure Logging & Error Recovery
1. **Direct Routing to `logs/`:** When an error occurs during runtime or evaluation, immediately append structured error objects to the matching endpoint[cite: 31, 32]:
   * `logs/hallucinations.json`: Log factual distortions or fabricated assertions[cite: 31, 32].
   * `logs/failed_math.json`: Log arithmetic errors, dimension mismatches, or calculation failures[cite: 31, 32].
   * `logs/failed_code.json`: Log syntax exceptions, failed sandbox assertions, and traceback dumps[cite: 31, 32].
2. **Self-Healing Loop:** Feed stack traces back into the agentic loop to generate iterative code patches until sandbox assertions pass[cite: 31].

---

### Stage 6: Progress & Checkpoint Synchronization
1. **Weight Checkpoints:** Save training iterations and model snapshots to `checkpoints/` (for scheduled 165 hours/week training runs)[cite: 31, 32].
2. **Git Discipline:** Verify that `.gitignore` strictly blocks all checkpoints, weights, and raw datasets in `data/` from repository commits[cite: 31, 32].
3. **Daily Update:** Update completed checklist items and next steps in `PROGRESS.md`[cite: 31].