# CHIMERA: SYSTEM RULES & OPERATING CONSTRAINTS

Always parse and adhere to these directives before generating code, running terminal commands, or altering workspace configurations.

---

## 1. Non-Negotiable System Ceilings
* **Sub-200 MB RAM Promise:** The entire running stack—including model weights, inference runtime, local database, and shell—must strictly operate within 200–500 MB of memory[cite: 31, 32]. Never introduce bulky runtimes or unquantized weight caches[cite: 31].
* **100% Offline Mandate:** The model, search indexing, code sandbox, and math parser must function with Wi-Fi completely disabled[cite: 31, 32]. No runtime network calls are permitted[cite: 31].
* **Desktop Shell Boundary:** The desktop UI must use a native, lightweight framework under 20 MB, strictly banning heavy dependencies such as Electron[cite: 31].

---

## 2. Core Architecture Rules
* **45-Million-Parameter Backbone:** All model architectures must fit a compact 45M parameter design runnable on commodity laptops without a dedicated GPU[cite: 31].
* **BitNet Ternary Quantization:** All weight representations must quantize strictly to $\{-1, 0, +1\}$, reducing raw weight storage to 9–12 MB and substituting additions for heavy floating-point operations[cite: 31].
* **Recurrent Weight-Sharing Loop:** Core reasoning must reuse the same six transformer building blocks three times sequentially to emulate deep-stage compute without increasing parameter count[cite: 31].
* **Attention Constraints:** Use Multi-Head Latent Attention (MLA) with Logic-Gating and Sparse Hybrid attention locked to the top 32 relevant tokens to compress context memory[cite: 31].
* **ThinKV & Scratchpad:** Thinking memory must use ThinKV compression (up to 95% reduction) and write intermediate states to a dedicated 32-Slot Scratchpad[cite: 31].
* **Protected Reasoning Tags:** All internal chains of thought must be enclosed inside reserved tags to strictly isolate reasoning from output deliverables[cite: 31].
* **State Space Model (SSM) Fallback:** Support CPU-only execution without graphics acceleration for cross-device operability[cite: 31].

---

## 3. Knowledge, Reasoning & Validation Rules
* **Micro-GraphRAG:** Maintain local graph relations directly on-device using markdown and vault parsers[cite: 31].
* **Local Embeddings:** All text representations must use the offline 22 MB embedding engine[cite: 31].
* **Symbolic Branch Pruning (Symb-RATT):** All calculations must pass through a strict local code evaluator; any branch with arithmetic contradictions must be pruned immediately[cite: 31].
* **Multi-Critic Scratchpad (M-RAT):** Internal logic must pit generator and critic personas against each other to evaluate correctness before final response generation[cite: 31].
* **Keyless Web Scraper Fallback:** Web retrieval is restricted to anonymous fallback queries strictly when local context stores are deficient[cite: 31].

---

## 4. Directory & File Routing Map
* **`chimera-gla/` (Root Directory):** Root workspace[cite: 32].
* **`.gitignore`:** Blocks heavy data and weight checkpoints from uploading to GitHub to enforce repository efficiency and the 200–500 MB memory rule[cite: 31, 32].
* **`docs/FEATURE_LEDGER.md`:** Reference file detailing all 72 technical specifications across Domains 1–6 and the orbital roadmap[cite: 31]. Consult before architectural revisions.
* **`src/`:** Source logic repository[cite: 32].
  * `src/tokenizer.py`: Purpose-built 20,000-word vocabulary handling numbers and whitespace indentation to prevent AI math and code errors[cite: 31, 32].
  * `src/dataset.py`: Streams training datasets, sequenced from natural language, to code, to mathematics[cite: 31, 32].
* **`data/`:** Stores offline local datasets securely on-device[cite: 31, 32].
* **`checkpoints/`:** Stores weight snapshots produced during the scheduled 165 hours per week of training cycles[cite: 31, 32].
* **`logs/`:** Automated failure logging repository[cite: 31, 32].
  * `logs/hallucinations.json`: Records fabricated claims, unsupported facts, and hallucinated entities[cite: 31, 32].
  * `logs/failed_math.json`: Records arithmetic slips, calculation errors, and pruned equation paths[cite: 31, 32].
  * `logs/failed_code.json`: Records broken scripts, syntax failures, and sandbox tracebacks[cite: 31, 32].

---

## 5. Execution State Guardrails
* **Current Stage Status:** File syncing and workspace scaffold are complete; official core coding has not started.
* **Autonomous Failure Logging:** Upon encountering reasoning slips, execution breaks, or false outputs, log the failure immediately to its corresponding JSON file in `logs/`[cite: 31, 32].
* **Sandboxed Execution Loop:** Code changes must be tested in the local sandbox, capturing runtime errors and repairing them iteratively before final presentation[cite: 31].