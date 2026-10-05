---
trigger: always_on
---

# CHIMERA WORKFLOW ENFORCEMENT RULES

Every autonomous agent session, terminal execution, and code generation loop must strictly adhere to these 10 enforcement rules:

---

#### Rule 1: Zero External Network Dependency
* All generated code, dependency choices, tokenizers, embedding models, and search indexers must operate with zero internet connectivity[cite: 31].
* The app must be fully functional with Wi-Fi switched off[cite: 31, 32].
* Never introduce libraries that require cloud telemetry, online CDNs, or external API pings[cite: 31].

#### Rule 2: Strict Memory Budget (200–500 MB Ceiling)
* The entire running stack—including model weights, database engines, and runtime shells—must remain between 200 MB and 500 MB of RAM[cite: 31, 32].
* Desktop UI solutions must use lightweight shells under 20 MB; Electron and other resource-heavy frameworks are strictly banned[cite: 31].

#### Rule 3: Architectural Consistency
* Model designs must target the 45-Million-Parameter Core Backbone[cite: 31].
* Weight representations must conform to BitNet Ternary Quantization (-1, 0, +1), enabling small 9–12 MB models that operate via basic addition rather than heavy floating-point operations[cite: 31].
* Structural reasoning must reuse the 6 core building blocks sequentially across 3 cycles via the Recurrent Weight-Sharing Loop[cite: 31].

#### Rule 4: Protected Scopes & File Routing
* Code modifications must strictly respect established project boundaries[cite: 32]:
  * Custom 20,000-word vocabulary scripts belong exclusively in `src/tokenizer.py`[cite: 31, 32].
  * Training data streams (Language -> Code -> Math) belong exclusively in `src/dataset.py`[cite: 31, 32].
  * Raw local datasets must stay exclusively in `data/`[cite: 31, 32].
  * Binary weight files must stay exclusively in `checkpoints/`[cite: 31, 32].
  * Never alter `.gitignore` to expose large binary or data assets[cite: 31, 32].

#### Rule 5: Mandatory Structured Failure Routing
* The agent must never silently suppress or bypass a failure[cite: 31].
* Fabrications must be appended to `logs/hallucinations.json`[cite: 31, 32].
* Arithmetic and mathematical mistakes must be appended to `logs/failed_math.json`[cite: 31, 32].
* Code failures and sandbox exceptions must be appended to `logs/failed_code.json`[cite: 31, 32].

#### Rule 6: Verification Precedes Commits
* Code cannot be declared complete until executed and validated inside the Local Code Sandbox[cite: 31].
* Mathematical steps must undergo Symbolic Branch Pruning via code checkers before output generation[cite: 31].

#### Rule 7: Separation of Thought and Output
* All agent reasoning steps must be enclosed inside Protected Reasoning Tags to prevent internal deliberation from contaminating output deliverables[cite: 31].

#### Rule 8: Modular Adapter Discipline (Micro-LoRAs)
* Continual fine-tuning or personalization must use small Micro-LoRAs (1–3 MB patches) rather than retraining the base backbone[cite: 31].
* Accumulated patches must be consolidated through Autonomous Weight Merging to eliminate clutter[cite: 31].

#### Rule 9: Human Approval for Unattended Destruction
* High-risk terminal actions (recursive deletion, force pushes, file removals outside `logs/` or `checkpoints/`) require explicit human authorization[cite: 32].
* The agent must pause and request permission before performing irreversible disk actions[cite: 31].

#### Rule 10: State Synchronization Mandate
* At the conclusion of any feature build or debugging session, the agent must record what was completed, existing blockers, and immediate next steps directly in `PROGRESS.md`[cite: 31].