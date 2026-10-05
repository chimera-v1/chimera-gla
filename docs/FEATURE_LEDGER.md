# CHIMERA MASTER FEATURE LEDGER

This ledger documents every feature specified across all six core technical domains and the orbital roadmap[cite: 17, 18, 31].

---

## Domain 1: The Core Brain
*How the tiny AI model itself is built to be small, fast, and smart[cite: 20, 31].*

1. **45-Million-Parameter Core Backbone:** A compact AI model small enough to run on an ordinary laptop or small chip without requiring expensive graphics cards[cite: 20, 31].
2. **Recurrent Weight-Sharing Loop:** Reuses the same six building blocks three times in a row, allowing a small model to "think" in stages like a larger model[cite: 20, 31].
3. **BitNet Ternary Quantization:** Shrinks internal values down to -1, 0, or +1, bringing the model size to 9–12 MB and enabling inference via simple addition instead of matrix multiplication[cite: 20, 31].
4. **Multi-Head Latent Attention (MLA):** Compresses short-term context memory by roughly 6x[cite: 20, 31].
5. **Logic-Gated MLA:** Locks crucial facts into memory to prevent degradation or garbling during reasoning[cite: 20, 31].
6. **Mixture of Depths Router:** Bypasses unnecessary compute steps on simple text tokens (such as punctuation and whitespace), reducing wasted effort by 30–50%[cite: 20, 31].
7. **GQA + Sparse Attention Hybrid:** Attends only to the 32 most relevant pieces of information rather than the full context[cite: 20, 31].
8. **32-Slot Scratchpad:** Dedicated memory slots where the model stores intermediate variables and partial results while solving tasks[cite: 20, 31].
9. **ThinKV Memory Compression:** Discards trivial thoughts during reasoning steps, reducing thinking memory by up to 95%[cite: 20, 31].
10. **Speculative Decoding:** Generates and validates several prospective words in parallel, accelerating inference 2–3x[cite: 20, 31].
11. **Thermal-Adaptive Throttling:** Monitors system temperature and scales down computational load if the machine begins overheating[cite: 20, 31].
12. **RoPE & SwiGLU Building Blocks:** Core positional and feed-forward primitives to preserve word ordering and ensure smooth representational merging[cite: 20, 31].
13. **Custom 20,000-Word Tokenizer:** Purpose-built vocabulary optimized for numeric strings and code indentation to eliminate common arithmetic and syntax mistakes[cite: 20, 31].
14. **Protected Reasoning Tags:** Dedicated delimiter tokens isolating private internal reasoning from final user-facing responses[cite: 21, 31].
15. **State Space Model (SSM) Mode:** CPU-first execution alternative that runs without a GPU for broad hardware compatibility[cite: 21, 31].
16. **Sub-200 MB RAM Promise:** Hard architectural rule mandating that the model, database, and system footprint operate within roughly 200–500 MB of total memory[cite: 21, 31].

---

## Domain 2: Knowledge & Reasoning
*How the app looks things up, checks its own logic, and avoids making things up[cite: 22, 31].*

17. **Micro-GraphRAG:** On-device local knowledge graph capturing facts and semantic relationships without external servers[cite: 22, 31].
18. **Markdown / Vault Parser:** Ingests local markdown notes, folder trees, and Obsidian vaults, indexing headers, tags, and bi-directional links[cite: 22, 31].
19. **Local Embedding Engine:** 22 MB offline embedding model converting text into semantic vectors without internet access[cite: 22, 31].
20. **1-Hop Neighbor Expansion:** Retrieves immediately adjacent relational facts during graph lookups to enrich contextual depth[cite: 22, 31].
21. **Retrieval-Augmented Thoughts (RAT):** Pauses mid-generation during reasoning steps to retrieve local facts and inject them into active context[cite: 22, 31].
22. **Retrieval-Augmented Thought Tree (RATT):** Branches multiple retrieval-grounded reasoning paths concurrently prior to selecting the optimal solution[cite: 22, 31].
23. **Multi-Critic Scratchpad (M-RAT):** Internal debate mechanism pitting a generator persona against a critic persona to identify logical errors[cite: 22, 31].
24. **Symbolic Branch Pruning (Symb-RATT):** Evaluates arithmetic and logic through an exact code checker, discarding any branch that fails execution[cite: 22, 31].
25. **Speculative RAG Drafting:** Pre-fetches predicted factual lookups before they are explicitly requested[cite: 22, 31].
26. **Keyless Web Scraper Fallback:** Performs anonymous fallback web queries to retrieve context when local knowledge stores are insufficient[cite: 22, 31].

---

## Domain 3: Personalization & Self-Improvement
*How the app learns your habits and gets better over time, entirely on your device[cite: 23, 31].*

27. **Living Intelligence (Micro-LoRAs):** Learns from user edits and corrections by creating lightweight 1–3 MB on-device parameter patches instead of requiring full retrains[cite: 23, 31].
28. **Autonomous Weight Merging:** Merges accumulated Micro-LoRAs into the core model weights periodically to avoid computational overhead[cite: 23, 31].
29. **Dream Mode:** Analyzes daily failure logs during idle night periods to synthesize insights and optimize prompts for the following morning[cite: 23, 31].
30. **Predictive Pre-Computation:** Anticipates probable follow-up queries and generates preliminary responses in the background[cite: 23, 31].
31. **Uncertainty Heatmap:** Visual confidence scoring using green, yellow, and red color codes across generated tokens[cite: 23, 31].
32. **Persistent Cognitive Fingerprint:** Local profile modeling user tone, vocabulary, and problem-solving workflows[cite: 23, 31].
33. **Structured Failure Logging:** Automated tracking routing hallucinations, math slips, and syntax failures into dedicated JSON logs (`logs/hallucinations.json`, `logs/failed_math.json`, `logs/failed_code.json`)[cite: 23, 31, 32].

---

## Domain 4: The App, UI & Study Tools
*What the everyday desktop application actually looks and feels like to use[cite: 24, 31].*

34. **100% Offline Mode:** Complete functionality—including reasoning, search, and coding—runs with network interfaces disabled[cite: 24, 31].
35. **Lightweight Desktop Shell:** Sub-20 MB native desktop shell framework avoiding bulky runtimes like Electron[cite: 24, 31].
36. **Dual-Panel Workspace:** Split-screen layout hosting chat/notes on the left and a live editor/sandbox preview on the right[cite: 24, 31].
37. **Interactive Knowledge Graph Canvas:** Dynamic node map visualizing indexed local files and illuminating nodes active during inference[cite: 24, 31].
38. **Graph-to-Code Blueprinting:** Bi-directional synchronization linking architectural graph nodes to source code files, flagging broken components in red[cite: 24, 31].
39. **Photo-to-Solution OCR:** Offline image ingestion and optical parsing of handwritten math and code[cite: 24, 31].
40. **Pedagogical Hint Mode:** Interactive tutoring engine providing progressive hints rather than direct solutions[cite: 24, 31].
41. **Offline LaTeX / Math Rendering:** In-app rendering engine for equations, calculus notation, and linear algebra matrices without external web CDNs[cite: 24, 31].
42. **Automated Exam Generator:** Generates structured test papers tailored to difficulty parameters and curriculum standards[cite: 24, 31].
43. **Local Code Sandbox:** Isolated offline execution environment that captures runtime exceptions and feeds tracebacks back to the agent for auto-repair[cite: 24, 31].
44. **Formula & Snip Book:** Local repository for pinning, bookmarking, and querying recurring equations and code patterns[cite: 24, 31].
45. **Step Replay Timeline:** Visual scrub bar enabling frame-by-frame review of the model's reasoning trajectory[cite: 24, 31].
46. **Local Progress Analytics:** On-device dashboard tracking study streaks, mastery topics, and weak concept areas[cite: 24, 31].
47. **Native PDF Export:** Compiles chats, code snippets, and rendered math into formatted PDF documents[cite: 24, 31].
48. **Offline Voice Input:** Speech-to-text input parsing using on-device operating system APIs[cite: 24, 31].
49. **Zero-Config Installers:** Standalone single-package installers for Windows, macOS, and Linux requiring no pre-installed dependencies[cite: 24, 31].

---

## Domain 5: Networking & Mesh Infrastructure
*How multiple devices, or a whole office, can share the AI's workload[cite: 25, 31].*

50. **Microcontroller Layer Loading:** Sequential layer-streaming loader enabling low-memory microcontrollers to process layers sequentially[cite: 25, 31].
51. **Zero-Config LAN Mesh Inference:** Peer-to-peer subnet discovery that splits model layer inference across local local machines[cite: 25, 31].
52. **Air-Gapped Server-Client Link:** Offloads complex reasoning from low-power endpoints to a primary local server without internet connectivity[cite: 25, 31].
53. **Network Diagnostic Agent:** Diagnostic module identifying local ping latency, router bottlenecks, and interface connectivity offline[cite: 25, 31].
54. **Split-Brain Cloud Router:** Dual-tier router running standard tasks locally and routing heavy edge cases to cloud endpoints for paid tiers[cite: 25, 31].
55. **Agentic Execution Loop:** Autonomous agent executing iterative terminal loops—reading code, running test suites, and modifying files until tests pass[cite: 25, 31].
56. **Enterprise & Air-Gap Licensing:** Offline multi-seat software licensing system for air-gapped industrial, clinical, and defense setups[cite: 25, 31].
57. **Merchant of Record Payments:** Automated tax-compliant payment gateway handling subscriptions and global transactions[cite: 25, 31].

---

## Domain 6: Chimera Orbital (Space & Aerospace)
*Taking the same core technology and putting it to work on satellites in orbit[cite: 26, 31].*

58. **Smart Bandwidth Filtering:** Edge payload filter determining data relevance on-orbit, decreasing downlink bandwidth costs by up to 80%[cite: 26, 31].
59. **Autonomous In-Orbit Diagnostics:** Telemetry monitoring system tracking bus voltage, gyroscopes, and thermal state, executing autonomous corrective routines[cite: 26, 31].
60. **Radiation-Tolerant Memory Protection:** Bit-level scrubbing designed for ternary (-1, 0, +1) weights to detect and reload single-event upsets caused by cosmic radiation[cite: 26, 31].
61. **Swarm Optical Mesh:** Inter-satellite laser cross-link communication sharing distributed reasoning workloads across satellite clusters[cite: 26, 31].
62. **Satellite-to-Satellite Repair:** Peer diagnostics over optical laser links allowing healthy satellites to push software patches to adjacent degraded units[cite: 26, 31].
63. **Deep-Space Eclipse Dream Mode:** Self-optimization and data pruning cycle triggered when orbiting spacecraft pass through planetary shadow (eclipse)[cite: 26, 31].
64. **Trajectory Safety Checks:** Kinematic verification pipeline checking planned thruster burns against orbital physics constraints prior to engine ignition[cite: 26, 31].
65. **Mission Control Uncertainty Heatmap:** Downlink telemetry encoding model confidence metrics into visual heatmaps for flight operators[cite: 26, 31].
66. **CubeSat Hardware Payload:** Standardized circuit board payload designed to drop directly into standard CubeSat form-factor chassis[cite: 26, 31].

---

## Suggested Features: Orbital Roadmap Extensions
*Next-generation aerospace extensions to expand Domain 6 capabilities[cite: 27, 31].*

67. **Space Weather Early Warning:** Identifies solar storm and CME alerts during ground downlink, configuring onboard rad-hard protection modes prior to storm arrival[cite: 27, 31].
68. **Federated Learning Across the Swarm:** Exchanges lightweight weight updates over laser links instead of raw telemetry to train fleet models collectively[cite: 27, 31].
69. **Smart Adaptive Re-Imaging:** Identifies terrestrial events (such as wildfires or floods) and autonomously tasks higher-resolution payload imaging on subsequent orbital passes[cite: 27, 31].
70. **Command Authentication Check:** Telemetry validation engine cross-checking uplinked ground commands against fuel reserves, thermal thresholds, and orbit parameters[cite: 27, 31].
71. **Swarm Consensus Before Risky Actions:** Distributed consensus protocol requiring validation from at least two neighboring satellites before executing non-reversible orbital burns[cite: 27, 31].
72. **Predictive Hardware Aging Model:** Tracks degradation curves on reaction wheels, batteries, and sensors to forecast component failure before end-of-life[cite: 27, 31].