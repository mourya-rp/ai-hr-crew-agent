# 🤖 Autonomous Multi-Agent HR Screener
A privacy-first, local AI system that evaluates resumes against job descriptions.

# Technical Stack
- **Orchestration:** CrewAI (Multi-Agent Workflow)
- **Local LLM:** Qwen 2.5 14B (via Ollama)
- **Vector Engine:** PyTorch & Sentence Transformers (Optimized for M4 MPS)
- **Data Handling:** Pandas, Pydantic, pypdfium2

# Key Features
- **Privacy-First:** Entirely local execution; no data leaves the machine.
- **Hardware Optimized:** Leverages Apple M4 GPU acceleration for similarity math.
- **Structured Logic:** Uses a Senior Technical Screener and HR Manager to provide deterministic scoring.

- ![Project Dashboard](images/output-sample.png)
