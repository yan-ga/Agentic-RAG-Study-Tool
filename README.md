# Agentic RAG Study Tool

An AI-powered study assistant that answers questions about university lecture slides using retrieval-augmented generation (RAG) with vision capabilities. Ask a question in natural language and the agent retrieves the most relevant slides, visually analyses them, and produces a cited answer.

Built for **UNSW COMP9517 (Computer Vision)** lectures, but the architecture generalises to any slide-based course material.

## Features

- **Multimodal retrieval** — Lecture slides are embedded as images via the Doubao multimodal embedding API, enabling semantic search that understands both text and visual content (diagrams, equations, figures).
- **Agentic workflow** — A LangGraph agent decides when to search, which slides to analyse in detail with a vision LLM, and how to synthesise a final answer.
- **Vision analysis** — Individual slides are sent to a vision-capable LLM for pixel-level understanding of diagrams and figures.
- **Conversation memory** — Persistent threads with SQLite-backed checkpoints so you can continue conversations across sessions.
- **Two interfaces** — Streamlit web app and a CLI, both with full thread management (create, switch, delete conversations).

## Architecture

```
Question
   │
   ▼
┌──────────────────────┐
│  LangGraph Agent     │
│  (Doubao LLM)        │
│                      │
│  Tools:              │
│  ├─ search_sources   │──▶ Cosine similarity over precomputed
│  │                   │    multimodal embeddings (916 pages)
│  │                   │
│  └─ analyse_visuals  │──▶ Vision LLM reads the actual slide image
│                      │
│  Middleware:          │
│  ├─ Context editing  │    (trims old tool calls)
│  └─ Summarisation    │    (compresses long histories)
└──────────────────────┘
   │
   ▼
Cited answer with [lectureX p.Y] references
```

## Quick Start

### Prerequisites

- Python 3.11+
- A [Doubao API key](https://console.volcengine.com/ark) (ByteDance Volcano Engine)

### Setup

```bash
git clone https://github.com/yan-ga/Agentic-RAG-Study-Tool.git
cd Agentic-RAG-Study-Tool

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set your API key:

```bash
export DOUBAO_API_KEY="your-api-key-here"
```

Or create a `.env` file (see `.env.example`).

### Prepare Data

Place your lecture PDFs in `data/raw_pdfs/`, then run the build pipeline:

```bash
# 1. Extract page images from PDFs
python build/build_pages.py

# 2. Extract text chunks with visual summaries
python build/build_new_chunks.py

# 3. Compute multimodal embeddings for each page
python build/build_doubao_embeddings.py
```

This produces:
- `data/page_images/` — PNG renderings of every slide
- `data/new_chunks.jsonl` — Text + visual summary per page
- `data/doubao_page_emb.npy` — Precomputed embedding matrix (N x 2048)

### Run

**Streamlit web app:**

```bash
cd "Version 10"
streamlit run app_doubao.py
```

**CLI:**

```bash
cd "Version 10"
python ask_doubao.py
```

CLI commands: `/threads`, `/new <name>`, `/switch <name>`, `/delete <name>`, `/help`, `/exit`

## Project Structure

```
├── Version 10/              # Current production app
│   ├── app_doubao.py        #   Streamlit web interface
│   └── ask_doubao.py        #   CLI interface
├── Version 1–9/             # Earlier iterations (for reference)
├── build/                   # Data preprocessing scripts
│   ├── build_pages.py       #   PDF → PNG page images
│   ├── build_chunks.py      #   PDF → text chunks
│   ├── build_new_chunks.py  #   Add visual summaries via vision LLM
│   └── build_doubao_embeddings.py  # Compute multimodal embeddings
├── search/                  # Standalone retrieval experiments
├── test/                    # Evaluation benchmarks
├── data/                    # (gitignored) PDFs, images, embeddings
├── memory state/            # (gitignored) Conversation checkpoints
├── requirements.txt
└── .env.example
```

### Version History

| Version | What changed |
|---------|-------------|
| 1 | Hybrid BM25 + sentence-transformer retrieval, local Ollama LLM |
| 2 | Added reranking |
| 3 | LangChain agent with tool-use orchestration |
| 4–5 | CLIP vision embeddings for image-aware retrieval |
| 6–7 | ColPali document-vision embeddings |
| 8 | Conversation memory |
| 9 | Persistent memory with LangGraph SQLite checkpoints |
| **10** | **Doubao multimodal embeddings + agentic middleware (current)** |

## Configuration

| Environment Variable | Description |
|---------------------|-------------|
| `DOUBAO_API_KEY` | Required. Your Doubao API key from Volcano Engine. |

The LLM model (`doubao-seed-1-6-lite-251015`) and embedding model (`doubao-embedding-vision-250615`) are configured in the source files.

## License

MIT
