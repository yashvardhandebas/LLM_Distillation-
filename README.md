# AccuLLM — Knowledge Graph Teacher-Student LLM Distillation

<div align="center">

![AccuLLM](https://img.shields.io/badge/AccuLLM-KG_Distillation-6c5ce7?style=for-the-badge&logo=pytorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-2.x-000000?style=for-the-badge&logo=flask&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-VectorDB-00d2a0?style=for-the-badge)

**A complete Knowledge Graph structured distillation pipeline — Teacher LLM → Knowledge Graph → Student Transformer**

[Live Demo](#running-locally) • [Architecture](#architecture) • [Setup](#setup)

</div>

---

## 🧠 What is AccuLLM?

AccuLLM is an end-to-end **Knowledge Graph (KG) powered Teacher-Student LLM Distillation** system. A large teacher LLM (via Groq API) extracts structured knowledge into a graph, which is then used to train a compact student Transformer using a novel **Structured Supervision Loss** that combines Language Modeling and KG relational alignment.

### Key Highlights

- 🏫 **Teacher LLM** (Groq): Extracts entities/relations and generates high-quality answers
- 🗂️ **Knowledge Graph**: Built using NetworkX with GraphML/image export
- 🧩 **FAISS VectorDB**: Semantic vector indexing for KG facts and past queries
- 🎓 **Student Transformer**: GPT-style decoder trained from scratch with structured KG supervision
- 📊 **Evaluation Framework**: Precision/Recall/F1 for extractor, Jaccard/ROUGE-L for distillation quality, and compression ratio
- 🌐 **Interactive Web UI**: Real-time pipeline visualization, KG graph viewer, attention heatmap, and chat interface

---

## Architecture

```
Query
  │
  ▼
[1] Groq Teacher LLM (Information Retrieval)
  │
  ▼
[2] Knowledge Extractor (Entity + Relation Extraction via Groq)
  │
  ▼
[3] Knowledge Graph Builder (NetworkX)
  │
  ├──► GraphML / PNG export
  │
  ▼
[4] FAISS Vector Database (KG facts + past query indexing)
  │
  ▼
[5] KG Retrieval Module (Graph traversal + FAISS semantic search)
  │
  ▼
[6] Teacher Answer Generation (Groq, context-grounded)
  │
  ▼
[7] Student Transformer Training (GPT-style Decoder, from scratch)
     ┌─────────────────────────────────────────────────────────────┐
     │  Structured Supervision Loss (α=0.35):                      │
     │  L_total = (1-α)·L_LM  +  α·L_Structure                    │
     │  L_LM:        Language Modeling Cross-Entropy               │
     │  L_Structure:  KG Entity Cosine Alignment Loss              │
     └─────────────────────────────────────────────────────────────┘
  │
  ▼
[8] Evaluation (Extractor F1, Jaccard, ROUGE-L, Compression Ratio)
  │
  ▼
[9] Store in VectorDB → Interactive Chat with Student LLM
```

---

## Pipeline Steps

| Step | Module | Description |
|------|--------|-------------|
| 0 | `FAISSVectorDB` | Load vector store, search similar past queries |
| 1 | `KnowledgeExtractor` | Fetch information from Groq Teacher LLM |
| 2 | `KnowledgeExtractor` | Extract entities and relations (JSON) |
| 3 | `KGBuilder` | Build NetworkX graph, export GraphML/PNG |
| 4 | `FAISSVectorDB` | Index KG facts into FAISS |
| 5 | `KGRetrievalModule` | Graph traversal + FAISS semantic retrieval |
| 6 | `KnowledgeExtractor` | Teacher generates grounded answer |
| 7 | `StudentLLMInference` | Train student Transformer with structured supervision |
| 8 | `EvaluationFramework` | Compute all distillation metrics |
| 9 | `FAISSVectorDB` | Store query/answers, enable interactive chat |

---

## Student Transformer Architecture

- **Type**: GPT-style autoregressive Decoder-only Transformer
- **Tokenizer**: Custom `StudentTokenizer` (BPE-like, character-level regex)
- **Embeddings**: Learned token embeddings + sinusoidal positional encoding
- **Attention**: Causal Multi-Head Self-Attention (scaled dot-product)
- **Blocks**: N × [LayerNorm → MultiHeadAttention → FFN → LayerNorm]
- **Output**: Linear projection to vocabulary logits

Default config: `embed_dim=128, heads=4, layers=2, ff_dim=256, seq_len=256`

---

## Setup

### Prerequisites

- Python 3.9+
- A [Groq API Key](https://console.groq.com) (free tier works)

### Installation

```bash
# Clone the repository
git clone https://github.com/yashvardhandebas/LLM_Distillation-.git
cd LLM_Distillation-

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate    # Windows
# source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

### `.env` Configuration

```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=qwen/qwen3.8-27b
```

Supported Groq models (auto-fallback enabled):
- `qwen/qwen3.8-27b` (default)
- `llama-3.3-70b-versatile`
- `llama-3.1-8b-instant`
- `allam-2-7b`

---

## Running Locally

### Web Interface (Recommended)

```bash
python server.py
# Open http://localhost:5000
```

### CLI Pipeline

```bash
python main.py
```

---

## Project Structure

```
LLM_Distillation-/
├── server.py              # Flask web server + REST API
├── main.py                # CLI pipeline runner
├── requirements.txt
├── .env.example
├── src/
│   ├── extractor.py       # Groq Teacher LLM: info retrieval + entity/relation extraction
│   ├── builder.py         # Knowledge Graph construction (NetworkX)
│   ├── retriever.py       # KG retrieval module + FAISS KG index
│   ├── generator.py       # Training dataset generator
│   ├── student.py         # Student Transformer + Structured Supervision Loss
│   ├── evaluate.py        # Evaluation framework (all metrics)
│   └── vectordb.py        # FAISS vector database
├── static/
│   ├── index.html         # Web UI
│   ├── app.js             # Frontend logic (pipeline, graph, attention, chat)
│   └── style.css          # Dark glassmorphism UI design
└── vector_store/          # Persisted FAISS index (gitignored)
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Web UI |
| `GET` | `/api/stats` | VectorDB stats |
| `GET` | `/api/history` | Past query history |
| `POST` | `/api/pipeline` | Run full distillation pipeline |
| `POST` | `/api/chat` | Chat with trained Student LLM |
| `POST` | `/api/attention` | Get attention visualization for text |
| `POST` | `/api/search` | Semantic search in VectorDB |

---

## Evaluation Metrics

| Metric | What it measures |
|--------|-----------------|
| Extractor Precision/Recall/F1 | Quality of entity-relation extraction |
| Jaccard Similarity | Token overlap between student and teacher answers |
| Token F1 Overlap | Precision/recall of token-level matching |
| ROUGE-L (approx) | Longest common subsequence similarity |
| KG Fact Grounding Rate | Fraction of KG entities mentioned in student answer |
| Compression Ratio | Teacher params / Student params |
| Parameter Reduction % | How much smaller the student is |

---

## Requirements

```
flask
flask-cors
groq
python-dotenv
networkx
matplotlib
torch
faiss-cpu
numpy
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">
Built with ❤️ using PyTorch, Groq, NetworkX, FAISS, and Flask
</div>
