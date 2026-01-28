# MARA - Multimodal Agentic Reasoning Assistant

Built a LangGraph-based orchestration system coordinating RAG, Vision, Data, and Web Search agents for complex multi-modal queries. Implemented hybrid retrieval (FAISS + BM25), adaptive planning with error recovery, and production-ready FastAPI REST API.

## 🎯 Features

- **Multi-Agent Orchestration**: LangGraph coordinates RAG, Vision, Data Analysis, and Web Search agents
- **Intelligent Planning**: Automatic task decomposition and agent selection based on query analysis
- **Hybrid Retrieval**: FAISS vector search (70%) + BM25 keyword matching (30%) for optimal accuracy
- **Real-Time Web Search**: DuckDuckGo integration for current events without API keys
- **Vision Analysis**: GPT-4o image understanding, chart extraction, and OCR
- **Data Analytics**: Pandas-based analysis with case-insensitive column matching
- **Self-Verification**: Critic agent validates outputs and assigns confidence scores
- **Production API**: FastAPI with 10 endpoints, Swagger docs, comprehensive error handling

## 🏗️ Architecture

```
User Query → FastAPI → LangGraph Orchestrator
                ↓
        Planner Agent (decomposes tasks)
                ↓
   Vision | RAG | Data | Web Search (parallel)
                ↓
          Critic Agent (validates)
                ↓
        Report Generator
                ↓
    Structured JSON Response
```

## 🧠 Agents

| Agent | Purpose |
|-------|---------|
| **Planner** | Task decomposition, agent selection, workflow management |
| **RAG** | Hybrid document retrieval (vector + BM25), Q&A with citations |
| **Vision** | Image analysis, chart extraction, OCR (GPT-4o) |
| **Data** | Statistical analysis, trend detection, anomaly detection |
| **Web Search** | Real-time information from DuckDuckGo |
| **Critic** | Output verification, hallucination detection, confidence scoring |
| **Report** | Structured report generation with evidence and recommendations |

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- OpenAI API key

### Installation

```bash
# Clone repository
git clone https://github.com/trayan4/MARA-AI-Agent.git
cd mara

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

```bash
# Create .env file
echo "OPENAI_API_KEY=sk-your-key-here" > .env
```

### Run

```bash
# Quick start (checks + starts server)
python start.py

# Or direct uvicorn
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Access at:
- **API**: http://localhost:8000
- **Docs**: http://localhost:8000/docs
- **Health**: http://localhost:8000/health

### Test

```bash
# Run system tests
python test_setup.py
```

## 📖 Usage Examples

### Using Swagger UI (Easiest)

1. **Start the server** and open http://localhost:8000/docs
2. **Click on any endpoint** to expand it
3. **Click "Try it out"**
4. **Fill in the parameters** (JSON for queries, file upload for images/CSV)
5. **Click "Execute"** to see the response

**Example: Query endpoint**
- Expand `POST /query`
- Click "Try it out"
- Paste JSON:
```json
{
  "query": "What is artificial intelligence?",
  "context": {},
  "include_metadata": true
}
```
- Click "Execute"

**Example: Upload file**
- Expand `POST /upload`
- Click "Try it out"
- Click "Choose File" and select your image/CSV
- Click "Execute"
- Copy the `filepath` from response

### Using Python (Programmatic)

### 1. Document Q&A

```python
import requests

# Add document to knowledge base
requests.post(
    "http://localhost:8000/documents/add",
    json={
        "doc_id": "ai_basics",
        "content": "Artificial Intelligence is the simulation of human intelligence...",
        "metadata": {"category": "tech"}
    }
)

# Query
response = requests.post(
    "http://localhost:8000/query",
    json={"query": "What is AI?"}
)

print(response.json()["executive_summary"])
```

### 2. Data Analysis

```python
# Upload CSV
with open("sales.csv", "rb") as f:
    upload = requests.post(
        "http://localhost:8000/upload",
        files={"file": f}
    )

# Analyze
response = requests.post(
    "http://localhost:8000/query",
    json={
        "query": "Analyze sales trends and provide insights",
        "context": {"uploaded_files": [upload.json()["filepath"]]}
    }
)

print(response.json()["data_insights"])
```

### 3. Image Analysis

```python
# Upload image
with open("chart.png", "rb") as f:
    upload = requests.post(
        "http://localhost:8000/upload",
        files={"file": f}
    )

# Analyze
response = requests.post(
    "http://localhost:8000/query",
    json={
        "query": "Describe this image",
        "context": {"uploaded_files": [upload.json()["filepath"]]}
    }
)

print(response.json()["visual_insights"])
```

### 4. Web Search

```python
# Current events query
response = requests.post(
    "http://localhost:8000/query",
    json={"query": "What are the latest AI developments?"}
)

print(response.json()["executive_summary"])
```

## 📊 Response Format

```json
{
  "query": "Analyze Q4 sales trends",
  "executive_summary": "Q4 revenue shows 23% YoY growth...",
  "visual_insights": {},
  "data_insights": {
    "summary": "Analyzed 9994 rows, 28 columns",
    "key_findings": ["Peak in December", "APAC outperformed"],
    "statistics": {"mean": 125000, "growth_rate": 0.23}
  },
  "evidence": [
    {
      "source": "sales.csv",
      "content": "Total revenue: $15.2M",
      "type": "data"
    }
  ],
  "recommendations": [
    "Focus on high-growth segments",
    "Prepare inventory for December peak"
  ],
  "confidence": 0.94,
  "metadata": {
    "agents_used": ["planner", "data", "critic", "report"],
    "execution_time": 12.3
  }
}
```

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/query` | POST | Main query processing |
| `/upload` | POST | Upload files (images, CSV, docs) |
| `/documents/add` | POST | Add document to knowledge base |
| `/documents/{id}` | DELETE | Remove document |
| `/documents` | GET | List all documents |
| `/search` | POST | Search documents (no answer generation) |
| `/analyze/data` | POST | Quick data analysis |
| `/health` | GET | Health check |
| `/stats` | GET | System statistics |
| `/tools` | GET | List available agents/tools |

Full API documentation: http://localhost:8000/docs

## 🛠️ Tech Stack

- **LLM**: OpenAI GPT-4o, GPT-4 Turbo
- **Embeddings**: HuggingFace Sentence Transformers (BAAI/bge-base-en-v1.5)
- **Vector Store**: FAISS + SQLite
- **Orchestration**: LangGraph
- **API**: FastAPI
- **Data Engine**: Pandas
- **Web Search**: DuckDuckGo (ddgs)
- **Logging**: Loguru

## 📁 Project Structure

```
mara/
├── agents/              # 7 AI agents
│   ├── planner.py      # Task planning
│   ├── rag.py          # Document retrieval
│   ├── vision.py       # Image analysis
│   ├── data.py         # Data analytics
│   ├── web_search.py   # Web search
│   ├── critic.py       # Verification
│   └── report.py       # Report generation
├── api/
│   └── main.py         # FastAPI app (10 endpoints)
├── orchestrator/       # LangGraph workflow
│   ├── graph.py        # Main orchestration
│   ├── state.py        # State management
│   ├── router.py       # Routing logic
│   └── error_handler.py
├── tools/              # Utilities
│   ├── openai_client.py
│   ├── local_vector_store.py
│   ├── chunking.py
│   ├── python_executor.py
│   └── definitions/    # Tool schemas (JSON)
├── config/
│   ├── settings.yaml   # Configuration
│   └── __init__.py
├── data/               # Runtime data
│   ├── uploads/        # User files
│   ├── logs/           # System logs
│   ├── vector_store.db
│   └── faiss.index
├── start.py            # Quick start script
├── test_setup.py       # System tests
└── requirements.txt

For more info regarding the directory structure, refer to the structure.pdf file
```

## ⚙️ Configuration

Edit `config/settings.yaml`:

```yaml
# LLM settings
llm:
  model: "gpt-4o"
  temperature: 0.1

# Vector search
vector_store:
  top_k: 5
  use_hybrid: true
  hybrid_alpha: 0.7  # 70% vector, 30% BM25

# Agents
agents:
  critic:
    confidence_threshold: 0.8
```

## 🔧 Troubleshooting

**OpenAI API Error**
```bash
# Check .env file
cat .env
# Should contain: OPENAI_API_KEY=sk-...
```

**Import Errors**
```bash
pip install -r requirements.txt
```

**Port in Use**
```bash
# Use different port
uvicorn api.main:app --port 8001
```

**Vector Store Issues**
```bash
# Reset vector store
rm data/*.db data/*.index
python test_setup.py
```

## 📈 Performance

- **Average query time**: 8-15 seconds (3-5 agents)
- **Hybrid search**: 30% faster than pure vector search
- **Parallel execution**: Agents run concurrently
- **Caching**: LLM responses cached in SQLite
- **File logging**: All operations logged to `data/logs/mara.log`

## 🔒 Security

- API keys stored in `.env` (git-ignored)
- Safe mode for Python code execution
- Local vector storage (SQLite + FAISS)
- File upload validation
- CORS configurable

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create feature branch
3. Commit changes
4. Open pull request

---

**Built for enterprise-grade multi-modal AI analysis**
