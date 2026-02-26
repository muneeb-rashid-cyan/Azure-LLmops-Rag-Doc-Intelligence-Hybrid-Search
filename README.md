# Azure-LLmops-Rag-Doc-Intelligence-Hybrid-Search

> Production-grade Multi-Modal RAG system using Azure Document Intelligence, Azure AI Search hybrid retrieval, cross-encoder reranking, context compression, GPT-4.1 generation, and Azure AI Evaluation — all served via a FastAPI dashboard.

---

## What This Project Does

This system ingests PDFs, Word documents, and scanned images, extracts structured content (text + tables) using Azure Document Intelligence, chunks semantically, indexes into Azure AI Search, and answers questions using a 4-stage advanced retrieval pipeline with full citation tracking and automatic quality evaluation.

```
Document Upload
    ↓
Azure Document Intelligence  (OCR, layout, table extraction)
    ↓
Semantic + Table-Aware Chunking
    ↓
Azure OpenAI Embeddings  (text-embedding-ada-002)
    ↓
Azure AI Search Index  (vector + BM25 hybrid)
    ↓
──── QUERY TIME ────────────────────────────────
User Question
    ↓
Query Rewriting       GPT-4.1 → 3 alternative phrasings
    ↓
Multi-Query Hybrid Search    vector + BM25 + RRF fusion
    ↓
Cross-Encoder Reranker       ms-marco-MiniLM  (top 20 → top 5)
    ↓
Context Compression          extract only relevant sentences
    ↓
GPT-4.1 Answer Generation    grounded + cited
    ↓
Azure AI Evaluation          groundedness / relevance / coherence / hallucination
    ↓
Cosmos DB Storage            sessions, documents, eval scores
    ↓
FastAPI Dashboard            upload · chat · citations · eval metrics
```

---

## Advanced RAG Concepts Implemented

| Concept | What It Does | Why It Matters |
|---|---|---|
| **Query Rewriting** | GPT-4.1 rewrites user question into 3 alternative phrasings | Different chunks are written differently — multiple phrasings improve recall |
| **Hybrid Search + RRF** | Combines vector search (semantic) with BM25 (keyword) via Reciprocal Rank Fusion | Vector misses exact terms; BM25 misses paraphrases — hybrid covers both |
| **Cross-Encoder Reranker** | ms-marco-MiniLM reads query + chunk together for precise relevance scoring | Bi-encoders score independently — cross-encoder models the interaction |
| **Context Compression** | GPT extracts only the 2-3 sentences from each chunk that answer the question | Removes noise, saves context window, improves answer quality |
| **Semantic Chunking** | Splits on cosine similarity drops between sentences | Fixed-size chunking splits mid-concept — semantic preserves coherent ideas |
| **Table-Aware Extraction** | Document Intelligence returns tables as JSON; each table is a separate chunk | Tables are never split mid-row; column headers preserved for embedding quality |
| **Citation Tracking** | Every answer claim includes `[filename, Page X]` reference | Full traceability from answer back to source document |
| **Auto Evaluation** | GPT-4.1 scores every answer on groundedness, relevance, coherence, hallucination | Catches bad answers before users see them |

---

## Project Structure

```
azure-multimodal-rag/
├── ingestion/
│   ├── document_loader.py      # Azure Document Intelligence integration
│   ├── chunker.py              # Semantic + table-aware chunking
│   ├── embedder.py             # text-embedding-ada-002 with batching
│   ├── indexer.py              # Azure AI Search index creation + upload
│   └── pipeline.py             # Full ingestion orchestrator
├── retrieval/
│   ├── query_rewriter.py       # GPT-4.1 query expansion
│   ├── hybrid_search.py        # Vector + BM25 + RRF fusion
│   ├── reranker.py             # Cross-encoder reranking
│   ├── context_compressor.py   # Sentence-level compression
│   └── pipeline.py             # Full retrieval orchestrator
├── generation/
│   ├── answer_generator.py     # GPT-4.1 grounded generation
│   ├── citation_formatter.py   # Citation extraction + formatting
│   └── prompt_templates.py     # System prompts
├── evaluation/
│   ├── evaluator.py            # Azure AI Evaluation integration
│   ├── synthetic_dataset.py    # Auto-generate Q&A eval pairs
│   └── metrics.py              # F1, Precision@K, hallucination rate
├── dashboard/
│   ├── app.py                  # FastAPI backend
│   ├── cosmos_client.py        # Cosmos DB operations
│   └── templates/
│       └── index.html          # Upload + chat + eval UI
├── tests/
│   ├── test_ingestion.py
│   ├── test_retrieval.py
│   └── test_generation.py
├── data/
│   ├── sample_docs/            # Place test PDFs here
│   └── eval_dataset/           # Generated Q&A pairs saved here
├── infrastructure/
│   └── setup.ps1               # Azure resource provisioning
├── config.py                   # Centralised settings via pydantic-settings
├── main.py
├── Dockerfile
├── azure-pipelines.yml
├── pyproject.toml
└── .env
```

---

## Azure Resources Required

| Resource | Purpose |
|---|---|
| Azure OpenAI | GPT-4.1 (generation, rewriting, evaluation, compression) + text-embedding-ada-002 |
| Azure Document Intelligence | OCR, layout analysis, table extraction |
| Azure AI Search | Hybrid vector + BM25 index |
| Azure Cosmos DB | Document records, chat sessions, eval scores |
| Azure Container Registry | Docker image storage |
| Azure App Service (Web App) | Container hosting |

---

## Local Setup

### 1. Clone and initialise

```powershell
git clone https://github.com/YOUR_USERNAME/azure-multimodal-rag.git
cd azure-multimodal-rag
pip install uv
uv venv .venv
.venv\Scripts\activate
uv pip install -r requirements.txt
```

### 2. Configure environment

Copy `.env.template` to `.env` and fill in all values:

```
AZURE_OPENAI_ENDPOINT=https://YOUR_RESOURCE.openai.azure.com/
AZURE_OPENAI_KEY=YOUR_KEY
AZURE_OPENAI_DEPLOYMENT=gpt-4.1
AZURE_OPENAI_EMB_DEPLOYMENT=text-embedding-ada-002
AZURE_OPENAI_API_VERSION=2024-08-01-preview

DOCUMENT_INTELLIGENCE_ENDPOINT=https://YOUR_RESOURCE.cognitiveservices.azure.com/
DOCUMENT_INTELLIGENCE_KEY=YOUR_KEY

AZURE_SEARCH_ENDPOINT=https://YOUR_SERVICE.search.windows.net
AZURE_SEARCH_KEY=YOUR_KEY
AZURE_SEARCH_INDEX=rag-documents

COSMOS_ENDPOINT=https://YOUR_ACCOUNT.documents.azure.com:443/
COSMOS_KEY=YOUR_KEY
COSMOS_DATABASE=MultiModalRAG
COSMOS_CONTAINER_DOCS=Documents
COSMOS_CONTAINER_SESSIONS=ChatSessions
COSMOS_CONTAINER_EVALS=Evaluations

ACR_NAME=YOUR_ACR_NAME
WEBAPP_NAME=YOUR_WEBAPP_NAME
RESOURCE_GROUP=YOUR_RESOURCE_GROUP
```

### 3. Verify config loads

```powershell
python -c "from config import get_settings; s = get_settings(); print(s.azure_openai_endpoint)"
```

---

## Running the Pipeline

### Ingest a document

```powershell
python -c "
from ingestion.pipeline import run_ingestion_pipeline
result = run_ingestion_pipeline('data/sample_docs/your_file.pdf')
print(result)
"
```

Expected output:
```
{'doc_id': 'your_file', 'file_name': 'your_file.pdf', 'page_count': 5, 'chunk_count': 42, 'table_count': 3}
```

### Test retrieval + generation

```powershell
python -c "
from retrieval.pipeline import run_retrieval_pipeline
from generation.answer_generator import generate_answer
r = run_retrieval_pipeline('what is this document about?')
a = generate_answer(r.original_query, r.context)
print(a.answer)
"
```

### Run the dashboard locally

```powershell
uvicorn dashboard.app:app --reload --port 8000
```

Open `http://localhost:8000`

### Run unit tests

```powershell
pytest tests/ -v
```

Tests run without Azure calls — pure logic tests for chunking, RRF math, citation parsing, and metric calculations.

### Generate evaluation dataset

Run this after ingesting at least one document:

```powershell
python -c "
from evaluation.synthetic_dataset import generate_eval_dataset
dataset = generate_eval_dataset(n_chunks=20, questions_per_chunk=3)
print(f'Generated {len(dataset)} questions')
"
```

Saves to `data/eval_dataset/eval_questions.json`.

---

## Docker

### Build image

```powershell
docker build -t rag-dashboard:latest .
```

### Run locally with Docker

```powershell
docker run -p 8000:8000 --env-file .env rag-dashboard:latest
```

### Push to Azure Container Registry

```powershell
# Login to ACR
az acr login --name YOUR_ACR_NAME

# Tag image
docker tag rag-dashboard:latest YOUR_ACR_NAME.azurecr.io/rag-dashboard:latest

# Push
docker push YOUR_ACR_NAME.azurecr.io/rag-dashboard:latest
```

---

## Deploy to Azure Web App

### Step 1 — Create App Service Plan (if not exists)

```powershell
az appservice plan create `
  --name rag-app-plan `
  --resource-group YOUR_RESOURCE_GROUP `
  --sku B2 `
  --is-linux
```

### Step 2 — Create Web App for containers

```powershell
az webapp create `
  --resource-group YOUR_RESOURCE_GROUP `
  --plan rag-app-plan `
  --name YOUR_WEBAPP_NAME `
  --deployment-container-image-name YOUR_ACR_NAME.azurecr.io/rag-dashboard:latest
```

### Step 3 — Configure container registry credentials

```powershell
$ACR_PASS = az acr credential show `
  --name YOUR_ACR_NAME `
  --query "passwords[0].value" -o tsv

az webapp config container set `
  --resource-group YOUR_RESOURCE_GROUP `
  --name YOUR_WEBAPP_NAME `
  --docker-custom-image-name YOUR_ACR_NAME.azurecr.io/rag-dashboard:latest `
  --docker-registry-server-url https://YOUR_ACR_NAME.azurecr.io `
  --docker-registry-server-user YOUR_ACR_NAME `
  --docker-registry-server-password $ACR_PASS
```

### Step 4 — Inject all environment variables

```powershell
az webapp config appsettings set `
  --resource-group YOUR_RESOURCE_GROUP `
  --name YOUR_WEBAPP_NAME `
  --settings `
    AZURE_OPENAI_ENDPOINT="https://YOUR_RESOURCE.openai.azure.com/" `
    AZURE_OPENAI_KEY="YOUR_KEY" `
    AZURE_OPENAI_DEPLOYMENT="gpt-4.1" `
    AZURE_OPENAI_EMB_DEPLOYMENT="text-embedding-ada-002" `
    AZURE_OPENAI_API_VERSION="2024-08-01-preview" `
    DOCUMENT_INTELLIGENCE_ENDPOINT="https://YOUR_RESOURCE.cognitiveservices.azure.com/" `
    DOCUMENT_INTELLIGENCE_KEY="YOUR_KEY" `
    AZURE_SEARCH_ENDPOINT="https://YOUR_SERVICE.search.windows.net" `
    AZURE_SEARCH_KEY="YOUR_KEY" `
    AZURE_SEARCH_INDEX="rag-documents" `
    COSMOS_ENDPOINT="https://YOUR_ACCOUNT.documents.azure.com:443/" `
    COSMOS_KEY="YOUR_KEY" `
    COSMOS_DATABASE="MultiModalRAG" `
    WEBSITES_PORT=8000
```

### Step 5 — Restart and verify

```powershell
az webapp restart `
  --resource-group YOUR_RESOURCE_GROUP `
  --name YOUR_WEBAPP_NAME

# Check logs
az webapp log tail `
  --resource-group YOUR_RESOURCE_GROUP `
  --name YOUR_WEBAPP_NAME
```

App will be live at: `https://YOUR_WEBAPP_NAME.azurewebsites.net`

---

## Azure DevOps CI/CD Pipeline

The `azure-pipelines.yml` automates 4 stages on every push to `main`:

| Stage | What Happens |
|---|---|
| **Validate** | Ruff lint + pytest unit tests with mocked Azure credentials |
| **Evaluate** | Runs eval dataset as a quality gate — blocks deploy if groundedness < 3.0 or hallucination rate > 30% |
| **Build** | Docker build + push to ACR with build ID tag |
| **Deploy** | Configure Web App container + inject secrets + health check with 5 retries |

### Setup steps

1. In Azure DevOps → Project Settings → Service Connections, create:
   - `ACR_SERVICE_CONNECTION` → Docker Registry type → point to your ACR
   - `AZURE_SERVICE_CONNECTION` → Azure Resource Manager type → your subscription

2. In Pipelines → Library → Variable Group, add all secrets from `.env` as pipeline variables (mark as secret)

3. Create pipeline pointing to `azure-pipelines.yml` in your repo

4. Push to `main` — pipeline runs automatically

---

## Evaluation Metrics

Every query response is automatically scored:

| Metric | Scale | Description |
|---|---|---|
| **Groundedness** | 1–5 | Are all claims supported by retrieved context? |
| **Relevance** | 1–5 | Does the answer address the question? |
| **Coherence** | 1–5 | Is the answer well-structured and clear? |
| **Hallucination** | True/False | Does the answer contain unsupported claims? |
| **Overall** | 0–1 | Weighted average (groundedness 40%, relevance 35%, coherence 25%) |

Quality gate thresholds (enforced in CI/CD):
- Groundedness mean ≥ 3.0
- Hallucination rate ≤ 30%

---

## Tech Stack

| Layer | Technology |
|---|---|
| Document Extraction | Azure Document Intelligence (prebuilt-layout) |
| Embeddings | Azure OpenAI text-embedding-ada-002 |
| Vector Search | Azure AI Search (HNSW, 1536-dim) |
| Keyword Search | Azure AI Search BM25 |
| Reranker | sentence-transformers cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Generation | Azure OpenAI GPT-4.1 |
| Evaluation | Azure AI Evaluation + custom GPT-based scoring |
| Storage | Azure Cosmos DB (NoSQL) |
| Backend | FastAPI + Uvicorn |
| Containerisation | Docker (multi-stage build) |
| Registry | Azure Container Registry |
| Hosting | Azure App Service (Linux container) |
| CI/CD | Azure DevOps Pipelines |
| Experiment Tracking | MLflow |
| Dependency Management | uv |

---

## Key Design Decisions

**Why hybrid search over pure vector search?**
Vector search captures semantic similarity but misses exact keyword matches (product codes, names, dates). BM25 captures exact matches but misses paraphrases. RRF combines ranked lists without needing to tune weights — both approaches reinforce each other.

**Why cross-encoder reranking after hybrid search?**
Bi-encoders (used for initial retrieval) encode query and passage independently — fast but imprecise. Cross-encoders read them together, modelling the full interaction. Too slow for the full index but ideal as a final precision stage over the top 20 candidates.

**Why context compression before generation?**
A 400-token chunk often contains only 2 relevant sentences. Sending the full chunk wastes the context window and adds noise that can confuse the model. Compression reduces input tokens and improves answer focus.

**Why semantic chunking over fixed-size?**
Fixed-size chunking (every 500 tokens) frequently splits mid-sentence or mid-concept. Semantic chunking detects topic boundaries via embedding similarity drops, keeping coherent ideas together for better retrieval and generation.

---

```
