# VisionRAG

> **Multimodal Document Intelligence** — Upload any image, scanned PDF, invoice, or screenshot. GPT-4o Vision extracts structured data. pgvector stores embeddings. Ask natural language questions and get grounded answers with source citations.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![GPT-4o Vision](https://img.shields.io/badge/GPT--4o-Vision-412991?style=flat&logo=openai&logoColor=white)
![pgvector](https://img.shields.io/badge/pgvector-PostgreSQL-336791?style=flat&logo=postgresql&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white)

---

## What Problem It Solves

Standard RAG works on text. Real-world documents — invoices, lab reports, scanned forms, dashboards — are **images**. Text extraction from images loses tables, layouts, and relationships.

VisionRAG uses GPT-4o Vision to understand the document *visually*, preserving structure that OCR would destroy. Then it stores embeddings in pgvector for semantic search.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    DOCUMENT UPLOAD PIPELINE                      │
│                                                                  │
│  User uploads: invoice.pdf / lab_report.png / dashboard.jpg     │
│                          │                                       │
│                          ▼                                       │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │                 DOCUMENT PROCESSOR                         │  │
│  │                                                            │  │
│  │  PDF? → pdf2image converts each page → PNG                 │  │
│  │  Image? → passed directly                                  │  │
│  └──────────────────────────┬─────────────────────────────────┘  │
│                             │  Image (PNG/JPG)                   │
│                             ▼                                    │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │              GPT-4o VISION EXTRACTION                      │  │
│  │                                                            │  │
│  │  Input:  raw image                                         │  │
│  │                                                            │  │
│  │  Output: structured text including:                        │  │
│  │    • All visible text (headings, paragraphs, labels)       │  │
│  │    • Tables → Markdown table format                        │  │
│  │    • Key-value pairs (Invoice No: 1042, Date: Jan 5)       │  │
│  │    • Spatial relationships preserved                       │  │
│  └──────────────────────────┬─────────────────────────────────┘  │
│                             │  Structured text                   │
│                             ▼                                    │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │              EMBEDDING + STORAGE                           │  │
│  │                                                            │  │
│  │  text-embedding-3-small → 1536-dim vector                  │  │
│  │                                                            │  │
│  │  PostgreSQL + pgvector:                                    │  │
│  │  ┌─────────────────────────────────────────────────────┐   │  │
│  │  │  documents table                                    │   │  │
│  │  │  id | filename | page_num | content | embedding     │   │  │
│  │  │  1  | inv.pdf  | 1        | "Total: $1,240..." | […]│   │  │
│  │  └─────────────────────────────────────────────────────┘   │  │
│  │                                                            │  │
│  │  IVFFlat index: cosine similarity, 100 lists               │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│                    QUERY PIPELINE                                │
│                                                                  │
│  User: "What is the total amount on the invoice?"               │
│                          │                                       │
│                          ▼                                       │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  Query Embedding                                           │  │
│  │  text-embedding-3-small → 1536-dim vector                  │  │
│  └──────────────────────────┬─────────────────────────────────┘  │
│                             │                                    │
│                             ▼                                    │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  pgvector Similarity Search                                │  │
│  │                                                            │  │
│  │  SELECT content, filename, page_num                        │  │
│  │  FROM documents                                            │  │
│  │  ORDER BY embedding <=> query_embedding                    │  │
│  │  LIMIT 5                                                   │  │
│  │                                                            │  │
│  │  Returns top-5 most relevant chunks                        │  │
│  └──────────────────────────┬─────────────────────────────────┘  │
│                             │  Retrieved context                 │
│                             ▼                                    │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  GPT-4o Answer Generation                                  │  │
│  │                                                            │  │
│  │  System: "Answer from the provided context only.           │  │
│  │           Cite the source document and page."              │  │
│  │  Context: [retrieved chunks]                               │  │
│  │  User: "What is the total amount on the invoice?"          │  │
│  │                                                            │  │
│  │  Response: "The total amount is $1,240.00 as shown on      │  │
│  │            page 1 of invoice.pdf (Invoice No. 1042)."      │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

---

## Why pgvector over Pinecone/ChromaDB?

```
pgvector = vector search inside PostgreSQL

Advantages for this use case:
  ✅ Same DB for documents + embeddings (no separate vector DB)
  ✅ ACID transactions — embedding + metadata always consistent
  ✅ SQL joins — "find invoice from customer X with total > $1000"
  ✅ IVFFlat index — production-grade ANN search
  ✅ Docker: one service (postgres:pgvector) instead of two

Trade-off vs Pinecone:
  ❌ Doesn't scale to 100M+ vectors without tuning
  ✅ Fine for <1M docs — which covers most enterprise document use cases
```

---

## pgvector Schema

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE documents (
    id          SERIAL PRIMARY KEY,
    filename    TEXT NOT NULL,
    page_num    INT DEFAULT 1,
    content     TEXT NOT NULL,          -- GPT-4o extracted text
    embedding   vector(1536) NOT NULL,  -- text-embedding-3-small output
    created_at  TIMESTAMP DEFAULT NOW()
);

-- IVFFlat index: approximate nearest neighbor (ANN)
-- 100 lists = good balance for <100K documents
CREATE INDEX ON documents
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);
```

---

## Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| Vision LLM | GPT-4o Vision | Understands images, preserves table structure |
| Embeddings | OpenAI text-embedding-3-small | 1536 dims, cost-efficient |
| Vector DB | pgvector (PostgreSQL extension) | SQL + vectors in one DB |
| Database | PostgreSQL 16 | Production-grade, ACID |
| API | FastAPI | Async, auto-generates Swagger docs |
| Frontend | Streamlit | Upload UI + Q&A chat interface |
| PDF Processing | pdf2image, PyMuPDF | Converts PDF pages to images |
| Container | Docker + docker-compose | One command brings up app + DB |

---

## Supported Document Types

| Type | Examples | What GPT-4o Extracts |
|------|---------|----------------------|
| Invoices | billing PDFs, receipts | Line items, totals, vendor details |
| Reports | annual reports, research papers | Tables, charts descriptions, key figures |
| Forms | KYC docs, application forms | Field labels + filled values |
| Screenshots | UI dashboards, error screens | All visible text + layout |
| Medical | lab reports, prescriptions | Test results, dosage, patient info |

---

## Features

- **Any image or PDF** — scanned docs, screenshots, invoices, reports, forms
- **GPT-4o Vision extraction** — preserves tables and structure that OCR loses
- **pgvector similarity search** — production-grade ANN with IVFFlat index
- **Multi-document support** — query across all uploaded files simultaneously
- **Source citations** — every answer shows filename + page number
- **REST API** — programmatic upload + query via FastAPI endpoints
- **Streamlit UI** — drag-and-drop upload + conversational Q&A
- **Fully Dockerized** — `docker-compose up` starts app + PostgreSQL/pgvector

---

## Quick Start

```bash
git clone https://github.com/iampriyabrat14/VisionRAG
cd VisionRAG
cp .env.example .env           # add OpenAI key
docker-compose up --build      # starts app + postgres with pgvector
```

- API + Swagger: `http://localhost:8000/docs`
- Streamlit UI: `http://localhost:8501`

### Environment Variables

```env
OPENAI_API_KEY=your_key
POSTGRES_HOST=localhost
POSTGRES_DB=visionrag
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/upload` | Upload PDF or image, triggers extraction + embedding |
| POST | `/query` | Ask a question across all uploaded docs |
| GET | `/documents` | List all uploaded documents |
| DELETE | `/documents/{id}` | Remove a document + its embeddings |
| GET | `/health` | Health check |

---

## Project Structure

```
VisionRAG/
├── app/
│   ├── vision/
│   │   ├── extractor.py        # GPT-4o Vision extraction logic
│   │   └── pdf_to_image.py     # PDF page → PNG converter
│   ├── embeddings/
│   │   └── embedder.py         # OpenAI embedding wrapper
│   ├── db/
│   │   ├── schema.sql          # pgvector table + index definition
│   │   └── vector_store.py     # pgvector CRUD operations
│   ├── rag/
│   │   └── chain.py            # Retrieval + GPT-4o generation chain
│   ├── api/
│   │   └── routes.py           # FastAPI upload + query endpoints
│   └── main.py
├── frontend/
│   └── app.py                  # Streamlit UI
├── tests/
│   ├── test_extractor.py
│   └── test_vector_store.py
├── docker-compose.yml           # App + PostgreSQL/pgvector
├── Dockerfile
├── requirements.txt
└── .env.example
```

---

## Interview Talking Points

**Q: Why use GPT-4o Vision instead of PyPDF2 or OCR?**
> Traditional PDF parsers extract text linearly — they lose table structure, column alignment, and key-value relationships. GPT-4o Vision reads the page as a human would, preserving "Invoice Total: $1,240" as a semantic unit rather than scattered tokens across the page.

**Q: How does pgvector's IVFFlat index work?**
> IVFFlat (Inverted File with Flat quantization) clusters embeddings into N lists. At query time, it only searches the closest few clusters instead of the full table — giving approximate nearest neighbor (ANN) search in O(√n) instead of O(n). For <100K documents, 100 lists is the standard tuning.

**Q: What happens with a multi-page PDF?**
> `pdf2image` converts each page to a PNG. Each page is extracted separately by GPT-4o Vision and stored as an individual row in the documents table with its `page_num`. This means queries can locate the exact page, not just the document.

---

## Roadmap

- [ ] Batch upload support
- [ ] Cross-encoder reranker for better retrieval precision
- [ ] Table extraction → structured JSON output
- [ ] Deploy to AWS ECS + RDS PostgreSQL
- [ ] Export Q&A session as PDF report

---

## Connect

**Priyabrat Dalbehera** — AI Engineer | Building production GenAI systems

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/priyabrat-dalbehera-p521)
[![Portfolio](https://img.shields.io/badge/Portfolio-000000?style=flat&logo=vercel&logoColor=white)](https://www.aiwithpriyabrat.com/)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white)](https://github.com/iampriyabrat14)
[![Email](https://img.shields.io/badge/Email-D14836?style=flat&logo=gmail&logoColor=white)](mailto:ipriyabrat689@gmail.com)
