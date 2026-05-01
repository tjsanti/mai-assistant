# Personal Knowledge Agent

Local-first personal AI assistant with a FastAPI backend, a LangGraph main agent, a browser UI, and a local RAG tool over personal documents.

## Architecture Overview

- `backend/`: FastAPI API, provider abstraction, LangGraph router, RAG ingestion and retrieval.
- `frontend/`: Vite + React chat UI for localhost use.
- `docs/`: Local document directory for `.txt`, `.md`, and `.pdf` files.
- `evals/`: Promptfoo configuration and test cases.

## Setup

### Backend with `uv`

```bash
cd backend
uv sync
cp .env.example .env
```

### Ollama Setup

Install Ollama, run the local server, and pull the configured model:

```bash
ollama pull llama3.1:8b
```

### OpenAI Setup

Set `OPENAI_API_KEY` in `backend/.env`.

## Add Documents

Place `.txt`, `.md`, or `.pdf` files in [`docs/`](/Users/trevor/github/mai-assitant/docs:1).

## Ingest Documents

```bash
cd backend
uv run python -m app.rag.ingestion
```

Or call:

```bash
POST /ingest
```

## Run The App

```bash
./start-app.sh
```

This starts:

- the FastAPI backend on `http://localhost:8000`
- the Vite frontend on `http://localhost:5173`

Then open `http://localhost:5173` in your browser.

If you only want the backend:

```bash
cd backend
uv run uvicorn app.main:app --reload
```

## Run Frontend

```bash
cd frontend
npm install
npm run dev
```

## Run Tests

```bash
cd backend
uv run pytest
```

## Run Promptfoo Evals

```bash
cd evals
promptfoo eval
```

## Current Limitations

- No auth, streaming, document upload UI, or persistent chat history.
- Routing sends the raw user query to the general OpenAI router by default.
- Retrieved document context stays local unless `ALLOW_CLOUD_RAG_CONTEXT=true`.
- Retrieval is vector-only; no reranking or metadata filters yet.

## Future Roadmap

- V1.5: better eval dataset, RAGAS, LangSmith tracing, streaming, improved source display.
- V2: hybrid search, reranking, metadata filters, configurable model settings in UI, MCP tools.
- V3: email/calendar tools, long-term memory, background jobs, authentication, remote access.
