# APEX - Self-Improving GraphRAG

APEX is a FastAPI + React project that compares a baseline local LLM answer with a self-improving GraphRAG pipeline. It is designed for a polished demo: every query records latency, tokens, CRAG quality, cache behavior, and savings.

## What Makes It Strong

- Local LLM support through Ollama, defaulting to `llama3.2:latest`
- Side-by-side Baseline vs GraphRAG comparison
- CRAG grading to decide whether retrieval is good enough
- Semantic cache for zero-token repeat answers
- TigerGraph-ready graph layer with local PRD/README fallback retrieval for demos
- SQLite metrics store powering improvement charts
- Professional React dashboard with health, readiness, architecture, and benchmark views

## Run Locally

Start Ollama and confirm the model exists:

```powershell
ollama list
```

Run the backend. Port `8000` is blocked on some Windows setups, so `8010` is the recommended local port:

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8010 --reload
```

Run the frontend:

```powershell
cd frontend
npm run dev -- --host 127.0.0.1 --port 3000
```

Open:

```text
http://127.0.0.1:3000
```

API docs:

```text
http://127.0.0.1:8010/docs
```

## Important Environment Values

```env
LLM_PROVIDER=ollama
LLM_MODEL=llama3.2:latest
OLLAMA_HOST=http://localhost:11434
```

The Vite dev proxy defaults to `http://127.0.0.1:8010`. Override it only if you run the backend elsewhere:

```powershell
$env:VITE_API_TARGET="http://127.0.0.1:8000"
npm run dev -- --host 127.0.0.1 --port 3000
```

## Optional Services

The app works without TigerGraph or Redis by using local fallback retrieval and in-memory cache. For a fuller production-style demo:

```powershell
docker compose up redis
docker compose up tigergraph
```

Redis Insight:

```text
http://127.0.0.1:8001
```

TigerGraph Studio:

```text
http://127.0.0.1:14240
```

## Validate

Backend import check:

```powershell
python -m compileall app llm services routes evaluation graph orchestration security
```

Frontend production build:

```powershell
cd frontend
npm run build
```

Health check:

```powershell
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8010/api/health
```

## Browser Console Note

This message is usually caused by a Chrome extension, not this app:

```text
A listener indicated an asynchronous response by returning true, but the message channel closed before a response was received
```

APEX does not use `chrome.runtime` or extension message listeners. Test in an incognito window with extensions disabled if you want to confirm.
