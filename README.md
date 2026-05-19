# Document-Only Chatbot (Python + FastAPI + LangChain + Chroma + Ollama)

This project creates a chatbot that answers only from your provided data:
- Documentation pages (via URLs list)
- Private Azure DevOps wiki pages (via PAT + wiki API)
- PDF files
- Text / Markdown / HTML files

If relevant context is not found in the indexed data, it returns:
`I don't know based on the provided documents.`

## 1. Project structure

```
drs-chatbot/
  app/
    config.py
    document_loader.py
    ingestion.py
    main.py
    rag.py
    schemas.py
    vectorstore.py
  data/                # Put your docs here
  ingest.py            # CLI indexing script
  requirements.txt
  .env.example
```

## 2. Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy env file:

```bash
Copy-Item .env.example .env
```

4. Install and start Ollama, then pull the chat model:

```bash
ollama pull llama3.2:3b
```

Embeddings now run locally via [fastembed](https://github.com/qdrant/fastembed) (ONNX). The first ingest auto-downloads the configured embedding model (`nomic-ai/nomic-embed-text-v1.5` by default, ~300 MB) from HuggingFace into `~/.cache/fastembed/`. After that it works fully offline. Override with `EMBEDDING_MODEL` in `.env` (e.g. `BAAI/bge-small-en-v1.5` for a smaller/faster option), or set `EMBEDDING_CACHE_DIR` to relocate the cache.

## 3. Add your data

Create `data/` and place files inside (`.pdf`, `.txt`, `.md`, `.html`).

To ingest documentation websites, create `data/urls.txt` with one URL per line:

```txt
https://your-doc-site.com/getting-started
https://your-doc-site.com/api-reference
```

For private Azure DevOps Wiki URLs, set `AZURE_DEVOPS_PAT` in `.env` and include the wiki URLs in `data/urls.txt`.
The loader automatically sends PAT auth for `dev.azure.com` and `*.visualstudio.com` URLs.
For full private wiki ingestion through the official Azure DevOps wiki API, set:

```txt
AZURE_DEVOPS_PAT=your_pat
AZURE_DEVOPS_ORG=your-org
AZURE_DEVOPS_PROJECT=your-project
AZURE_DEVOPS_PROJECTS=project-a,project-b,project-c
AZURE_DEVOPS_WIKI=
AZURE_DEVOPS_WIKI_PATH=/
AZURE_DEVOPS_API_VERSION=7.1
```

`AZURE_DEVOPS_PROJECTS` is a comma-separated list of project names under the same `AZURE_DEVOPS_ORG`. When set, the bot iterates each project and recursively indexes every wiki it can read. Use this for multi-project setups so a single ingest can train on several wikis at once.

If only `AZURE_DEVOPS_PROJECT` is set (and `AZURE_DEVOPS_PROJECTS` is empty), the bot falls back to single-project mode. If `AZURE_DEVOPS_WIKI` is blank, every wiki in the project is indexed; if it is set (and only one project is configured), only that wiki is indexed. `AZURE_DEVOPS_WIKI` is ignored when multiple projects are configured.

Each indexed wiki chunk carries `project` and `wiki` metadata so retrieved answers stay traceable to their source project. Your PAT needs wiki read access across all listed projects. For Azure DevOps REST wiki APIs, the documented scope is `vso.wiki`. If one project fails (e.g., the PAT lacks access), the ingest logs a warning and continues with the remaining projects.
For public Google Docs links, you can paste the normal sharing URL; the loader automatically uses the text export endpoint.

## 4. Build index

Rebuild index from scratch:

```bash
python ingest.py
```

Append mode:

```bash
python ingest.py --append
```

## 5. Run API

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Endpoints:
- `GET /health`
- `POST /ingest`
- `POST /chat`

## 6. Example API calls

Chat request:

```bash
curl -X POST http://localhost:8000/chat ^
  -H "Content-Type: application/json" ^
  -d "{\"question\":\"How do I reset my password?\"}"
```

Ingest request:

```bash
curl -X POST http://localhost:8000/ingest ^
  -H "Content-Type: application/json" ^
  -d "{\"append\":false}"
```

## 7. React integration example

```ts
const res = await fetch("http://localhost:8000/chat", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ question: userInput }),
});
const data = await res.json();
// data.answer, data.used_context
```

## 8. Tuning for stricter behavior

In `.env`:
- Increase `MIN_RELEVANCE` (for example `0.65`) to be stricter.
- Increase `TOP_K` if some valid answers are missed.
- Lower `TOP_K`, `MAX_CONTEXT_DOCS`, or `LLM_NUM_PREDICT` if replies are still too slow.
- Increase `LLM_NUM_PREDICT` or `MAX_ANSWER_SENTENCES` if answers are too short.
- Lower `FAST_PATH_MIN_RELEVANCE` if you want the bot to use direct extractive answers more often instead of waiting for the LLM.
- Increase `CHUNK_SIZE` for broader context per chunk.
- Set `AZURE_DEVOPS_ORG`, `AZURE_DEVOPS_PROJECT`, and `AZURE_DEVOPS_PAT` to ingest all private project wikis.
- Set `AZURE_DEVOPS_PROJECTS` (comma-separated) to ingest wikis from multiple projects in the same org.
- Set `AZURE_DEVOPS_WIKI` only when you want to limit ingestion to one wiki (single-project mode only).

## Notes

- The strict prompt and retrieval threshold reduce hallucinations.
- Repeated questions are cached in memory, and highly relevant single-page answers can skip the LLM for lower latency.
- No model can guarantee perfection, but this setup is designed to reject out-of-scope questions by default.
