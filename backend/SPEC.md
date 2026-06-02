# Backend Specification — Document-Only Chatbot API

## 1. Overview

A document-grounded (RAG) chatbot backend that answers questions **only** from
ingested source material. If the answer is not present in the indexed documents,
it returns a fixed refusal string:

> `I don't know based on the provided documents.`

The system retrieves relevant chunks from a local vector store, optionally builds
an extractive answer, and otherwise asks a locally hosted LLM to answer strictly
from the retrieved context. Access to projects is enforced per-request via the
caller's Azure DevOps Personal Access Token (PAT).

Ingestion sources: local files, URLs, Azure DevOps wikis, and **SharePoint**
(Microsoft Graph, app-only auth) — every supported file in every site/folder the
registered app can read.

- **Language / runtime:** Python 3 (uses `from __future__ import annotations`)
- **Web framework:** FastAPI (served by Uvicorn)
- **RAG framework:** LangChain
- **Vector store:** Chroma (persisted to disk)
- **Embeddings:** FastEmbed (ONNX, local), default `BAAI/bge-small-en-v1.5`
- **LLM:** Ollama (`ChatOllama`), default `llama3.2:3b`, `temperature=0`

## 2. Goals & Non-Goals

### Goals
- Answer strictly from ingested documents; reject out-of-scope questions.
- Ingest content from local files, arbitrary URLs, and Azure DevOps wikis.
- Enforce project-level access control using the caller's PAT (not the server's).
- Run fully locally/offline after first model download (no external LLM API).
- Keep latency low via an in-memory answer cache and an optional extractive
  "fast path" that can skip the LLM.

### Non-Goals
- Multi-turn conversational memory (each `/chat` request is stateless).
- User identity/auth beyond the Azure DevOps PAT.
- Streaming responses.
- Guaranteed hallucination-free output (mitigated, not eliminated).

## 3. Architecture

```
              ┌─────────────┐
   client ───▶│  FastAPI    │  app/main.py  (HTTP layer, PAT auth, project scoping)
              └──────┬──────┘
                     │
        ┌────────────┼─────────────────────────┐
        ▼            ▼                          ▼
  RAGService    run_ingestion          list_user_accessible_projects
  (app/rag.py)  (app/ingestion.py)     (app/document_loader.py)
        │            │
        ▼            ▼
   Chroma vectorstore  ◀── load_documents + split_documents
   (app/vectorstore.py)    (app/document_loader.py)
        │
        ├── FastEmbed embeddings
        └── ChatOllama LLM
```

### Module responsibilities

| Module | Responsibility |
|---|---|
| `app/main.py` | FastAPI app, routes, CORS, PAT extraction, per-PAT project-access cache, project authorization. |
| `app/config.py` | `Settings` (pydantic-settings) loaded from `.env`; validators; derived lists. |
| `app/schemas.py` | Pydantic request/response models. |
| `app/document_loader.py` | Load documents from files, URLs, Azure DevOps wikis; list accessible projects; chunk splitting. |
| `app/ingestion.py` | Orchestrate load → split → embed → upsert into Chroma; deterministic chunk IDs. |
| `app/vectorstore.py` | Build embeddings + Chroma store; clear store. |
| `app/rag.py` | Retrieval, relevance filtering, fast-path extractive answer, LLM prompting, answer cache. |
| `ingest.py` | CLI entry point for ingestion (`python ingest.py [--append]`). |

## 4. Configuration (`app/config.py`)

Loaded from environment / `.env` (case-insensitive, extra keys ignored).

### Retrieval / indexing
| Setting | Default | Meaning |
|---|---|---|
| `DATA_DIR` | `data` | Local documents directory. |
| `URLS_FILE` | `data/urls.txt` | Newline-delimited URL list (`#` comments allowed). |
| `VECTORSTORE_DIR` | `vectorstore` | Chroma persist directory. |
| `COLLECTION_NAME` | `docs` | Chroma collection name. |
| `CHUNK_SIZE` | `900` | Character chunk size. |
| `CHUNK_OVERLAP` | `150` | Character overlap between chunks. |
| `TOP_K` | `4` | Number of nearest chunks retrieved. |
| `MAX_CONTEXT_DOCS` | `3` | Max chunks passed into the prompt. |
| `MAX_CONTEXT_CHARS` | `3500` | Char budget for assembled context. |
| `MIN_RELEVANCE` | `0.55` | Min similarity (0–1) to accept a chunk. Validated to [0,1]. |

### Fast path (extractive, LLM-skipping)
| Setting | Default | Meaning |
|---|---|---|
| `ENABLE_FAST_PATH` | `False` | Enable extractive answer attempt before the LLM. |
| `FAST_PATH_MIN_RELEVANCE` | `0.6` | Min top-doc similarity to attempt fast path. Validated to [0,1]. |
| `FAST_PATH_MIN_OVERLAP` | `0.2` | Min term-overlap score for a passage. Validated to [0,1]. |
| `FAST_PATH_MAX_DOCS` | `2` | Docs scanned for candidate passages. |
| `MAX_ANSWER_SENTENCES` | `6` | Max passages in an extractive answer. |
| `ANSWER_CACHE_SIZE` | `128` | LRU answer cache capacity. |

### LLM / embeddings
| Setting | Default | Meaning |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL. |
| `LLM_MODEL` | `llama3.2:3b` | Ollama chat model. |
| `LLM_NUM_PREDICT` | `192` | Max tokens generated. |
| `OLLAMA_KEEP_ALIVE` | `30m` | Keep model loaded in Ollama. |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | FastEmbed model name. |
| `EMBEDDING_THREADS` | `None` | Optional FastEmbed thread count. |
| `EMBEDDING_CACHE_DIR` | `None` | Optional FastEmbed cache directory. |

> Note: `.env.example` ships some legacy/divergent values (`EMBEDDING_MODEL=nomic-embed-text`,
> `EMBEDDING_KEEP_ALIVE`) that do not match the code defaults; code defaults above are authoritative.

### Azure DevOps / CORS
| Setting | Default | Meaning |
|---|---|---|
| `AZURE_DEVOPS_PAT` | `None` | Server-side PAT used **only for ingestion**. |
| `AZURE_DEVOPS_ORG` | `None` | Organization; required for `/chat` and `/projects` access checks. |
| `AZURE_DEVOPS_PROJECT` | `None` | Single-project fallback. |
| `AZURE_DEVOPS_PROJECTS` | `None` | Comma-separated project list (multi-project ingest). |
| `AZURE_DEVOPS_WIKI` | `None` | Specific wiki id/name (honored only in single-project mode). |
| `AZURE_DEVOPS_WIKI_PATH` | `/` | Root wiki path to ingest from. |
| `AZURE_DEVOPS_API_VERSION` | `7.1` | Azure DevOps REST API version. |
| `CORS_ORIGINS` | `*` | `*` or comma-separated allowed origins. |

### SharePoint (Microsoft Graph, app-only / client-credentials)
| Setting | Default | Meaning |
|---|---|---|
| `SHAREPOINT_TENANT_ID` | `None` | Azure AD tenant id. |
| `SHAREPOINT_CLIENT_ID` | `None` | App registration (client) id. |
| `SHAREPOINT_CLIENT_SECRET` | `None` | App client secret. |
| `SHAREPOINT_SITE` | `None` | Optional: limit to one site (Graph site id or `host:/sites/Name`). Blank = all readable sites. |
| `SHAREPOINT_FOLDERS` | `None` | Optional comma-separated folder names/paths to restrict ingestion. Blank = all folders. Also the seam for later per-folder answer scoping. |
| `SHAREPOINT_GRAPH_BASE_URL` | `https://graph.microsoft.com/v1.0` | Microsoft Graph base URL. |
| `SHAREPOINT_LOGIN_BASE_URL` | `https://login.microsoftonline.com` | Azure AD token authority base URL. |

SharePoint ingestion activates only when all three creds (`TENANT_ID`, `CLIENT_ID`,
`CLIENT_SECRET`) are present. The app registration needs **`Sites.Read.All`** /
**`Files.Read.All`** (Application) permissions with admin consent.

Derived properties:
- `cors_origins_list` → `["*"]` or parsed list.
- `azure_devops_projects_list` → de-duplicated list from `AZURE_DEVOPS_PROJECTS`, falling back to `AZURE_DEVOPS_PROJECT`.
- `sharepoint_enabled` → `True` when tenant id + client id + client secret are all set.
- `sharepoint_folders_list` → de-duplicated list parsed from `SHAREPOINT_FOLDERS`.

## 5. HTTP API (`app/main.py`)

App: `FastAPI(title="Document-Only Chatbot API", version="1.0.0")`, CORS middleware
from `cors_origins_list`, credentials allowed, all methods/headers.

### Auth model
- `/chat` and `/projects` require header **`X-Azure-Devops-Pat`** (missing/blank → `401`).
- The PAT determines which projects the caller may query. The server resolves
  accessible projects by calling Azure DevOps `_apis/projects` with the caller's PAT.
- A per-PAT cache (`sha256(pat)` fingerprint) holds the project list for
  `PAT_PROJECTS_TTL_SECONDS = 300` seconds, guarded by a lock.
- The server's `AZURE_DEVOPS_*` env only controls **what is ingested**, never what
  a user may ask about. Non-ingested projects simply return no results (no leak).

### Endpoints

#### `GET /health`
Returns store/model status. No auth.
```json
{ "status": "ok", "indexed_chunks": 123, "llm_model": "llama3.2:3b", "embedding_model": "BAAI/bge-small-en-v1.5" }
```

#### `POST /ingest`
Body `IngestRequest`: `{ "append": false }`.
Runs ingestion, then reloads the RAG vector store. Loader errors → `400`.
Returns `IngestResponse`:
```json
{ "documents_loaded": 10, "chunks_indexed": 42, "total_chunks_in_store": 42 }
```

#### `POST /chat`
Header: `X-Azure-Devops-Pat` (required).
Body `ChatRequest`: `{ "question": "...", "project": "optional-project-name" }`.

Flow:
1. Empty question → `400`.
2. Missing PAT → `401`.
3. Resolve accessible projects; empty → `403`.
4. If `project` provided and not in accessible set (case-insensitive) → `403`;
   otherwise canonicalize to the AzDO-cased name.
5. If index is empty (`indexed_document_count() == 0`) → `400` ("Run /ingest first").
6. Answer scoped to the single requested project, or to all accessible projects.

Returns `ChatResponse`:
```json
{ "answer": "...", "used_context": true, "project": "Selected-Project-or-null" }
```

#### `GET /projects`
Header: `X-Azure-Devops-Pat` (required).
Returns `ProjectsResponse`: `{ "projects": ["A", "B"] }` — projects the PAT can read.

### Error codes summary
| Code | Condition |
|---|---|
| 400 | Empty question; ingest loader error; empty index on `/chat`. |
| 401 | Missing PAT; Azure DevOps auth failure (`DocumentLoadError`). |
| 403 | PAT has no project access; requested project not authorized. |
| 500 | `AZURE_DEVOPS_ORG` not configured on server. |

## 6. Schemas (`app/schemas.py`)

```python
ChatRequest    { question: str (min_length=1), project: str | None }
ChatResponse   { answer: str, used_context: bool, project: str | None }
IngestRequest  { append: bool = False }
IngestResponse { documents_loaded: int, chunks_indexed: int, total_chunks_in_store: int }
ProjectsResponse { projects: list[str] }
```

## 7. Document loading (`app/document_loader.py`)

`load_documents(...)` aggregates from four sources:

1. **Local files** (`load_local_documents`): recursive scan of `DATA_DIR`.
   Supported extensions: `.txt .md .markdown .pdf .html .htm .docx .xlsx`.
   - PDFs (`pypdf`): one `Document` per non-empty page, with `page` metadata.
   - HTML: stripped of `script/style/noscript`, text extracted via BeautifulSoup.
   - Word `.docx` (`python-docx`): paragraph text plus table rows
     (`cell | cell`), one `Document` per file.
   - Excel `.xlsx` (`openpyxl`, read-only/data-only): per sheet, non-empty rows
     joined as `cell | cell`, prefixed with a `# Sheet: <name>` header.
   - Text/Markdown: read as UTF-8 (errors ignored).
   - All text is line-trimmed and blank lines removed (`_clean_text`).
2. **URLs** (`load_url_documents`): reads `URLS_FILE`, fetches each URL with a
   browser-like User-Agent. Google Docs share links are rewritten to the
   `export?format=txt` endpoint. Azure DevOps hosts (`dev.azure.com`,
   `*.visualstudio.com`) get PAT Basic-auth headers automatically. HTML responses
   are text-extracted; other content used as-is.
3. **Azure DevOps wikis** (when PAT + org + project(s) present):
   - Single wiki only if `AZURE_DEVOPS_WIKI` set **and** exactly one project.
   - Otherwise `load_all_azure_devops_project_wikis` enumerates every wiki in each
     project and recurses all pages (`recursionLevel=full`).
   - Per-page content fetched if not inlined. Each chunk carries metadata:
     `source, path, page_id, project, wiki, type="azure_devops_wiki"`.
   - A failing project logs a warning and is skipped (ingest continues).

4. **SharePoint** (`load_sharepoint_documents`, when `SHAREPOINT_TENANT_ID` +
   `CLIENT_ID` + `CLIENT_SECRET` are set):
   - Acquires an app-only token via the OAuth2 client-credentials flow
     (`POST {login}/{tenant}/oauth2/v2.0/token`, scope
     `https://graph.microsoft.com/.default`).
   - Resolves sites (`_resolve_sharepoint_sites`): a specific `SHAREPOINT_SITE`
     (addressable path or id, with search fallback) or **all** readable sites via
     `GET /sites?search=*`.
   - For each site → lists drives (`GET /sites/{id}/drives`) → recursively walks
     every folder from `root` (`GET /drives/{id}/items/{item}/children`,
     `_walk_sharepoint_folder`), paginating through `@odata.nextLink`.
   - Each supported file (same `SUPPORTED_EXTENSIONS`) is downloaded
     (`/items/{id}/content`) and text-extracted from bytes via `_extract_file_text`
     (`_extract_pdf_bytes`, `_extract_html_bytes`, `_extract_docx_bytes`,
     `_extract_xlsx_bytes`, or UTF-8 decode). Unsupported extensions (e.g. legacy
     `.doc`/`.xls`, `.pptx`) are skipped.
   - Optional `SHAREPOINT_FOLDERS` restricts ingestion to matching folders
     (`_folder_in_scope`: contiguous path-segment match, case-insensitive).
   - Each chunk carries metadata: `source` (webUrl), `folder`, `site`, `drive`,
     `file_name`, `type="sharepoint"`.
   - Failures degrade gracefully: a bad token raises and the whole SharePoint pass
     is skipped with a warning; a failing site/drive logs a warning and the rest
     continue. `401/403` from Graph → `DocumentLoadError` with a permissions hint.

`list_user_accessible_projects(org, pat, api_version)` → GET
`https://dev.azure.com/{org}/_apis/projects?stateFilter=wellFormed`, returns
de-duplicated project names. `401/403` → `DocumentLoadError`.

`split_documents(...)` uses `RecursiveCharacterTextSplitter`
(separators `["\n\n", "\n", ". ", " ", ""]`) and stamps each chunk with `chunk_id`.

Azure auth: Basic auth with `base64(":" + PAT)`.

## 8. Ingestion (`app/ingestion.py` + `ingest.py`)

`run_ingestion(append=False, settings=None)`:
1. Build Chroma store.
2. If not append → `clear_vectorstore` (delete all existing IDs).
3. `load_documents(...)` then `split_documents(...)`.
4. Compute deterministic ID per chunk: `sha256(source|page|chunk_id|content)` —
   makes re-ingest idempotent (same content → same ID → upsert, no duplicates).
5. `vectorstore.add_documents(chunks, ids=ids)`.
6. Return counts `{documents_loaded, chunks_indexed, total_chunks_in_store}`.

CLI: `python ingest.py [--append]` prints the result JSON.

## 9. RAG service (`app/rag.py`)

`RAGService(settings)` holds the Chroma store, a `ChatOllama` LLM, and an LRU
answer cache (`OrderedDict`) with a `_cache_revision` counter.

- `reload_vectorstore()` — rebuild store, bump revision, clear cache.
- `indexed_document_count()` — `vectorstore._collection.count()`.

### `answer(question, project=None, allowed_projects=None)`

1. **Scope:** if a single `project` is given, filter to it. Else if
   `allowed_projects` given, scope to that set (sorted, de-duplicated). Empty
   allowed set → immediate `UNKNOWN_ANSWER`.
2. **Cache:** key = `{revision}:{scope}:{normalized question}`; LRU hit returns cached.
3. **Retrieve:** `similarity_search_with_score(question, k=top_k, filter=…)`.
   Chroma filter is `{"project": name}` for one project or `{"project": {"$in": [...]}}`
   for multiple.
4. **Score:** distance → similarity via `1 / (1 + max(distance, 0))`. Keep chunks
   with `similarity >= MIN_RELEVANCE`.
5. **Near-miss fallback:** if nothing passes but matches exist, accept the top
   match only if term-overlap of question vs. its text `>= 0.5`.
6. If still empty → cache + return `UNKNOWN_ANSWER` (`used_context=False`).
7. Trim to `MAX_CONTEXT_DOCS`; compute display sources.
8. **Fast path** (if `ENABLE_FAST_PATH`): rank candidate passages by
   term-overlap + `0.15 * doc_score`, select diverse passages above thresholds,
   return an extractive answer (joined sentences or a bullet list) if any qualify.
9. **LLM path:** assemble context (capped by `MAX_CONTEXT_CHARS`, each snippet
   labeled with source/page/similarity), build the strict prompt, invoke the LLM,
   sanitize the output (strip leaked `source=`/`similarity=`/index tags). Empty →
   `UNKNOWN_ANSWER`.
10. Cache and return `RAGResult(answer, sources, used_context=True)`.

### Prompt contract
- "Answer like a helpful teammate" but **only** from context.
- If not explicitly in context, reply exactly with `UNKNOWN_ANSWER`.
- No outside knowledge, no guessing.
- Moderate detail: 1–3 short paragraphs, or a short ordered list for steps.
- Return only the final answer text; no metadata/index/similarity labels.

### Helper functions
`_distance_to_similarity`, `_term_overlap_ratio` (alphanumeric terms ≥3 chars),
`_sanitize_answer`, `_candidate_passages` (sentence split, 20–260 char window,
dedup), `_select_diverse_passages` (dedup + ≥0.8 overlap suppression),
`_format_extractive_answer`.

## 10. Vector store (`app/vectorstore.py`)

- `get_embeddings(settings)` → `FastEmbedEmbeddings(model_name, [threads], [cache_dir])`.
- `get_vectorstore(settings)` → `Chroma(collection_name, embedding_function, persist_directory)`.
- `clear_vectorstore(store)` → deletes all IDs, returns count removed.

## 11. Dependencies (`requirements.txt`)

`fastapi`, `uvicorn[standard]`, `pydantic-settings`, `python-dotenv`,
`langchain`, `langchain-community`, `langchain-chroma`, `langchain-ollama`,
`chromadb`, `fastembed`, `beautifulsoup4`, `pypdf`, `python-docx`, `openpyxl`,
`requests`.

External runtime dependency: a running **Ollama** server with the chat model pulled
(`ollama pull llama3.2:3b`).

## 12. Run & operate

```bash
pip install -r requirements.txt
Copy-Item .env.example .env          # then edit
ollama pull llama3.2:3b
python ingest.py                     # build index (use --append to add)
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## 13. Security & access-control notes

- Project authorization is always sourced from the **caller's** PAT, re-checked
  (with a 5-minute cache) on every `/chat` and `/projects` call.
- The vector store may contain chunks from projects a given caller cannot access;
  retrieval is filtered by the caller's accessible-project set so cross-project
  data is never returned.
- PATs are never logged; only their `sha256` fingerprint is used as a cache key.
- Server-side `AZURE_DEVOPS_PAT` is ingestion-only and distinct from caller PATs.

## 14. Future work — per-folder SharePoint answer scoping

The current request ingests **all** accessible SharePoint files. To later restrict
answers to a named folder (mirroring the AzDO `project` filter), the seam is already
in place — each chunk carries `folder` metadata:

1. Add an optional `folder: str | None` to `ChatRequest` (parallel to `project`).
2. In `RAGService.answer(...)`, add a `folder` arg and apply a Chroma filter
   `{"folder": name}` (or `{"folder": {"$in": [...]}}`) exactly like the existing
   project filter; include `folder` in the cache key.
3. Optionally expose accessible folders via a `/folders` endpoint by reading the
   distinct `folder` values from the collection.

`SHAREPOINT_FOLDERS` already restricts *ingestion* scope using the same folder
notion, so the metadata and matching logic (`_folder_in_scope`) are reusable.

## 15. Known gaps / observations

- Relies on Chroma private `_collection` API (`indexed_document_count`,
  ingestion count) — could break on Chroma upgrades.
- `.env.example` embedding settings diverge from code defaults (see §4 note).
- No automated tests in the backend tree.
- Answer cache and per-PAT project cache are in-memory and per-process (not shared
  across workers); running multiple Uvicorn workers gives each its own caches.
- No rate limiting, request size limits, or streaming.
- Supported file types: pdf, txt, md/markdown, html/htm, docx, xlsx. Legacy binary
  Office formats (`.doc`, `.xls`) and `.pptx` are skipped — adding them needs extra
  parsers in `_extract_file_text`. The `.docx`/`.xlsx` parsers import lazily and
  degrade to an empty string (with a warning) if the optional deps are missing.
- SharePoint access is governed by the app registration's granted permissions, not
  per-end-user identity; unlike the AzDO PAT flow, `/chat` does not currently scope
  results by the caller's SharePoint access.
