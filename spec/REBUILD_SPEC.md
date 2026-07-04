# Build Spec — Enterprise Knowledge Assistant (Azure Wiki + SharePoint)

A clean-slate specification for a **new** RAG chatbot that answers questions
grounded in **Azure DevOps wikis** and **SharePoint** content. This design does
**not** carry over the current implementation; it is written so a team can build
the whole system from scratch.

---

## 1. Product goals

1. **Grounded answers only.** Every answer is derived from indexed Azure Wiki /
   SharePoint content, with **inline citations** to the exact source page/file.
   When the corpus doesn't support an answer, say so plainly.
2. **Per-project RBAC over both sources.** Users authenticate with their own Azure
   DevOps **PAT**; the bot lists the projects that PAT can read, the user selects
   **one**, and the bot answers **only** from that selected project. This applies
   to **both** Azure DevOps wiki *and* SharePoint content — every chunk is tagged
   with a `project`, and SharePoint documents are mapped to projects at ingestion
   (see §3). A user can never get an answer from a project their PAT can't access.
3. **Fresh.** Edits and deletions at the source propagate to answers within
   minutes, not on a manual full rebuild.
4. **Conversational.** Multi-turn, with follow-up questions understood in context.
5. **Fast & transparent.** Streamed responses, visible sources, honest confidence.

### Non-goals (v1)
- Writing back to wikis/SharePoint (read-only assistant).
- Cross-tenant federation.
- Non-Microsoft sources (kept as a pluggable connector interface for later).

---

## 2. High-level architecture

```
                          ┌────────────────────────────────────────┐
   Browser (PAT) ───────▶ │  API Gateway / BFF  (auth, rate limit)  │
                          └───────────────┬────────────────────────┘
                                          │  (streamed answers, SSE)
                    ┌─────────────────────┼──────────────────────────┐
                    ▼                     ▼                          ▼
             Query Service         Ingestion Service           Sync Scheduler
          (retrieve → rerank      (parse → chunk → embed      (Graph delta +
           → generate → cite)      → upsert, tag `project`)    AzDO change feed)
                    │                     │                          │
        ┌───────────┼──────────┐          ▼                          │
        ▼           ▼          ▼    Object store (raw blobs)          │
   Vector DB    Reranker    LLM     + Metadata DB (project,hash,etag)◀┘
  (Qdrant /     (cross-     (Ollama
   pgvector /    encoder)    gemma4:e4b)
   Chroma)
```

**Components**
- **BFF / API gateway** — validates the caller's PAT, enforces per-project authz, rate-limits, streams.
- **Query service** — retrieval (scoped to the selected project) → rerank → grounded generation with citations.
- **Ingestion service** — connectors + parsing + chunking + embedding + upsert; tags every chunk with a `project`. Uses the **bot's dedicated ingestion PAT** (never a user PAT).
- **Sync scheduler** — incremental delta sync; drives the ingestion service.
- **Vector DB** — self-hosted dense + sparse index with metadata filtering
  (Qdrant / pgvector / Chroma — **not** Azure AI Search).
- **Metadata DB** — content hash, etag, `project`, source pointers, sync state.
- **Object store** — original file blobs (for previews / re-parse).

Each service is independently deployable and stateless except for its datastore,
so the API scales horizontally.

---

## 3. Identity, authentication & authorization

This build has **no Entra ID SSO**. Users are identified by their own **Azure
DevOps Personal Access Token (PAT)**, supplied per request. Access is decided at
**project granularity**: the PAT determines which projects a user may read, and
**both** wiki and SharePoint content is scoped to a single **selected project**.
There are two distinct PATs in the system — never confuse them:

| PAT | Held by | Used for |
|---|---|---|
| **User PAT** | each end user (pasted in the browser) | identifying the user and resolving *their* accessible projects at query time |
| **Ingestion PAT** | the bot (server-side secret) | ingesting **all** projects' wikis; never exposed to users |

### 3.1 User authentication & project selection
- The caller sends their **User PAT** in a request header. The server resolves the
  projects that PAT can read via `GET /_apis/projects` (cached briefly, keyed by a
  `sha256` fingerprint — the raw PAT is never persisted).
- The UI shows the user their accessible projects; the user **selects one**. Every
  `/chat` call carries that `project`, and the server re-validates it is in the
  caller's accessible set (else `403`).
- The browser keeps the PAT **in memory**; if it must persist, an `HttpOnly`,
  `Secure` cookie — **never `localStorage`**.

### 3.2 Project RBAC — enforced for wiki AND SharePoint
Every indexed chunk, regardless of source, carries a `project` tag. At query time
the retrieval filter is `project == <selected project>`, and the selected project
must belong to the caller PAT's accessible set. Therefore:

- A user only ever retrieves/cites content — wiki **or** SharePoint — for a project
  their PAT grants access to.
- Answers are scoped to the **one** selected project, not the union of accessible
  projects (keeps context focused and authorization simple).

Access is genuine and identity-derived: the PAT *is* the identity, and project
membership *is* the permission.

### 3.3 How SharePoint content gets its `project` (site→project mapping)
SharePoint has no native notion of an Azure DevOps project, so the association is
made at ingestion via a configured **site→project (and optionally folder→project)
mapping**:

- The bot ingests SharePoint documents from **all configured sites** using the
  app-only identity, and stamps each document with the `project` its site/folder
  maps to.
- At query time those SharePoint chunks are filtered by the same
  `project == selected` rule, so a user only sees SharePoint answers for projects
  they can access — even though ingestion itself is app-only and org-wide.
- **Unmapped SharePoint content fails closed:** if a site/folder maps to no
  project, its documents are either skipped or assigned to an explicit, opt-in
  "common" project that every user can access. Default = skip (never expose
  unmapped content).

This is project-level RBAC, not per-file SharePoint ACL — matching the requirement
"users only get answers for projects they have access to." True per-file SharePoint
trimming would require SSO + on-behalf-of and is out of scope.

### 3.4 Ingestion identity & project discovery
- **Azure DevOps:** the server-side **Ingestion PAT** reads every project's wiki.
  If the env project list is **empty**, the bot calls `GET /_apis/projects` with
  the Ingestion PAT and ingests **all** discovered projects; if the list is set,
  only those are ingested.
- **SharePoint:** an **app registration** with least-privilege application
  permissions (`Sites.Read.All`, `Files.Read.All`), used only to read content —
  never to decide what a user may see.

**Invariant:** neither source ever returns content outside the caller PAT's
accessible projects, because every chunk is `project`-tagged and every query is
filtered to a single authorized project.

---

## 4. Connectors & ingestion

### 4.1 Connector interface
A connector yields a stream of `SourceItem`:

```
SourceItem {
  source_id: str            # stable unique id (Graph driveItem id / wiki page id)
  source_type: "sharepoint" | "azure_wiki"
  uri: str                  # canonical, clickable URL
  title: str
  breadcrumb: [str]         # site>library>folder / wiki>path
  etag / last_modified      # for change detection
  content_hash: str         # sha256 of extracted text
  raw_bytes | text          # payload
  project: str              # REQUIRED — the AzDO project this item is scoped to.
                            #   wiki: the page's own project.
                            #   sharepoint: resolved via the site→project map (§3.3).
                            #   drives query-time RBAC for BOTH sources (§3.2).
  metadata: {...}           # site, library, wiki, author, ...
}
```

### 4.2 SharePoint connector (Microsoft Graph, app-only)
- Ingest **all configured sites** (every drive/folder), using the app registration.
- **Incremental** via Graph **delta queries** per drive
  (`/drives/{id}/root/delta`): first run = full, later runs = only changed/
  deleted items. Deleted items carry a `deleted` facet → remove from index.
- **Resolve `project` per item** via the configured **site→project map** (and
  optional folder→project overrides). Stamp it onto the item so query-time RBAC
  (§3.2) can filter SharePoint exactly like wiki. Unmapped items **fail closed**
  (skipped, or routed to an opt-in "common" project — §3.3).
- Extract text per file type (see §4.5).

### 4.3 Azure DevOps wiki connector
- **Project discovery:** if the env project list is set, ingest those; if it is
  **empty**, call `GET /_apis/projects` with the **Ingestion PAT** and ingest
  **every** project in the org.
- Enumerate wikis per project; pull pages recursively.
- Change detection via page version / last-updated; deletions reconciled by
  diffing the current page set against indexed `source_id`s.
- Stamp each chunk with its `project` — this is what query-time RBAC (§3.2)
  filters on against the caller PAT's accessible projects.

### 4.4 Ingestion pipeline (per item)
```
fetch → extract text → structure-aware chunk → embed → upsert(vector + metadata)
```
- **Idempotent upsert** keyed on `source_id`; skip when `content_hash` unchanged.
- **Delete** when the connector reports the item removed.
- Runs in a **background worker** off a job queue — never inside the request path.

### 4.5 Parsing
- Markdown / HTML → normalized Markdown (preserve headings, tables, code).
- PDF → text; **OCR fallback** for image-only pages.
- DOCX / PPTX / XLSX → structured text; tables rendered as Markdown.
- Attach heading breadcrumbs to every chunk.

### 4.6 Chunking
- **Structure-aware**: split on headings/sections, keep tables and code blocks
  intact, target ~500–800 tokens with small overlap.
- Store parent-document pointers so retrieval can expand a hit to its section.

### 4.7 Ingestion configuration (env)
| Setting | Meaning |
|---|---|
| `INGESTION_PAT` | Bot-owned privileged PAT for reading all project wikis. **Never sent to clients.** |
| `AZURE_DEVOPS_ORG` | Organization to enumerate/ingest. |
| `AZURE_DEVOPS_PROJECTS` | Comma-separated project list to ingest. **If empty → discover and ingest all projects** via `INGESTION_PAT` (§4.3). |
| `SHAREPOINT_TENANT_ID` / `_CLIENT_ID` / `_CLIENT_SECRET` | App-registration credentials (app-only Graph). |
| `SHAREPOINT_SITES` | Sites to ingest (all listed sites are crawled in full). |
| `SHAREPOINT_SITE_PROJECT_MAP` | **site → project** (and optional `site/folder → project`) mapping that assigns each SharePoint doc its `project` for RBAC (§3.3). |
| `SHAREPOINT_UNMAPPED_POLICY` | `skip` (default, fail-closed) or `common:<project>` to route unmapped docs to an all-users project. |

The **User PAT** is never configured here — it arrives only at request time.

---

## 5. Retrieval & generation

### 5.1 Retrieval pipeline
```
verify project access → query rewrite → hybrid search (BM25 + dense, RRF)
→ project filter → cross-encoder rerank → parent-doc expansion → context assembly
```
0. **Verify project access** — confirm the request's `project` is in the caller
   PAT's accessible set (else `403`). Everything downstream is scoped to it.
1. **Query rewrite** — condense chat history into a standalone question; expand
   acronyms; optionally generate a couple of query variants.
2. **Hybrid search** — sparse (BM25) and dense (embeddings) retrieved in
   parallel and fused with **Reciprocal Rank Fusion**.
3. **Project filter** — apply `project == <selected project>` at the vector-DB
   query level for **both** wiki and SharePoint chunks, so nothing outside the
   authorized project is ever scored into context (§3.2).
4. **Rerank** — cross-encoder reranker over top-N candidates; the reranker score
   is the relevance gate (calibrated on the eval set), replacing hand-tuned
   distance heuristics.
5. **Assemble** — expand top chunks to their parent sections, dedupe, and pack to
   a token budget.

### 5.2 Generation
- Grounded prompt: answer **only** from provided context; emit **inline citation
  markers** `[n]` mapping to a sources array; if unsupported, return an explicit
  "not found in the knowledge base" message.
- **Stream** tokens to the client (SSE).
- **LLM: `gemma4:e4b` via Ollama** (local/private — data stays on the network).
  - `temperature=0` for grounding.
  - `gemma4:e4b` is a **reasoning ("thinking") model** — disable thinking so it
    returns a direct answer instead of spending the token budget on internal
    reasoning (matches the `LLM_REASONING=false` note already in `config.py`).
  - The backend stays behind an interface so the model is a config swap, not a
    code change.
- Return a **faithfulness/confidence** signal alongside the answer.

### 5.3 Response contract (streamed)
```
event: sources   → [{ n, title, uri, source_type, breadcrumb }]
event: token     → "partial text…"          (repeated)
event: done      → { confidence, used_context, message_id }
```

---

## 6. API surface

| Method | Path | Auth | Purpose |
|---|---|---|---|
| `GET`  | `/v1/projects` | User PAT | List the projects the caller's PAT can access (drives the project picker). |
| `POST` | `/v1/chat` | User PAT | Ask a question **within the selected project**; **streams** answer + citations. |
| `GET`  | `/v1/conversations` | User PAT | List the caller's conversations. |
| `GET`  | `/v1/conversations/{id}` | User PAT | Fetch history. |
| `GET`  | `/v1/sources/{id}` | User PAT | Source preview (project access re-checked via PAT). |
| `POST` | `/v1/admin/sync` | admin token | Trigger/replan a sync job (enqueues). |
| `GET`  | `/v1/admin/sync/{job}` | admin token | Job status. |
| `GET`  | `/healthz` `/readyz` | none | Liveness / readiness. |
| `GET`  | `/metrics` | infra | Prometheus metrics. |

Auth = the caller's **User PAT** in a request header (no SSO). Admin endpoints use
a separate server-side admin token; the **Ingestion PAT** is never accepted from a
client.

`/v1/chat` request — `project` is **required** and must be in the caller's
accessible set (else `403`):
```json
{ "conversation_id": "uuid|null", "project": "Selected-Project", "message": "…",
  "filters": { "source_type": ["azure_wiki", "sharepoint"] } }
```
`filters` are **narrowing only** within the selected project; they can never widen
beyond it.

---

## 7. Data model (metadata DB)

```
documents(
  source_id PK, source_type, uri, title, breadcrumb,
  etag, last_modified, content_hash,
  site, library, project, wiki, author, indexed_at, deleted_at
)
chunks(
  chunk_id PK, source_id FK, ordinal, heading_path,
  token_count, vector_id, created_at
)
sync_state(connector, drive_or_project, delta_token, last_run, status)
conversations(id, user_key, created_at)   -- user_key = sha256(PAT) fingerprint
messages(id, conversation_id, role, content, citations[], confidence, created_at)
```
Vector DB stores the embedding + a filterable copy of `project` (present on **both**
wiki and SharePoint chunks), `source_type`, and scoping fields. `project` is the
RBAC filter key (§3.2).

---

## 8. Technology choices (recommended defaults)

| Concern | Recommendation | Why |
|---|---|---|
| API | Python + FastAPI (async) | Streaming, mature RAG ecosystem. |
| Vector DB | **Qdrant** (or pgvector / Chroma) | Self-hosted, free, native hybrid (sparse+dense) search + metadata filters. **No Azure AI Search.** |
| Embeddings | A strong current embedding model (local or hosted) | Better recall than `bge-small`. |
| Reranker | Cross-encoder (`bge-reranker` / hosted rerank) | Biggest single quality lever. |
| LLM | **`gemma4:e4b` via Ollama** (local), thinking disabled | Private/local; behind an interface so it can be swapped. |
| Queue/worker | Celery/RQ/Arq + Redis, or cloud queue | Off-request ingestion. |
| Cache | Redis | Shared across workers (answers, accessible-project lookups). |
| Auth | Azure DevOps **PAT** (no SSO) — user PAT at query, bot Ingestion PAT for ingest | Per-project RBAC for **both** wiki and SharePoint. |
| Frontend | React + Vite, SSE streaming | PAT entry (in-memory), inline citations, source preview. |
| Observability | OpenTelemetry + Prometheus + structured logs | Trace retrieval + generation spans. |

All backends are behind interfaces so any one can be swapped.

---

## 9. Evaluation & quality gates

- **Golden Q&A set** curated from real wiki/SharePoint content, with expected
  source documents.
- Metrics tracked in CI on every change to retrieval/generation:
  - **Retrieval hit-rate / recall@k** (did we fetch the right doc?)
  - **Faithfulness** (is every claim supported by cited context?)
  - **Answer relevance** and **citation correctness**.
- **Project-RBAC tests (blocking)**: for a fixture User PAT with access to project
  A but not B, assert that **no chunk of either source** (wiki *or* SharePoint)
  belonging to B is ever retrieved or cited, and that requesting project B returns
  `403`. Include a SharePoint doc mapped to B to prove the site→project mapping is
  enforced at query time.
- Load/latency budget: p95 time-to-first-token and full-answer targets.

---

## 10. Observability & operations

- Structured JSON logs; **never log tokens, PATs, or document content**.
- Per-request trace: rewrite → retrieve → rerank → generate spans with timings and
  candidate counts.
- Metrics: query latency, retrieval recall proxy, cache hit rate, sync lag
  (time since last successful delta), index freshness, error rates.
- Sync health dashboard: last delta token time per drive/project, deletions
  applied, items reindexed.
- Backups for metadata DB and vector index; the raw blob store is the source of
  truth for re-indexing.

---

## 11. Security checklist (must pass before production)

- [ ] Every chunk (wiki **and** SharePoint) carries a `project`; queries filter on the selected project.
- [ ] Selected `project` re-validated against the caller PAT's accessible set on every `/chat` (else `403`).
- [ ] SharePoint site→project mapping applied at ingest; **unmapped content fails closed** (skipped by default).
- [ ] **Ingestion PAT never accepted from a client** and never returned in any response.
- [ ] User PAT never in `localStorage`; in memory or `HttpOnly` cookie; only a `sha256` fingerprint cached server-side.
- [ ] Explicit CORS allow-list; credentials only with concrete origins.
- [ ] Auth + rate limits on every endpoint; request size caps.
- [ ] Secrets (Ingestion PAT, app-registration secret) in a vault, not `.env` in the repo.
- [ ] Vector store and blobs **git-ignored**; encrypted at rest.
- [ ] Audit log of who asked what, in which project, and which sources were returned.
- [ ] Project-RBAC eval (both sources) is a blocking CI gate.

---

## 12. Delivery phases

1. **Foundations** — User-PAT auth + accessible-project resolution + project
   picker, Ingestion PAT with **auto-discover-all-projects**, BFF skeleton,
   metadata DB, vector DB (Qdrant), Azure Wiki connector with full + delta sync,
   per-project RBAC, `gemma4:e4b` generation (non-streaming). *Exit: a user gets a
   cited answer scoped to a selected project's wiki, and cannot query a project
   their PAT lacks.*
2. **SharePoint + freshness** — SharePoint app-only connector over all sites,
   **site→project mapping** so SharePoint obeys the same project RBAC, delta +
   delete handling, OCR, parent-doc expansion. *Exit: SharePoint answers appear
   only for the user's authorized project; edits/deletes reflected within minutes.*
3. **Quality** — hybrid BM25+dense fusion, cross-encoder reranker, query rewrite,
   multi-turn, streaming + inline citations, eval harness in CI.
4. **Scale & ops** — background ingestion workers, shared cache, observability,
   load testing, admin sync dashboard.

Each phase is independently shippable; Phase 1 already delivers the anchor
guarantee — **grounded answers under per-project RBAC** — and Phase 2 extends that
same RBAC to SharePoint via the site→project mapping.
