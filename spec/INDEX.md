# Spec Index — Knowledge Assistant (Azure Wiki + SharePoint RAG Chatbot)

Single entry point for all specs. Read a doc by its path below.

Project = document-grounded RAG chatbot. Answers **only** from Azure DevOps wikis
+ SharePoint docs. Per-project RBAC via the caller's Azure DevOps PAT: user picks
one accessible project, bot answers only from that project (wiki **and**
SharePoint). LLM `gemma4:e4b` (Ollama, thinking off). Dedicated ingestion PAT
crawls all projects; users never see it.

## Documents

| File | What it is | Read when |
|---|---|---|
| [REBUILD_SPEC.md](./REBUILD_SPEC.md) | **Backend build spec (target).** Clean-slate design: PAT auth, per-project RBAC over both sources, site→project mapping, delta sync, hybrid retrieval + reranker, streaming, `gemma4:e4b`. | Building/changing backend. **Primary spec.** |
| [FRONTEND_SPEC.md](./FRONTEND_SPEC.md) | **Frontend build spec (target).** RR7 SPA, shadcn/Tailwind, react-hook-form+TypeBox, PAT gate → project picker → streamed chat with citations. | Building/changing web UI. |
| [react.md](./react.md) | **React coding rules (gates).** Binding conventions — RR7 primitives, no-`useEffect`-for-data, shadcn, no Zod, kebab-case, componentize. FRONTEND_SPEC must obey this. | Writing any `.ts`/`.tsx`. Do **not** edit. |

## Read order

1. `INDEX.md` (this) — orientation.
2. `REBUILD_SPEC.md` — target backend.
3. `FRONTEND_SPEC.md` + `react.md` — target frontend + rules.

## Constraints (apply everywhere)

- **No SSO, no Azure AI Search.** PAT-based project RBAC; self-hosted vector DB (Qdrant/pgvector/Chroma).
- **Two PATs:** user PAT (query, in browser) vs. ingestion PAT (server secret, ingest-only).
- **LLM:** `gemma4:e4b` via Ollama, `temperature=0`, thinking disabled.
- **Fail closed:** unmapped SharePoint content skipped by default.
