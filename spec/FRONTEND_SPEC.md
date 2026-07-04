# Build Spec — Frontend (Knowledge Assistant Web UI)

Clean-slate spec for the **new** frontend of the RAG chatbot. Pairs with
`REBUILD_SPEC.md` (backend). Binding React conventions live in `react.md` — this
spec **must not** contradict it; where they overlap, `react.md` wins.

Current frontend (`frontend/`, Vite + plain React + TanStack Query + fetch-in-
components) is documented only as the *starting point*; this spec replaces it.

---

## 1. Goals

1. **PAT gate → project pick → chat.** User pastes Azure DevOps **PAT**, sees the
   projects that PAT grants, selects **one**, chats scoped to it.
2. **Streamed answers with inline citations.** Render SSE token stream live; show
   `[n]` markers linking to source pages/files.
3. **Per-project isolation.** History, selected project, and scope are per-project;
   switching projects never bleeds context.
4. **Follows `react.md` gates.** RR7 SPA, shadcn/ui + Tailwind, framework data
   layer (no `fetch`/`useEffect` for data), kebab-case files, componentize hard.

### Non-goals (v1)
- No SSO login UI (PAT only — matches backend §3).
- No admin/ingestion UI (separate concern).
- No write-back to sources.

---

## 2. Stack (aligns with `react.md`)

| Concern | Choice | Note |
|---|---|---|
| Router/data | **React Router 7 (SPA mode)** | `clientLoader`/`clientAction`, `<Link>`/`<Form>`, `useSearchParams`. **No `useEffect` data fetch.** |
| Styling | **shadcn/ui + Tailwind** | No MUI/Chakra; no CSS modules/styled-components. |
| Forms | **react-hook-form + TypeBox resolver** | PAT form + composer. **Never Zod.** Schemas shared from `@acsm/shared-types`. |
| Server state | RR7 loaders/actions | No `fetch` inside components. |
| Local state | `useState`/`useReducer`; store (zustand) only at real need | Lift state up, push subscriptions down. |
| Streaming | **SSE** via `EventSource`/fetch stream, wrapped in a subscription hook | Consume at leaf, not route module. |
| Language | TypeScript strict | Props via `interface`/`type`, no `any`. |

Build tool inherited from repo (RR7 SPA). Filenames kebab-case; components
PascalCase; named exports (default only for RR7 route modules).

---

## 3. Routes (RR7)

```
/                      redirect → /login or /p/:project
/login                 PAT gate (public)
/p/:project            chat for selected project
/p/:project/c/:conv    a specific conversation
```

- **`/login`** — `clientAction` submits PAT, calls `GET /v1/projects`. Success:
  store PAT (§6), redirect to first project. Failure: form error.
- **`/p/:project`** — `clientLoader` verifies `:project` is in the caller's
  accessible set (from `/v1/projects`); not accessible → redirect + toast. Loads
  conversation list.
- Project switch = navigation (`<Link>` / `useNavigate`), not state mutation.
  URL is the source of truth for selected project (`useSearchParams` / route param).

**No `useEffect(() => fetch())`.** Projects, conversations, history all load via
`clientLoader`. Chat send is a `clientAction` or an explicit event-handler call
into the streaming hook (user action, not effect).

---

## 4. Component tree (componentize aggressively)

```
app-shell                       layout, theme, toaster
  pat-gate                      /login form (react-hook-form + TypeBox)
  project-sidebar               accessible projects, active highlight, sign-out
    project-list-item
  chat-view                     /p/:project
    chat-header                 project name, conversation switch
    message-list                scroll container
      message-bubble            one message
        markdown-content        react-markdown + remark-gfm
        citation-list           [n] → source links (wiki/sharepoint/file)
        answer-meta             used_context / confidence / "not found" note
      typing-indicator
    chat-composer               textarea + send (form)
```

Rules: one component per file; extract when a block does a distinct job; parent
reads as list of named children. Component pushing ~80 lines → split.

---

## 5. Data contracts (from backend `REBUILD_SPEC.md`)

TypeBox schemas shared with server (`@acsm/shared-types`) validate both request
and any parsed response shape.

```
GET /v1/projects            → { projects: string[] }          # header: User PAT
POST /v1/chat  (SSE stream) → header: User PAT; body:
  { conversation_id: string|null, project: string, message: string,
    filters?: { source_type?: ("azure_wiki"|"sharepoint")[] } }

SSE events:
  sources → [{ n, title, uri, source_type, breadcrumb }]
  token   → "partial text"        (repeated, append in order)
  done    → { confidence, used_context, message_id }
```

- `project` **required** on every chat send; must be the selected route project.
- `403` on chat/projects = PAT invalid or lost project access → drop PAT, bounce
  to `/login`, clear that project's history.

---

## 6. Auth / PAT handling (security)

Backend has no SSO; PAT is the credential. Handle with care:

- Keep PAT **in memory** (module singleton / store), sent as request header.
- If persistence across reload is required, prefer server-set `HttpOnly` `Secure`
  cookie. If client-side is unavoidable, `sessionStorage` (tab-scoped) over
  `localStorage` — **never `localStorage`**. (Current build uses `sessionStorage`;
  in-memory is the target.)
- Never log the PAT. Mask in the input (`type=password`, show/hide toggle).
- On any `401/403`, clear PAT + per-project history, redirect to `/login`.

---

## 7. Streaming render

- Subscription hook `use-chat-stream` opens the SSE connection, exposes
  `{ status, tokens, sources, done }`. Lives at the **leaf** (`message-bubble` /
  `message-list`), so only it re-renders per token — parents stay still
  (`react.md`: push subscriptions down).
- Append `token` events in order into the pending assistant bubble.
- `sources` arrive first → render `citation-list` skeleton; `[n]` in text links to
  the matching source.
- **Cleanup mandatory**: close the stream on unmount / project switch / new send.
  Effect (if used for the subscription) carries a one-line justification comment
  and is flagged for review per `react.md` §useEffect.

---

## 8. State ownership

- **Selected project**: URL (route param) — single source of truth.
- **PAT**: in-memory store (app-wide).
- **Per-project chat history**: keyed by project id; store or loader cache. Switch
  project = different key, no bleed.
- **Composer input / pending send**: local `useState` in `chat-composer` /
  `chat-view`.
- Lift shared state to lowest common ancestor; 4+ prop levels → store.

---

## 9. Errors & empty states

| State | UI |
|---|---|
| No PAT | `/login` gate. |
| PAT valid, 0 projects | "No accessible projects for this PAT." |
| Project selected, no messages | Empty prompt: "Ask about `<project>`". |
| Backend `not found in knowledge base` | Render answer as-is; `answer-meta` shows no-context note. |
| `403` mid-session | Toast + drop PAT + `/login`. |
| Stream error | Mark bubble errored, offer retry. |

---

## 10. Alignment checklist (gates from `react.md`)

- [ ] RR7 primitives for nav + data; **zero `useEffect` data fetches**.
- [ ] Any `useEffect` (e.g. stream subscription, autoscroll) has justification comment + review flag + cleanup.
- [ ] shadcn/ui + Tailwind only; no other UI kit, no CSS modules.
- [ ] Forms use react-hook-form + TypeBox resolver; **no Zod**.
- [ ] Kebab-case filenames; one component per file; PascalCase components; named exports.
- [ ] TypeScript strict; props typed; no stray `any`.
- [ ] Lists keyed by stable id, never index.
- [ ] No speculative memo; profile first.
- [ ] Subscriptions at leaves; state lifted to lowest common ancestor.

---

## 11. Delivery phases

1. **Gate + projects** — `/login` PAT form, `GET /v1/projects`, project sidebar,
   route guard. *Exit: user authenticates and picks a project.*
2. **Chat (non-stream)** — composer, message list, markdown, per-project history.
   *Exit: scoped Q&A works end-to-end.*
3. **Streaming + citations** — SSE hook, live tokens, `citation-list`, confidence.
4. **Polish** — shadcn theming, empty/error states, a11y, autoscroll, retry.
