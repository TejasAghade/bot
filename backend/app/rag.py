from __future__ import annotations

from dataclasses import dataclass
from collections import OrderedDict
import re

from langchain_ollama import ChatOllama

from app.config import Settings
from app.vectorstore import get_vectorstore

UNKNOWN_ANSWER = "I don't know based on the provided documents."


@dataclass
class RAGResult:
    answer: str
    sources: list[dict]
    used_context: bool


def _origin_label(doc_type) -> str:
    """Map a chunk's ``type`` metadata to a human-facing origin label."""
    t = str(doc_type or "").lower()
    if t == "sharepoint":
        return "sharepoint"
    if t == "azure_devops_wiki":
        return "wiki"
    if t == "url":
        return "web"
    return "file"


class RAGService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.vectorstore = get_vectorstore(settings)
        self.llm = ChatOllama(
            model=settings.llm_model,
            base_url=settings.ollama_base_url,
            temperature=0,
            num_predict=settings.llm_num_predict,
            keep_alive=settings.ollama_keep_alive,
            reasoning=settings.llm_reasoning,
        )
        self._cache: OrderedDict[str, RAGResult] = OrderedDict()
        self._cache_revision = 0

    def reload_vectorstore(self) -> None:
        self.vectorstore = get_vectorstore(self.settings)
        self._cache_revision += 1
        self._cache.clear()

    def indexed_document_count(self) -> int:
        return self.vectorstore._collection.count()

    def answer(
        self,
        question: str,
        project: str | None = None,
        allowed_projects: list[str] | None = None,
    ) -> RAGResult:
        project_filter = (project or "").strip() or None
        scope_projects: list[str] | None = None
        if not project_filter and allowed_projects is not None:
            scope_projects = sorted({p.strip() for p in allowed_projects if p and p.strip()})

        where, scope_token = self._build_scope_filter(project_filter, scope_projects)

        cache_key = self._cache_key(question, scope_token)
        cached = self._cache.get(cache_key)
        if cached is not None:
            self._cache.move_to_end(cache_key)
            return cached

        filtered = self._retrieve(question, where)
        if not filtered:
            result = RAGResult(answer=UNKNOWN_ANSWER, sources=[], used_context=False)
            self._remember(cache_key, result)
            return result

        sources = self._source_infos(filtered)

        if self.settings.enable_fast_path:
            fast_answer = self._fast_answer(question, filtered)
            if fast_answer:
                result = RAGResult(answer=fast_answer, sources=sources, used_context=True)
                self._remember(cache_key, result)
                return result

        context = self._format_context(filtered)
        prompt = self._build_prompt(question, context)
        raw_response = self.llm.invoke(prompt)
        answer = _sanitize_answer(raw_response.content or "")
        if not answer:
            answer = UNKNOWN_ANSWER

        result = RAGResult(answer=answer, sources=sources, used_context=True)
        self._remember(cache_key, result)
        return result

    def _retrieve(self, question: str, where: dict | None):
        """Hybrid retrieval: vector search, then lexical re-rank + dedup by source.

        Pure vector similarity from a small embedding model buries docs whose title or
        wording exactly matches the query. We fetch a wider candidate pool and re-rank
        each candidate by ``semantic + w_content*content_overlap + w_name*name_overlap``,
        keep the best chunk per source, then apply the relevance gate.
        """
        k = max(self.settings.top_k, self.settings.rerank_candidates)
        search_kwargs: dict[str, object] = {"k": k}
        if where is not None:
            search_kwargs["filter"] = where
        raw_matches = self.vectorstore.similarity_search_with_score(question, **search_kwargs)

        # Re-rank with a lexical boost and keep only the strongest chunk per source.
        best: dict[str, tuple[float, object, float]] = {}
        for doc, distance in raw_matches:
            semantic = _distance_to_similarity(distance)
            name = str(doc.metadata.get("file_name") or doc.metadata.get("source") or "")
            name_norm = re.sub(r"[^a-zA-Z0-9]+", " ", name)
            content_overlap = _term_overlap_ratio(question, doc.page_content)
            name_overlap = _term_overlap_ratio(question, name_norm)
            combined = (
                semantic
                + self.settings.content_match_weight * content_overlap
                + self.settings.name_match_weight * name_overlap
            )
            key = str(doc.metadata.get("source") or id(doc))
            if key not in best or combined > best[key][0]:
                best[key] = (combined, doc, semantic)

        ranked = sorted(best.values(), key=lambda item: item[0], reverse=True)

        # Relevance gate on the semantic score, but keep re-ranked order (so a strong
        # title/keyword match outranks a marginally-higher-similarity but off-topic doc).
        filtered = [(doc, semantic) for combined, doc, semantic in ranked if semantic >= self.settings.min_relevance]

        # Near-miss fallback: allow the top re-ranked doc if it overlaps the query well.
        if not filtered and ranked:
            combined, doc, semantic = ranked[0]
            if _term_overlap_ratio(question, doc.page_content) >= 0.5:
                filtered = [(doc, semantic)]

        return filtered[: self.settings.max_context_docs]

    def _format_context(self, filtered_matches) -> str:
        parts: list[str] = []
        total_chars = 0
        for idx, (doc, score) in enumerate(filtered_matches, start=1):
            source = str(doc.metadata.get("source", "unknown"))
            page = doc.metadata.get("page")
            page_text = f", page {page}" if page else ""
            remaining = self.settings.max_context_chars - total_chars
            if remaining <= 0:
                break
            content = doc.page_content[:remaining]
            snippet = f"[{idx}] source={source}{page_text}, similarity={score:.3f}\n{content}"
            parts.append(snippet)
            total_chars += len(snippet) + 2
        return "\n\n".join(parts)

    def _build_prompt(self, question: str, context: str) -> str:
        return f"""You are the user's internal knowledge assistant.
Answer like a helpful teammate in chat, but use only the provided context.
If the answer is not explicitly available in the context, reply exactly:
"{UNKNOWN_ANSWER}"
Do not use outside knowledge. Do not guess.
Give a moderately detailed answer.
Prefer 1 to 3 short paragraphs, or a short ordered list when the docs describe steps.
Include the important details needed to act on the answer, not just a one-line summary.
Return only the final answer text.
Do not include source metadata, index tags, similarity scores, or context labels.

Context:
{context}

Question:
{question}
"""

    def _source_infos(self, filtered_matches) -> list[dict]:
        """Structured citations so the API can show where each answer came from."""
        output: list[dict] = []
        seen: set[tuple] = set()
        for doc, _score in filtered_matches:
            meta = doc.metadata
            source = str(meta.get("source", "") or "")
            origin = _origin_label(meta.get("type"))
            title = str(meta.get("file_name") or meta.get("path") or source or "unknown")
            page = meta.get("page")
            if page:
                title = f"{title} (page {page})"
            key = (origin, title, source)
            if key in seen:
                continue
            seen.add(key)
            info: dict = {"origin": origin, "title": title, "url": source or None}
            for field in ("site", "folder", "project"):
                value = meta.get(field)
                if value:
                    info[field] = str(value)
            output.append(info)
        return output

    def _fast_answer(self, question: str, filtered_matches) -> str | None:
        top_doc, top_score = filtered_matches[0]
        if top_score < self.settings.fast_path_min_relevance:
            return None

        ranked_passages: list[tuple[float, str]] = []
        for doc, doc_score in filtered_matches[: self.settings.fast_path_max_docs]:
            for passage in _candidate_passages(doc.page_content):
                overlap = _term_overlap_ratio(question, passage)
                score = overlap + (doc_score * 0.15)
                ranked_passages.append((score, passage))

        ranked_passages.sort(key=lambda item: item[0], reverse=True)
        chosen = _select_diverse_passages(
            ranked_passages,
            max_items=self.settings.max_answer_sentences,
            min_score=self.settings.fast_path_min_overlap,
        )
        if not chosen:
            return None
        return _format_extractive_answer(chosen)

    def _build_scope_filter(
        self,
        project_filter: str | None,
        scope_projects: list[str] | None,
    ) -> tuple[dict | None, str]:
        """Build the metadata filter enforcing per-source authorization.

        Azure DevOps wiki chunks are project-scoped: they are only searchable for
        projects the caller can access. Every other source (SharePoint, uploaded
        files, URLs) is not project-gated and is always searchable. Because every
        chunk carries a ``type`` (only AzDO wiki chunks use ``azure_devops_wiki``),
        ``type != azure_devops_wiki`` reliably selects all non-AzDO content.
        """
        non_azdo = {"type": {"$ne": "azure_devops_wiki"}}
        if project_filter:
            azdo_clause: dict = {"project": project_filter}
            token = f"p={project_filter.lower()}"
        elif scope_projects:
            if len(scope_projects) == 1:
                azdo_clause = {"project": scope_projects[0]}
            else:
                azdo_clause = {"project": {"$in": scope_projects}}
            token = "s=" + ",".join(p.lower() for p in scope_projects)
        else:
            # No accessible Azure DevOps projects (e.g. no/invalid PAT): non-AzDO only.
            return non_azdo, "nonazdo"
        return {"$or": [non_azdo, azdo_clause]}, token

    def _cache_key(self, question: str, scope_token: str) -> str:
        normalized = " ".join(question.lower().split())
        return f"{self._cache_revision}:{scope_token}:{normalized}"

    def _remember(self, cache_key: str, result: RAGResult) -> None:
        self._cache[cache_key] = result
        self._cache.move_to_end(cache_key)
        while len(self._cache) > self.settings.answer_cache_size:
            self._cache.popitem(last=False)


def _distance_to_similarity(distance: float) -> float:
    safe_distance = max(float(distance), 0.0)
    return 1.0 / (1.0 + safe_distance)


def _term_overlap_ratio(question: str, content: str) -> float:
    terms = {term for term in re.findall(r"[a-zA-Z0-9]+", question.lower()) if len(term) >= 3}
    if not terms:
        return 0.0
    haystack = content.lower()
    hits = sum(1 for term in terms if term in haystack)
    return hits / len(terms)


def _sanitize_answer(text: str) -> str:
    cleaned = (text or "").strip()
    cleaned = re.sub(r"(?mi)^\[\d+\]\s*source=.*(?:\r?\n)?", "", cleaned)
    cleaned = re.sub(r"(?mi)^source=.*(?:\r?\n)?", "", cleaned)
    cleaned = re.sub(r"(?mi)^similarity=.*(?:\r?\n)?", "", cleaned)
    return cleaned.strip()


def _candidate_passages(text: str) -> list[str]:
    raw_sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
    sentences: list[str] = []
    seen: set[str] = set()
    for sentence in raw_sentences:
        cleaned = " ".join(sentence.split()).strip()
        if len(cleaned) < 20:
            continue
        if len(cleaned) > 260:
            cleaned = cleaned[:257].rstrip() + "..."
        normalized = cleaned.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        sentences.append(cleaned)
    return sentences


def _select_diverse_passages(
    ranked_passages: list[tuple[float, str]],
    max_items: int,
    min_score: float,
) -> list[str]:
    chosen: list[str] = []
    normalized_seen: set[str] = set()
    for score, passage in ranked_passages:
        if score < min_score:
            continue
        normalized = re.sub(r"[^a-z0-9]+", " ", passage.lower()).strip()
        if normalized in normalized_seen:
            continue
        if any(_term_overlap_ratio(existing, passage) >= 0.8 for existing in chosen):
            continue
        normalized_seen.add(normalized)
        chosen.append(passage)
        if len(chosen) >= max_items:
            break
    return chosen


def _format_extractive_answer(passages: list[str]) -> str:
    if not passages:
        return ""
    if len(passages) <= 2:
        return " ".join(passages).strip()
    return "\n".join(f"- {passage}" for passage in passages)
