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
    sources: list[str]
    used_context: bool


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
        )
        self._cache: OrderedDict[str, RAGResult] = OrderedDict()
        self._cache_revision = 0

    def reload_vectorstore(self) -> None:
        self.vectorstore = get_vectorstore(self.settings)
        self._cache_revision += 1
        self._cache.clear()

    def indexed_document_count(self) -> int:
        return self.vectorstore._collection.count()

    def answer(self, question: str) -> RAGResult:
        cache_key = self._cache_key(question)
        cached = self._cache.get(cache_key)
        if cached is not None:
            self._cache.move_to_end(cache_key)
            return cached

        raw_matches = self.vectorstore.similarity_search_with_score(question, k=self.settings.top_k)
        matches = [(doc, _distance_to_similarity(distance)) for doc, distance in raw_matches]
        filtered = [(doc, score) for doc, score in matches if score >= self.settings.min_relevance]

        # Safe fallback for near-miss vector scores: only allow if question terms
        # overlap strongly with the top retrieved chunk text.
        if not filtered and matches:
            top_doc, top_score = max(matches, key=lambda item: item[1])
            if _term_overlap_ratio(question, top_doc.page_content) >= 0.5:
                filtered = [(top_doc, top_score)]

        if not filtered:
            result = RAGResult(answer=UNKNOWN_ANSWER, sources=[], used_context=False)
            self._remember(cache_key, result)
            return result

        filtered = filtered[: self.settings.max_context_docs]
        sources = self._sources(filtered)

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

    def _sources(self, filtered_matches) -> list[str]:
        output: list[str] = []
        for doc, _score in filtered_matches:
            source = str(doc.metadata.get("source", "unknown"))
            page = doc.metadata.get("page")
            display = f"{source} (page {page})" if page else source
            if display not in output:
                output.append(display)
        return output

    def _fast_answer(self, question: str, filtered_matches) -> str | None:
        top_doc, top_score = filtered_matches[0]
        if top_score < self.settings.fast_path_min_relevance:
            return None

        sentences = _candidate_sentences(top_doc.page_content)
        ranked = sorted(
            (
                (_term_overlap_ratio(question, sentence), sentence)
                for sentence in sentences
            ),
            key=lambda item: item[0],
            reverse=True,
        )
        chosen = [sentence for score, sentence in ranked if score >= 0.45][: self.settings.max_answer_sentences]
        if not chosen:
            return None
        return " ".join(chosen).strip()

    def _cache_key(self, question: str) -> str:
        normalized = " ".join(question.lower().split())
        return f"{self._cache_revision}:{normalized}"

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


def _candidate_sentences(text: str) -> list[str]:
    raw_sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
    sentences: list[str] = []
    seen: set[str] = set()
    for sentence in raw_sentences:
        cleaned = " ".join(sentence.split()).strip()
        if len(cleaned) < 20:
            continue
        normalized = cleaned.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        sentences.append(cleaned)
    return sentences
