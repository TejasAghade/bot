from __future__ import annotations

from dataclasses import dataclass
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
        )

    def reload_vectorstore(self) -> None:
        self.vectorstore = get_vectorstore(self.settings)

    def indexed_document_count(self) -> int:
        return self.vectorstore._collection.count()

    def answer(self, question: str) -> RAGResult:
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
            return RAGResult(answer=UNKNOWN_ANSWER, sources=[], used_context=False)

        context = self._format_context(filtered)
        prompt = self._build_prompt(question, context)
        raw_response = self.llm.invoke(prompt)
        answer = _sanitize_answer(raw_response.content or "")
        if not answer:
            answer = UNKNOWN_ANSWER

        sources = self._sources(filtered)
        return RAGResult(answer=answer, sources=sources, used_context=True)

    def _format_context(self, filtered_matches) -> str:
        parts: list[str] = []
        for idx, (doc, score) in enumerate(filtered_matches, start=1):
            source = str(doc.metadata.get("source", "unknown"))
            page = doc.metadata.get("page")
            page_text = f", page {page}" if page else ""
            parts.append(
                f"[{idx}] source={source}{page_text}, similarity={score:.3f}\n{doc.page_content}"
            )
        return "\n\n".join(parts)

    def _build_prompt(self, question: str, context: str) -> str:
        return f"""You are a strict documentation assistant.
You MUST answer using only the provided context.
If the answer is not explicitly available in the context, reply exactly:
"{UNKNOWN_ANSWER}"
Do not use outside knowledge. Do not guess.
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
