from __future__ import annotations

import logging
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pypdf import PdfReader

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown", ".pdf", ".html", ".htm"}


def load_documents(data_dir: str, urls_file: str | None = None) -> list[Document]:
    documents = load_local_documents(data_dir)
    if urls_file:
        documents.extend(load_url_documents(urls_file))
    return documents


def load_local_documents(data_dir: str) -> list[Document]:
    base = Path(data_dir)
    if not base.exists():
        logger.warning("Data directory does not exist: %s", base)
        return []

    documents: list[Document] = []
    for path in base.rglob("*"):
        if not path.is_file():
            continue

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        try:
            documents.extend(_load_single_file(path))
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("Failed to load file %s: %s", path, exc)
    return documents


def load_url_documents(urls_file: str) -> list[Document]:
    path = Path(urls_file)
    if not path.exists():
        return []

    urls = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    documents: list[Document] = []
    for url in urls:
        try:
            response = requests.get(url, timeout=20)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            for tag in soup(["script", "style", "noscript"]):
                tag.extract()
            text = _clean_text(soup.get_text("\n"))
            if text:
                documents.append(Document(page_content=text, metadata={"source": url, "type": "url"}))
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("Failed to load url %s: %s", url, exc)
    return documents


def split_documents(documents: list[Document], chunk_size: int, chunk_overlap: int) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    for idx, doc in enumerate(chunks):
        doc.metadata["chunk_id"] = idx
    return chunks


def _load_single_file(path: Path) -> list[Document]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _load_pdf(path)
    if suffix in {".html", ".htm"}:
        return _load_html_file(path)
    return _load_text_file(path)


def _load_pdf(path: Path) -> list[Document]:
    documents: list[Document] = []
    reader = PdfReader(str(path))
    for page_idx, page in enumerate(reader.pages, start=1):
        text = _clean_text(page.extract_text() or "")
        if not text:
            continue
        documents.append(
            Document(
                page_content=text,
                metadata={"source": str(path), "page": page_idx, "type": "pdf"},
            )
        )
    return documents


def _load_html_file(path: Path) -> list[Document]:
    html = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.extract()
    text = _clean_text(soup.get_text("\n"))
    if not text:
        return []
    return [Document(page_content=text, metadata={"source": str(path), "type": "html"})]


def _load_text_file(path: Path) -> list[Document]:
    text = _clean_text(path.read_text(encoding="utf-8", errors="ignore"))
    if not text:
        return []
    return [Document(page_content=text, metadata={"source": str(path), "type": "text"})]


def _clean_text(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)

