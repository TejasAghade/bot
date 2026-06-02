from __future__ import annotations

import base64
import logging
from pathlib import Path
import re
from urllib.parse import quote, urlparse

import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pypdf import PdfReader

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown", ".pdf", ".html", ".htm"}


class DocumentLoadError(RuntimeError):
    pass


def load_documents(
    data_dir: str,
    urls_file: str | None = None,
    azure_devops_pat: str | None = None,
    azure_devops_org: str | None = None,
    azure_devops_project: str | None = None,
    azure_devops_projects: list[str] | None = None,
    azure_devops_wiki: str | None = None,
    azure_devops_wiki_path: str = "/",
    azure_devops_api_version: str = "7.1",
) -> list[Document]:
    documents = load_local_documents(data_dir)
    if urls_file:
        documents.extend(load_url_documents(urls_file, azure_devops_pat=azure_devops_pat))

    projects = _resolve_project_list(azure_devops_projects, azure_devops_project)
    if azure_devops_pat and azure_devops_org and projects:
        # When multiple projects are configured, ignore AZURE_DEVOPS_WIKI (a single
        # wiki identifier cannot meaningfully apply across projects) and ingest every
        # wiki in each project.
        restrict_to_single_wiki = bool(azure_devops_wiki) and len(projects) == 1
        for project_name in projects:
            try:
                if restrict_to_single_wiki:
                    documents.extend(
                        load_azure_devops_wiki_documents(
                            organization=azure_devops_org,
                            project=project_name,
                            wiki_identifier=azure_devops_wiki,
                            wiki_path=azure_devops_wiki_path,
                            azure_devops_pat=azure_devops_pat,
                            api_version=azure_devops_api_version,
                        )
                    )
                else:
                    documents.extend(
                        load_all_azure_devops_project_wikis(
                            organization=azure_devops_org,
                            project=project_name,
                            wiki_path=azure_devops_wiki_path,
                            azure_devops_pat=azure_devops_pat,
                            api_version=azure_devops_api_version,
                        )
                    )
            except DocumentLoadError as exc:
                logger.warning("Skipping Azure DevOps project '%s': %s", project_name, exc)
    return documents


def _resolve_project_list(
    projects: list[str] | None,
    fallback_project: str | None,
) -> list[str]:
    resolved: list[str] = []
    seen: set[str] = set()
    for name in projects or []:
        cleaned = (name or "").strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            resolved.append(cleaned)
    if not resolved and fallback_project:
        cleaned = fallback_project.strip()
        if cleaned:
            resolved.append(cleaned)
    return resolved


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


def load_url_documents(urls_file: str, azure_devops_pat: str | None = None) -> list[Document]:
    path = Path(urls_file)
    if not path.exists():
        return []

    urls = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    documents: list[Document] = []
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/133.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.8",
        }
    )

    for url in urls:
        try:
            fetch_url = _normalize_url_for_fetch(url)
            response = session.get(
                fetch_url,
                timeout=20,
                headers=_auth_headers_for_url(fetch_url, azure_devops_pat),
            )
            response.raise_for_status()
            text = _extract_response_text(response)
            if text:
                documents.append(
                    Document(
                        page_content=text,
                        metadata={"source": url, "fetch_url": fetch_url, "type": "url"},
                    )
                )
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("Failed to load url %s: %s", url, exc)
    return documents


def load_azure_devops_wiki_documents(
    organization: str,
    project: str,
    wiki_identifier: str,
    wiki_path: str,
    azure_devops_pat: str,
    api_version: str = "7.1",
) -> list[Document]:
    session = requests.Session()
    session.headers.update(
        {
            "Accept": "application/json",
            "User-Agent": "drs-chatbot-ingestion/1.0",
            **_basic_auth_headers(azure_devops_pat),
        }
    )
    endpoint = _azure_devops_wiki_endpoint(organization, project, wiki_identifier)
    response = session.get(
        endpoint,
        params={
            "path": _normalize_wiki_path(wiki_path),
            "recursionLevel": "full",
            "includeContent": "true",
            "api-version": api_version,
        },
        timeout=30,
    )
    _raise_for_azure_response(
        response,
        context=(
            f"Azure DevOps wiki read failed for project '{project}' and wiki '{wiki_identifier}'. "
            "Check that AZURE_DEVOPS_PAT is valid, has wiki read access, and can access this project."
        ),
    )
    payload = response.json()

    documents: list[Document] = []
    _collect_azure_wiki_pages(
        node=payload,
        documents=documents,
        session=session,
        endpoint=endpoint,
        api_version=api_version,
        project=project,
        wiki_identifier=wiki_identifier,
    )
    return documents


def load_all_azure_devops_project_wikis(
    organization: str,
    project: str,
    wiki_path: str,
    azure_devops_pat: str,
    api_version: str = "7.1",
) -> list[Document]:
    session = requests.Session()
    session.headers.update(
        {
            "Accept": "application/json",
            "User-Agent": "drs-chatbot-ingestion/1.0",
            **_basic_auth_headers(azure_devops_pat),
        }
    )
    response = session.get(
        _azure_devops_wikis_endpoint(organization, project),
        params={"api-version": api_version},
        timeout=30,
    )
    _raise_for_azure_response(
        response,
        context=(
            f"Azure DevOps wiki listing failed for project '{project}'. "
            "Check that AZURE_DEVOPS_PAT is valid, has the 'vso.wiki' scope, and can access this project. "
            "If you changed .env after starting the API, restart uvicorn so the new PAT is loaded."
        ),
    )
    payload = response.json()

    if isinstance(payload, list):
        wikis = payload
    else:
        wikis = payload.get("value") or []

    documents: list[Document] = []
    seen_identifiers: set[str] = set()
    for wiki in wikis:
        if not isinstance(wiki, dict):
            continue
        wiki_identifier = str(wiki.get("id") or wiki.get("name") or "").strip()
        if not wiki_identifier or wiki_identifier in seen_identifiers:
            continue
        seen_identifiers.add(wiki_identifier)
        try:
            documents.extend(
                load_azure_devops_wiki_documents(
                    organization=organization,
                    project=project,
                    wiki_identifier=wiki_identifier,
                    wiki_path=wiki_path,
                    azure_devops_pat=azure_devops_pat,
                    api_version=api_version,
                )
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("Failed to load Azure DevOps wiki %s: %s", wiki_identifier, exc)
    return documents


def list_user_accessible_projects(
    organization: str,
    azure_devops_pat: str,
    api_version: str = "7.1",
) -> list[str]:
    org = quote(organization, safe="")
    response = requests.get(
        f"https://dev.azure.com/{org}/_apis/projects",
        params={"api-version": api_version, "stateFilter": "wellFormed"},
        headers={
            "Accept": "application/json",
            "User-Agent": "drs-chatbot-projects/1.0",
            **_basic_auth_headers(azure_devops_pat),
        },
        timeout=30,
    )
    _raise_for_azure_response(
        response,
        context=(
            f"Azure DevOps project listing failed for organization '{organization}'. "
            "Check that the supplied PAT is valid and has project-read access."
        ),
    )
    payload = response.json()
    items = payload if isinstance(payload, list) else (payload.get("value") or [])
    names: list[str] = []
    seen: set[str] = set()
    for entry in items:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name") or "").strip()
        if name and name.lower() not in seen:
            seen.add(name.lower())
            names.append(name)
    return names


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


def _extract_response_text(response: requests.Response) -> str:
    content_type = (response.headers.get("Content-Type") or "").lower()
    if "text/html" in content_type or "<html" in response.text.lower():
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.extract()
        return _clean_text(soup.get_text("\n"))
    return _clean_text(response.text)


def _normalize_url_for_fetch(url: str) -> str:
    # Public Google Docs sharing links are best fetched via export endpoint.
    match = re.search(r"https?://docs\.google\.com/document/d/([a-zA-Z0-9_-]+)", url)
    if match:
        doc_id = match.group(1)
        return f"https://docs.google.com/document/d/{doc_id}/export?format=txt"
    return url


def _auth_headers_for_url(url: str, azure_devops_pat: str | None) -> dict[str, str]:
    if not azure_devops_pat:
        return {}

    host = urlparse(url).netloc.lower()
    if not _is_azure_devops_host(host):
        return {}

    return _basic_auth_headers(azure_devops_pat)


def _is_azure_devops_host(host: str) -> bool:
    return "dev.azure.com" in host or host.endswith(".visualstudio.com")


def _basic_auth_headers(personal_access_token: str) -> dict[str, str]:
    token = base64.b64encode(f":{personal_access_token}".encode("utf-8")).decode("ascii")
    return {"Authorization": f"Basic {token}"}


def _azure_devops_wiki_endpoint(organization: str, project: str, wiki_identifier: str) -> str:
    org = quote(organization, safe="")
    proj = quote(project, safe="")
    wiki = quote(wiki_identifier, safe="")
    return f"https://dev.azure.com/{org}/{proj}/_apis/wiki/wikis/{wiki}/pages"


def _azure_devops_wikis_endpoint(organization: str, project: str) -> str:
    org = quote(organization, safe="")
    proj = quote(project, safe="")
    return f"https://dev.azure.com/{org}/{proj}/_apis/wiki/wikis"


def _normalize_wiki_path(path: str | None) -> str:
    raw = (path or "/").strip()
    if not raw:
        return "/"
    return raw if raw.startswith("/") else f"/{raw}"


def _collect_azure_wiki_pages(
    node: dict,
    documents: list[Document],
    session: requests.Session,
    endpoint: str,
    api_version: str,
    project: str | None = None,
    wiki_identifier: str | None = None,
) -> None:
    content = _clean_text(str(node.get("content") or ""))
    page_path = str(node.get("path") or "")
    remote_url = str(node.get("remoteUrl") or "")
    page_id = node.get("id")

    if not content and page_path:
        content = _fetch_azure_wiki_page_content(
            session=session,
            endpoint=endpoint,
            page_path=page_path,
            api_version=api_version,
        )

    if content:
        documents.append(
            Document(
                page_content=content,
                metadata={
                    "source": remote_url or f"azure-wiki:{project or ''}:{page_path}",
                    "path": page_path,
                    "page_id": page_id,
                    "project": project,
                    "wiki": wiki_identifier,
                    "type": "azure_devops_wiki",
                },
            )
        )

    for child in node.get("subPages") or []:
        if isinstance(child, dict):
            _collect_azure_wiki_pages(
                node=child,
                documents=documents,
                session=session,
                endpoint=endpoint,
                api_version=api_version,
                project=project,
                wiki_identifier=wiki_identifier,
            )


def _fetch_azure_wiki_page_content(
    session: requests.Session,
    endpoint: str,
    page_path: str,
    api_version: str,
) -> str:
    response = session.get(
        endpoint,
        params={
            "path": _normalize_wiki_path(page_path),
            "includeContent": "true",
            "api-version": api_version,
        },
        timeout=30,
    )
    _raise_for_azure_response(
        response,
        context=f"Azure DevOps wiki page read failed for path '{page_path}'.",
    )
    payload = response.json()
    return _clean_text(str(payload.get("content") or ""))


def _raise_for_azure_response(response: requests.Response, context: str) -> None:
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        status = response.status_code
        if status in {401, 403}:
            raise DocumentLoadError(context) from exc
        raise DocumentLoadError(f"{context} Azure returned HTTP {status}.") from exc
