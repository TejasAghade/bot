"""Diagnostic: show what SharePoint content the configured creds can actually see.

Read-only. Does NOT download file contents (only metadata) and does NOT touch the
vector store, so it's safe to run anytime to verify access before/without ingesting.

Usage:
    python sharepoint_check.py              # token + sites + drives + file counts
    python sharepoint_check.py --sample 10  # also list up to N example file paths
    python sharepoint_check.py --download-test  # try downloading one file to confirm
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import requests

from app.config import get_settings
from app.document_loader import (
    SUPPORTED_EXTENSIONS,
    DocumentLoadError,
    _get_graph_token,
    _graph_get_all,
    _resolve_sharepoint_sites,
)


def _walk(session, base, drive_id, item_id, folder_path, counter, samples, sample_limit):
    children = _graph_get_all(
        session,
        f"{base}/drives/{drive_id}/items/{item_id}/children",
        params={"$select": "id,name,folder,file,size", "$top": 200},
        context=f"Folder read failed for '{folder_path or '/'}'.",
    )
    for child in children:
        if not isinstance(child, dict):
            continue
        name = str(child.get("name") or "")
        child_path = f"{folder_path}/{name}" if folder_path else f"/{name}"
        if isinstance(child.get("folder"), dict):
            _walk(session, base, drive_id, str(child.get("id")), child_path, counter, samples, sample_limit)
        elif isinstance(child.get("file"), dict):
            suffix = Path(name).suffix.lower()
            counter[suffix] += 1
            if suffix in SUPPORTED_EXTENSIONS and len(samples) < sample_limit:
                samples.append((child_path, str(child.get("id") or ""), drive_id))


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify SharePoint access for the configured creds.")
    parser.add_argument("--sample", type=int, default=0, help="List up to N example ingestable file paths.")
    parser.add_argument("--download-test", action="store_true", help="Try downloading one ingestable file.")
    args = parser.parse_args()

    settings = get_settings()
    if not settings.sharepoint_enabled:
        print("SharePoint is NOT configured: set SHAREPOINT_TENANT_ID, "
              "SHAREPOINT_CLIENT_ID, and SHAREPOINT_CLIENT_SECRET in .env.")
        return

    base = settings.sharepoint_graph_base_url.rstrip("/")
    try:
        token = _get_graph_token(
            settings.sharepoint_tenant_id,
            settings.sharepoint_client_id,
            settings.sharepoint_client_secret,
            settings.sharepoint_login_base_url,
        )
    except DocumentLoadError as exc:
        print(f"[AUTH FAILED] {exc}")
        return
    print("[1/3] Auth OK - obtained an app-only token.")

    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {token}", "Accept": "application/json"})

    try:
        sites = _resolve_sharepoint_sites(session, base, settings.sharepoint_site)
    except DocumentLoadError as exc:
        print(f"[SITES FAILED] {exc}")
        return
    scope = f"site '{settings.sharepoint_site}'" if settings.sharepoint_site else "ALL readable sites"
    print(f"[2/3] Resolved {len(sites)} site(s) for scope: {scope}.")
    if not sites:
        print("      No sites visible. The app likely lacks Sites.Read.All (with admin consent).")
        return

    print("[3/3] Walking drives and folders (metadata only)...\n")
    grand = Counter()
    samples: list[tuple[str, str, str]] = []
    sample_limit = args.sample if args.sample > 0 else (1 if args.download_test else 0)
    for site in sites:
        if not isinstance(site, dict):
            continue
        site_name = str(site.get("displayName") or site.get("name") or site.get("id"))
        drives = _graph_get_all(
            session,
            f"{base}/sites/{site.get('id')}/drives",
            params={"$select": "id,name"},
            context=f"Drive listing failed for '{site_name}'.",
        )
        for drive in drives:
            counter: Counter = Counter()
            _walk(session, base, str(drive.get("id")), "root", "", counter, samples, sample_limit)
            supported = sum(v for k, v in counter.items() if k in SUPPORTED_EXTENSIONS)
            total = sum(counter.values())
            grand.update(counter)
            print(f"  [{site_name}] drive '{drive.get('name')}': "
                  f"{supported} ingestable / {total} total files  {dict(counter)}")

    total_supported = sum(v for k, v in grand.items() if k in SUPPORTED_EXTENSIONS)
    print(f"\nTOTAL ingestable files across all sites/drives: {total_supported}")
    print(f"Supported extensions: {sorted(SUPPORTED_EXTENSIONS)}")

    if samples and (args.sample > 0):
        print("\nExample ingestable files:")
        for path, _id, _drive in samples:
            print(f"  - {path}")

    if args.download_test:
        if not samples:
            print("\n[download-test] No ingestable file found to test.")
            return
        path, item_id, drive_id = samples[0]
        resp = session.get(f"{base}/drives/{drive_id}/items/{item_id}/content", timeout=60)
        if resp.status_code < 400:
            print(f"\n[download-test] OK - downloaded {len(resp.content)} bytes from '{path}'.")
        else:
            print(f"\n[download-test] FAILED for '{path}': HTTP {resp.status_code}. "
                  "Token can list metadata but not read content - check Files.Read.All.")


if __name__ == "__main__":
    main()
