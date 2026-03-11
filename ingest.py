from __future__ import annotations

import argparse
import json

from app.ingestion import run_ingestion


def main() -> None:
    parser = argparse.ArgumentParser(description="Index local docs into Chroma vector DB.")
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append new chunks instead of rebuilding the vector store.",
    )
    args = parser.parse_args()

    result = run_ingestion(append=args.append)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

