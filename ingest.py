"""
ingest.py  —  Step 1: Clone a GitHub repo and index it into ChromaDB.

Place this file inside the `debugger/` directory:

    debugger/
    ├── ingest.py                  ← this file
    ├── core/
    │   └── ingestion/
    │       ├── __init__.py
    │       ├── code_parser.py
    │       ├── file_walker.py
    │       ├── indexer.py
    │       └── repo_cloner.py
    ├── debugger_chroma/           ← created automatically
    └── debugger_repos/            ← created automatically

Usage:
    # Option A — pass token as env var (recommended)
    GITHUB_TOKEN=ghp_xxx python ingest.py https://github.com/avneeshjarangal-dot/quickcart

    # Option B — pass token as second argument
    python ingest.py https://github.com/avneeshjarangal-dot/quickcart ghp_xxx

    # Force re-clone + re-index even if already done
    python ingest.py https://github.com/avneeshjarangal-dot/quickcart ghp_xxx --force

Output:
    Prints a summary table and writes all chunks to ChromaDB under debugger_chroma/.
    The collection name is printed at the end — use it in Step 2 to query chunks.
"""

import os
import sys
import logging

# ---------------------------------------------------------------------------
# Logging — show INFO by default, DEBUG if INGEST_DEBUG=1
# ---------------------------------------------------------------------------
_level = logging.DEBUG if os.environ.get("INGEST_DEBUG") == "1" else logging.INFO
logging.basicConfig(
    level=_level,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ingest")


# ---------------------------------------------------------------------------
# Parse CLI args
# ---------------------------------------------------------------------------
def parse_args():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(1)

    github_url = args[0]
    force = "--force" in args

    # token: CLI arg takes priority, else env var
    token_args = [a for a in args[1:] if not a.startswith("--")]
    github_token = token_args[0] if token_args else os.environ.get("GITHUB_TOKEN")

    if not github_token:
        logger.error(
            "GitHub token required. Set GITHUB_TOKEN env var or pass it as the second argument."
        )
        sys.exit(1)

    return github_url, github_token, force


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    github_url, github_token, force = parse_args()

    logger.info("=" * 60)
    logger.info("Debugger — Step 1: Repo Ingestion")
    logger.info("=" * 60)
    logger.info(f"Repo  : {github_url}")
    logger.info(f"Force : {force}")
    logger.info("=" * 60)

    # Import here so any import error surfaces clearly
    try:
        from core.ingestion.indexer import index_repo, is_repo_indexed
    except ImportError as e:
        logger.error(
            f"Import failed: {e}\n"
            "Make sure you're running this from inside the debugger/ directory "
            "and that core/ingestion/ exists with all four modules."
        )
        sys.exit(1)

    # Check if already indexed (and not forcing)
    if not force and is_repo_indexed(github_url):
        logger.info("Repo is already indexed. Use --force to re-index.")
        logger.info("Skipping ingestion — collection is ready for querying.")
        _print_summary_stub(github_url)
        return

    # Run the full pipeline
    summary = index_repo(github_url, github_token, force=force)

    # Pretty-print the summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("Ingestion complete")
    logger.info("=" * 60)
    _print_summary(summary)


def _print_summary(summary: dict):
    rows = [
        ("Repo",              summary["repo"]),
        ("Local clone path",  summary["local_path"]),
        ("Files walked",      summary["total_files_walked"]),
        ("Files skipped",     summary["total_files_skipped"]),
        ("Chunks indexed",    summary["total_chunks_indexed"]),
        ("Chunks with callers", summary["chunks_with_callers"]),
        ("ChromaDB dir",      summary["chroma_dir"]),
        ("Collection name",   summary["collection_name"]),
    ]
    col_w = max(len(r[0]) for r in rows) + 2
    for label, value in rows:
        print(f"  {label:<{col_w}} {value}")
    print()
    print("  Next step: run Step 2 with the collection name above to query chunks.")


def _print_summary_stub(github_url: str):
    """Minimal output when we skip because already indexed."""
    try:
        from core.ingestion.indexer import get_collection
        from core.ingestion.repo_cloner import parse_repo_url
        info = parse_repo_url(github_url)
        col  = get_collection(info["full_name"])
        print(f"  Collection : {col.name}")
        print(f"  Chunks     : {col.count()}")
    except Exception:
        pass


if __name__ == "__main__":
    main()