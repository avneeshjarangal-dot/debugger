"""
Run this to test the full ingestion pipeline against QuickCart.

Usage:
    python test_ingestion.py
"""

import os
import logging
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

from core.ingestion.indexer import index_repo, is_repo_indexed
from core.ingestion.repo_cloner import parse_repo_url, get_local_path
from core.ingestion.file_walker import walk_repo, count_files
from core.ingestion.code_parser import parse_file, find_chunk_at_line
from core.ingestion.file_walker import get_file_content

GITHUB_URL = "https://github.com/avneeshjarangal-dot/quickcart"  # change this
GITHUB_TOKEN = os.environ["GITHUB_TOKEN"]


def test_clone_and_walk():
    print("\n--- TEST 1: Clone + Walk ---")
    from core.ingestion.repo_cloner import clone_repo
    local_path = clone_repo(GITHUB_URL, GITHUB_TOKEN)
    print(f"Cloned to: {local_path}")

    counts = count_files(local_path)
    print(f"Files found: {counts}")
    assert sum(counts.values()) > 0, "No files found!"
    print("PASS")


def test_parse_single_file():
    print("\n--- TEST 2: Parse a single file ---")
    repo_info = parse_repo_url(GITHUB_URL)
    local_path = get_local_path(repo_info["full_name"])

    # pick a known file
    target = os.path.join(local_path, "services/order-service/controllers/orderController.py")
    if not os.path.exists(target):
        print("SKIP — file not found, clone first")
        return

    content = get_file_content(target)
    chunks = parse_file(
        file_path="services/order-service/controllers/orderController.py",
        content=content,
        language="python",
    )

    print(f"Chunks extracted: {len(chunks)}")
    for c in chunks:
        print(f"  {c['chunk_type']:8} | {c['name']:30} | lines {c['start_line']}-{c['end_line']}")

    # test finding chunk at line 38 (where the bug is)
    bug_chunk = find_chunk_at_line(chunks, 38)
    if bug_chunk:
        print(f"\nChunk containing line 38: {bug_chunk['name']}()")
    assert len(chunks) > 0, "No chunks parsed!"
    print("PASS")


def test_full_index():
    print("\n--- TEST 3: Full Repo Index ---")
    summary = index_repo(GITHUB_URL, GITHUB_TOKEN, force=False)
    print(f"Repo:           {summary['repo']}")
    print(f"Files walked:   {summary['total_files_walked']}")
    print(f"Files skipped:  {summary['total_files_skipped']}")
    print(f"Chunks indexed: {summary['total_chunks_indexed']}")
    print(f"Collection:     {summary['collection_name']}")
    assert summary["total_chunks_indexed"] > 0, "Nothing was indexed!"
    print("PASS")


def test_already_indexed():
    print("\n--- TEST 4: is_repo_indexed check ---")
    result = is_repo_indexed(GITHUB_URL)
    print(f"is_repo_indexed: {result}")
    print("PASS")


if __name__ == "__main__":
    test_clone_and_walk()
    test_parse_single_file()
    test_full_index()
    test_already_indexed()
    print("\n All ingestion tests passed.")
