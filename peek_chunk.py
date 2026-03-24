"""
peek_chunk.py  —  Print one chunk from ChromaDB in a readable format.

Usage (run from inside debugger/):
    python peek_chunk.py                          # auto-picks the first chunk
    python peek_chunk.py create_order             # find by function name
    python peek_chunk.py orderController          # find by file name substring
    python peek_chunk.py create_order --raw       # also dump raw dict at the end
"""

import os
import sys
import json
import textwrap

# ---------------------------------------------------------------------------
# Config — must match what indexer.py used
# ---------------------------------------------------------------------------
_BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR      = os.environ.get("CHROMA_DIR", os.path.join(_BASE_DIR, "debugger_chroma"))
REPO_FULL_NAME  = "avneeshjarangal-dot/quickcart"   # change if you index a different repo


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def collection_name(repo: str) -> str:
    return repo.replace("/", "--").replace("_", "-").lower()


def load_collection():
    import chromadb
    from chromadb.utils import embedding_functions
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    return client.get_collection(
        name=collection_name(REPO_FULL_NAME),
        embedding_function=ef,
    )


def fetch_chunk(collection, query: str | None) -> dict | None:
    """
    Return one chunk.
    - No query  → first chunk in the collection.
    - query     → search by name first (exact), then file_path substring,
                  then fall back to semantic search.
    """
    if query is None:
        result = collection.get(limit=1, include=["metadatas", "documents"])
        if not result["ids"]:
            return None
        return _build(result, 0)

    # 1. exact name match
    result = collection.get(
        where={"name": {"$eq": query}},
        limit=1,
        include=["metadatas", "documents"],
    )
    if result["ids"]:
        return _build(result, 0)

    # 2. file_path substring (ChromaDB doesn't support LIKE, so fetch + filter)
    result = collection.get(
        limit=500,
        include=["metadatas", "documents"],
    )
    for i, meta in enumerate(result["metadatas"]):
        if query.lower() in meta.get("file_path", "").lower():
            return _build(result, i)

    # 3. semantic fallback
    result = collection.query(
        query_texts=[query],
        n_results=1,
        include=["metadatas", "documents"],
    )
    if result["ids"] and result["ids"][0]:
        return {
            "id":       result["ids"][0][0],
            "document": result["documents"][0][0],
            "metadata": result["metadatas"][0][0],
        }

    return None


def _build(result, i):
    return {
        "id":       result["ids"][i],
        "document": result["documents"][i],
        "metadata": result["metadatas"][i],
    }


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------
SEP  = "─" * 72
SEP2 = "═" * 72

def display(chunk: dict, raw: bool = False):
    m = chunk["metadata"]

    def row(label, value):
        print(f"  {label:<18} {value}")

    def json_list(val):
        try:
            items = json.loads(val) if isinstance(val, str) else val
            return ", ".join(items) if items else "—"
        except Exception:
            return str(val) or "—"

    print()
    print(SEP2)
    print(f"  CHUNK: {chunk['id']}")
    print(SEP2)

    print()
    print("  [ identity ]")
    print(SEP)
    row("chunk_type",   m.get("chunk_type", "—"))
    row("name",         m.get("name", "—"))
    row("file_path",    m.get("file_path", "—"))
    row("language",     m.get("language", "—"))
    row("lines",        f"{m.get('start_line')} → {m.get('end_line')}")
    row("parent_class", m.get("parent_class") or "—")
    row("signature",    m.get("signature") or "—")

    print()
    print("  [ docstring ]")
    print(SEP)
    doc = m.get("docstring") or "—"
    for line in textwrap.wrap(doc, width=68):
        print(f"  {line}")

    print()
    print("  [ call graph ]")
    print(SEP)
    row("calls",      json_list(m.get("calls", "[]")))
    row("called_by",  json_list(m.get("called_by", "[]")))

    print()
    print("  [ imports ]")
    print(SEP)
    try:
        imports = json.loads(m.get("imports", "[]"))
        if imports:
            for imp in imports:
                print(f"  {imp}")
        else:
            print("  —")
    except Exception:
        print(f"  {m.get('imports', '—')}")

    print()
    print("  [ source code ]")
    print(SEP)
    code = chunk["document"]
    for line in code.splitlines():
        print(f"  {line}")

    if raw:
        print()
        print("  [ raw dict ]")
        print(SEP)
        print(json.dumps(chunk, indent=4))

    print()
    print(SEP2)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args  = sys.argv[1:]
    raw   = "--raw" in args
    terms = [a for a in args if not a.startswith("--")]
    query = terms[0] if terms else None

    try:
        collection = load_collection()
    except Exception as e:
        print(f"\n  ERROR loading collection: {e}")
        print(f"  Make sure you've run ingest.py first and CHROMA_DIR is correct.")
        print(f"  CHROMA_DIR = {CHROMA_DIR}\n")
        sys.exit(1)

    total = collection.count()
    print(f"\n  Collection '{collection_name(REPO_FULL_NAME)}' has {total} chunks.")

    chunk = fetch_chunk(collection, query)
    if chunk is None:
        print(f"  No chunk found for query: '{query}'\n")
        sys.exit(1)

    display(chunk, raw=raw)


if __name__ == "__main__":
    main()