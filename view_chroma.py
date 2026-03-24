"""
View what's stored in your ChromaDB after indexing.

Usage:
    python view_chroma.py                        # show all collections + summary
    python view_chroma.py --list                 # list all codebases
    python view_chroma.py --chunks               # list every chunk
    python view_chroma.py --file orderController # filter by file name
    python view_chroma.py --search "user address pincode"  # semantic search
    python view_chroma.py --export out.json      # export results to JSON
"""

import argparse
import os
import json
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()

CHROMA_DIR = os.environ.get("CHROMA_DIR", "/tmp/debugger_chroma")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def get_client():
    return chromadb.PersistentClient(path=CHROMA_DIR)


def get_ef():
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )


def list_collections():
    """Lists all collections (codebases) available in the DB."""
    client = get_client()
    collections = client.list_collections()
    if not collections:
        print("No collections found.")
        return []

    print(f"\nAvailable Codebases (Collections) in {CHROMA_DIR}:")
    print("-" * 60)
    for col in collections:
        count = client.get_collection(col.name).count()
        print(f" - {col.name:<30} | {count:>5} chunks")
    return [col.name for col in collections]


def show_summary():
    client = get_client()
    collections = client.list_collections()

    if not collections:
        print("No collections found. Run test_ingestion.py first.")
        return

    print(f"\n{'='*60}")
    print(f"ChromaDB at: {CHROMA_DIR}")
    print(f"Total collections: {len(collections)}")
    print(f"{'='*60}")

    for col in collections:
        c = client.get_collection(col.name)
        print(f"\nCollection : {col.name}")
        print(f"Chunks     : {c.count()}")
        print(f"Metadata   : {col.metadata}")


def list_chunks(collection_name: str = None, file_filter: str = None, export_path: str = None):
    client = get_client()
    collections = client.list_collections()

    if not collections:
        print("No collections found.")
        return

    col_name = collection_name if collection_name else collections[0].name

    ef = get_ef()
    try:
        collection = client.get_collection(col_name, embedding_function=ef)
    except ValueError:
        print(f"Error: Collection '{col_name}' not found.")
        return

    total = collection.count()
    results = collection.get(include=["metadatas", "documents"])

    ids = results["ids"]
    metadatas = results["metadatas"]
    documents = results["documents"]

    export_data = []
    files_view = {}

    for i, meta in enumerate(metadatas):
        fp = meta.get("file_path", "unknown")
        if file_filter and file_filter.lower() not in fp.lower():
            continue

        chunk_info = {
            "id": ids[i],
            "metadata": meta,
            "document": documents[i],
            # OLD style: plain 120-char preview without trailing "..."
            "code_preview": documents[i][:120].replace("\n", " "),
        }
        export_data.append(chunk_info)

        if fp not in files_view:
            files_view[fp] = []
        files_view[fp].append(chunk_info)

    if export_path:
        with open(export_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        print(f"\n✅ Exported {len(export_data)} chunks to {export_path}")
        return

    if not files_view:
        print(f"No chunks matching filter: '{file_filter}'")
        return

    print(f"\nCollection: {col_name} ({total} chunks total)")
    print(f"{'='*60}")

    for file_path, chunks in sorted(files_view.items()):
        print(f"\n📄 {file_path}")
        for chunk in chunks:
            m = chunk["metadata"]
            # OLD display style: fn/cls label + name + lines + preview + "..."
            print(
                f"   {'fn' if m.get('chunk_type') == 'function' else 'cls':3} | "
                f"{m.get('name', 'N/A'):35} | "
                f"lines {m.get('start_line', 0):4}-{m.get('end_line', 0):4} | "
                f"{chunk['code_preview']}..."
            )


def semantic_search(query: str, collection_name: str = None, n: int = 5, export_path: str = None):
    client = get_client()
    collections = client.list_collections()

    if not collections:
        print("No collections found.")
        return

    col_name = collection_name if collection_name else collections[0].name
    ef = get_ef()

    try:
        collection = client.get_collection(col_name, embedding_function=ef)
    except ValueError:
        print(f"Error: Collection '{col_name}' not found.")
        return

    results = collection.query(
        query_texts=[query],
        n_results=n,
        include=["metadatas", "documents", "distances"],
    )

    export_results = []
    for i in range(len(results["ids"][0])):
        meta = results["metadatas"][0][i]
        export_results.append({
            "relevance": round((1 - results["distances"][0][i]) * 100, 1),
            "file_path": meta.get("file_path", "unknown"),
            "name": meta.get("name", "N/A"),
            "lines": f"{meta.get('start_line', 0)}-{meta.get('end_line', 0)}",
            "type": meta.get("chunk_type", "unknown"),
            "content": results["documents"][0][i],
        })

    if export_path:
        with open(export_path, 'w') as f:
            json.dump(export_results, f, indent=2)
        print(f"\n✅ Exported {len(export_results)} search results to {export_path}")
        return

    print(f"\nSearching for: \"{query}\"")
    print(f"Collection:    {col_name}")
    print(f"{'='*60}")

    for i, res in enumerate(export_results):
        print(f"\n#{i+1} — relevance: {res['relevance']}%")
        print(f"  File  : {res['file_path']}")
        print(f"  Name  : {res['name']}()")
        print(f"  Lines : {res['lines']}")
        print(f"  Type  : {res['type']}")
        print(f"  Code  :")
        print("-" * 40)
        # OLD style: first 20 lines + trailing count if truncated
        lines = res['content'].splitlines()
        for line in lines[:20]:
            print(f"  {line}")
        if len(lines) > 20:
            print(f"  ... ({len(lines) - 20} more lines)")
        print("-" * 40)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="View ChromaDB contents")
    parser.add_argument("--list", action="store_true", help="List all available codebases (collections)")
    parser.add_argument("--chunks", action="store_true", help="List all chunks grouped by file")
    parser.add_argument("--file", type=str, help="Filter chunks by file name")
    parser.add_argument("--search", type=str, help="Semantic search query")
    parser.add_argument("--collection", type=str, help="Collection name (default: first one found)")
    parser.add_argument("--export", type=str, help="Save output to a JSON file")
    parser.add_argument("--n", type=int, default=5, help="Number of search results (default: 5)")
    args = parser.parse_args()

    if args.list:
        list_collections()
    elif args.search:
        semantic_search(args.search, collection_name=args.collection, n=args.n, export_path=args.export)
    elif args.chunks or args.file:
        list_chunks(collection_name=args.collection, file_filter=args.file, export_path=args.export)
    else:
        show_summary()
        print("\nTip: run with --chunks to see all chunks, or --search 'your query'")