import os
import json
import logging
import chromadb
from chromadb.utils import embedding_functions
from core.ingestion.repo_cloner import clone_repo, parse_repo_url, get_local_path
from core.ingestion.file_walker import walk_repo, get_file_content
from core.ingestion.code_parser import parse_file, build_called_by

logger = logging.getLogger(__name__)

# walk up: ingestion/ -> core/ -> debugger/
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CHROMA_DIR     = os.environ.get("CHROMA_DIR",     os.path.join(_BASE_DIR, "debugger_chroma"))
CLONE_BASE_DIR = os.environ.get("CLONE_BASE_DIR", os.path.join(_BASE_DIR, "debugger_repos"))

# local sentence-transformers — no API cost
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def get_chroma_client() -> chromadb.Client:
    """Return a persistent ChromaDB client stored inside debugger/."""
    return chromadb.PersistentClient(path=CHROMA_DIR)


def get_embedding_function():
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )


def get_collection(repo_full_name: str):
    """
    Get or create a ChromaDB collection for a specific repo.
    Collection name: owner--repo  (slashes and underscores replaced).
    """
    client = get_chroma_client()
    ef = get_embedding_function()
    collection_name = _safe_collection_name(repo_full_name)
    return client.get_or_create_collection(
        name=collection_name,
        embedding_function=ef,
        metadata={"repo": repo_full_name},
    )


def _safe_collection_name(repo_full_name: str) -> str:
    """ChromaDB collection names: alphanumeric + hyphens only."""
    return repo_full_name.replace("/", "--").replace("_", "-").lower()


def index_repo(github_url: str, github_token: str, force: bool = False) -> dict:
    """
    Full ingestion pipeline.

    Pass 1 — Clone → walk every source file → parse with Tree-sitter.
              Produces chunks with: code, calls, parent_class, signature,
              docstring, imports. called_by is empty at this stage.

    Pass 2 — Invert the call graph across ALL chunks from ALL files.
              Populates called_by on every chunk.

    Pass 3 — Embed all chunks with all-MiniLM-L6-v2 and upsert into
              ChromaDB in batches of 100.

    Returns:
        Summary dict with counts and the ChromaDB collection name.
    """
    repo_info = parse_repo_url(github_url)
    logger.info(f"Starting ingestion for {repo_info['full_name']}")

    # ------------------------------------------------------------------
    # Clone
    # ------------------------------------------------------------------
    local_path = clone_repo(github_url, github_token, force=force)

    # ------------------------------------------------------------------
    # ChromaDB collection — reset if force
    # ------------------------------------------------------------------
    if force:
        client = get_chroma_client()
        col_name = _safe_collection_name(repo_info["full_name"])
        try:
            client.delete_collection(col_name)
            logger.info("Cleared existing index for re-indexing")
        except Exception:
            pass  # collection didn't exist yet — that's fine

    collection = get_collection(repo_info["full_name"])

    # ------------------------------------------------------------------
    # Pass 1 — walk + parse
    # ------------------------------------------------------------------
    total_files   = 0
    skipped_files = 0
    all_chunks: list[dict] = []

    logger.info("Pass 1 — parsing source files...")

    for file_info in walk_repo(local_path):
        total_files += 1
        try:
            content = get_file_content(file_info["path"])
        except Exception as e:
            logger.warning(f"Could not read {file_info['relative_path']}: {e}")
            skipped_files += 1
            continue

        chunks = parse_file(
            file_path=file_info["relative_path"],
            content=content,
            language=file_info["language"],
        )
        all_chunks.extend(chunks)
        logger.debug(f"  {file_info['relative_path']} → {len(chunks)} chunk(s)")

    logger.info(
        f"Pass 1 complete — {len(all_chunks)} chunks from "
        f"{total_files} files ({skipped_files} skipped)"
    )

    # ------------------------------------------------------------------
    # Pass 2 — invert call graph → populate called_by
    # ------------------------------------------------------------------
    logger.info("Pass 2 — inverting call graph (called_by)...")
    build_called_by(all_chunks)  # modifies in-place

    chunks_with_callers = sum(1 for c in all_chunks if c["called_by"])
    logger.info(
        f"Pass 2 complete — {chunks_with_callers}/{len(all_chunks)} chunks "
        f"have at least one known caller"
    )

    # ------------------------------------------------------------------
    # Pass 3 — embed + upsert into ChromaDB
    # ------------------------------------------------------------------
    logger.info("Pass 3 — embedding and upserting into ChromaDB...")

    ids       = [c["chunk_id"] for c in all_chunks]
    documents = [c["code"]     for c in all_chunks]

    # ChromaDB metadata values must be scalar (str/int/float/bool).
    # Lists are serialised to JSON strings; deserialise with json.loads() on read.
    metadatas = [
        {
            "file_path":    c["file_path"],
            "language":     c["language"],
            "chunk_type":   c["chunk_type"],
            "name":         c["name"],
            "start_line":   c["start_line"],
            "end_line":     c["end_line"],
            "context":      c["context"],
            "parent_class": c["parent_class"] or "",
            "signature":    c["signature"]    or "",
            "docstring":    c["docstring"]    or "",
            "imports":      json.dumps(c["imports"]),
            "calls":        json.dumps(c["calls"]),
            "called_by":    json.dumps(c["called_by"]),
        }
        for c in all_chunks
    ]

    BATCH_SIZE = 100
    total_batches = (len(all_chunks) + BATCH_SIZE - 1) // BATCH_SIZE
    for i in range(0, len(all_chunks), BATCH_SIZE):
        batch_num = i // BATCH_SIZE + 1
        end       = min(i + BATCH_SIZE, len(all_chunks))
        collection.upsert(
            ids=ids[i:end],
            documents=documents[i:end],
            metadatas=metadatas[i:end],
        )
        logger.debug(f"  Upserted batch {batch_num}/{total_batches} ({end}/{len(all_chunks)} chunks)")

    logger.info(
        f"Pass 3 complete — {len(all_chunks)} chunks upserted into "
        f"collection '{collection.name}'"
    )

    return {
        "repo":                  repo_info["full_name"],
        "local_path":            local_path,
        "total_files_walked":    total_files,
        "total_files_skipped":   skipped_files,
        "total_chunks_indexed":  len(all_chunks),
        "chunks_with_callers":   chunks_with_callers,
        "collection_name":       collection.name,
        "chroma_dir":            CHROMA_DIR,
    }


def is_repo_indexed(github_url: str) -> bool:
    """Return True if the repo has already been indexed (collection is non-empty)."""
    try:
        repo_info = parse_repo_url(github_url)
        collection = get_collection(repo_info["full_name"])
        return collection.count() > 0
    except Exception:
        return False