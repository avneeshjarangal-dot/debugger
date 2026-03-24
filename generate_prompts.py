"""
generate_prompts.py  —  Step 3: Map log groups → code chunks → generate fix prompts.

What this does:
  1. Reads grouped logs produced by fetch_logs.group_logs()
  2. For each group, finds the relevant code chunk(s) from ChromaDB
       - primary   : exact match on (file_path, function name, line number)
       - callers   : chunks that call the failing function (called_by graph)
       - semantic  : vector similarity search using the error message
  3. Assembles a rich context block (error, traceback, code, call graph)
  4. Calls gemini    to turn that context into a precise, actionable fix prompt
  5. Prints + saves all generated prompts to  debugger/generated_prompts/

Usage (run from inside debugger/):
    python generate_prompts.py
    python generate_prompts.py --service order-service
    python generate_prompts.py --hours 6 --top 3
    python generate_prompts.py --no-save

Env vars (same as fetch_logs.py):
    MONGO_URI   MONGO_DB   MONGO_COLL   GEMINI_API_KEY
"""

import os
import sys
import json
import argparse
from datetime import timezone
from pathlib import Path

from config import load_config
from llm_client import GEMINI_MODEL, GROQ_MODEL, OPENAI_MODEL, OPENROUTER_MODEL, generate_text

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
_BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR      = os.environ.get("CHROMA_DIR",  os.path.join(_BASE_DIR, "debugger_chroma"))
PROMPTS_DIR     = os.path.join(_BASE_DIR, "generated_prompts")
DEFAULT_MONGO_URI  = os.environ.get("MONGO_URI")
DEFAULT_DB_NAME    = os.environ.get("MONGO_DB", "log")
DEFAULT_COLLECTION = os.environ.get("MONGO_COLLECTION", "logs")
DEFAULT_HOURS      = 120
DEFAULT_TOP        = 1     # process top-N groups by occurrence count
SEMANTIC_RESULTS   = 3       # extra semantic chunks to pull per group


# ---------------------------------------------------------------------------
# ChromaDB helpers
# ---------------------------------------------------------------------------
def _collection_name(repo: str) -> str:
    return repo.replace("/", "--").replace("_", "-").lower()


def get_collection(repo_full_name: str | None = None):
    import chromadb
    from chromadb.utils import embedding_functions

    resolved_repo = repo_full_name or load_config().repo_full_name
    if not resolved_repo:
        raise RuntimeError("REPO_FULL_NAME is not configured.")

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    return client.get_collection(
        name=_collection_name(resolved_repo),
        embedding_function=ef,
    )


def _deserialize(meta: dict) -> dict:
    """JSON-decode list fields that were stored as strings by indexer.py."""
    for field in ("calls", "called_by", "imports"):
        raw = meta.get(field, "[]")
        try:
            meta[field] = json.loads(raw) if isinstance(raw, str) else raw
        except Exception:
            meta[field] = []
    return meta


# ---------------------------------------------------------------------------
# Chunk lookup — three strategies, merged and deduplicated
# ---------------------------------------------------------------------------
def find_chunks_for_group(collection, group: dict) -> dict:
    """
    Return a dict with keys:
        primary       — the exact chunk where the error occurred (or None)
        callers       — chunks that call the failing function
        semantic      — semantically similar chunks
    """
    rep       = group["representative"]
    func_name = group["function"]
    raw_file  = rep.get("file", "")
    raw_line  = rep.get("line")
    message   = rep.get("message", "")
    traceback = rep.get("traceback", "")

    # normalise file path — strip leading /app/ or similar deployment prefixes
    # and keep the portion that looks like a relative repo path
    file_hint = _normalise_path(raw_file)

    primary  = None
    callers  = []
    semantic = []
    seen_ids = set()

    # ------------------------------------------------------------------
    # 1. Exact match: function name + file path
    # ------------------------------------------------------------------
    try:
        result = collection.get(
            where={"name": {"$eq": func_name}},
            include=["metadatas", "documents"],
        )
        for i, meta in enumerate(result["metadatas"]):
            chunk_id  = result["ids"][i]
            meta      = _deserialize(meta)
            fp        = meta.get("file_path", "")

            # prefer the chunk whose file_path overlaps with the log file hint
            if file_hint and not _path_overlap(fp, file_hint):
                continue

            # if line number known, prefer the chunk that contains it
            if raw_line:
                if not (meta["start_line"] <= int(raw_line) <= meta["end_line"]):
                    continue

            primary = {
                "chunk_id": chunk_id,
                "document": result["documents"][i],
                "metadata": meta,
                "match_type": "exact",
            }
            seen_ids.add(chunk_id)
            break

        # fallback: just match on name without line constraint
        if primary is None:
            for i, meta in enumerate(result["metadatas"]):
                chunk_id = result["ids"][i]
                if chunk_id in seen_ids:
                    continue
                meta = _deserialize(meta)
                fp   = meta.get("file_path", "")
                if file_hint and not _path_overlap(fp, file_hint):
                    continue
                primary = {
                    "chunk_id": chunk_id,
                    "document": result["documents"][i],
                    "metadata": meta,
                    "match_type": "name_only",
                }
                seen_ids.add(chunk_id)
                break
    except Exception:
        pass

    # ------------------------------------------------------------------
    # 2. Caller chunks — who calls the failing function?
    # ------------------------------------------------------------------
    if primary:
        called_by_names = primary["metadata"].get("called_by", [])
        for caller_name in called_by_names[:5]:   # cap at 5 callers
            try:
                res = collection.get(
                    where={"name": {"$eq": caller_name}},
                    include=["metadatas", "documents"],
                )
                for i, meta in enumerate(res["metadatas"]):
                    cid = res["ids"][i]
                    if cid in seen_ids:
                        continue
                    callers.append({
                        "chunk_id": cid,
                        "document": res["documents"][i],
                        "metadata": _deserialize(meta),
                        "match_type": "caller",
                    })
                    seen_ids.add(cid)
                    break
            except Exception:
                continue

    # ------------------------------------------------------------------
    # 3. Semantic search — error message + traceback as query
    # ------------------------------------------------------------------
    semantic_query = f"{message}\n{traceback}".strip()
    if semantic_query:
        try:
            res = collection.query(
                query_texts=[semantic_query],
                n_results=SEMANTIC_RESULTS + len(seen_ids),
                include=["metadatas", "documents"],
            )
            for i, cid in enumerate(res["ids"][0]):
                if cid in seen_ids:
                    continue
                meta = _deserialize(res["metadatas"][0][i])
                semantic.append({
                    "chunk_id": cid,
                    "document": res["documents"][0][i],
                    "metadata": meta,
                    "match_type": "semantic",
                })
                seen_ids.add(cid)
                if len(semantic) >= SEMANTIC_RESULTS:
                    break
        except Exception:
            pass

    return {
        "primary":  primary,
        "callers":  callers,
        "semantic": semantic,
    }


def _normalise_path(raw: str) -> str:
    """Strip deployment prefixes like /app/quickcart/ from log file paths."""
    p = raw.replace("\\", "/")
    # drop everything up to and including known prefixes
    for prefix in ("/app/quickcart/", "/app/", "/home/", "/usr/"):
        if prefix in p:
            p = p[p.index(prefix) + len(prefix):]
            break
    return p.lstrip("/")


def _path_overlap(chunk_path: str, log_path: str) -> bool:
    """True if the two paths share enough tail segments to be the same file."""
    a = chunk_path.replace("\\", "/").strip("/").split("/")
    b = log_path.replace("\\", "/").strip("/").split("/")
    # compare up to 3 tail segments
    tail = min(3, len(a), len(b))
    return a[-tail:] == b[-tail:]


# ---------------------------------------------------------------------------
# Context assembly
# ---------------------------------------------------------------------------
def fmt_ts(ts) -> str:
    if ts is None:
        return "—"
    if hasattr(ts, "isoformat"):
        return ts.replace(tzinfo=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    if isinstance(ts, dict) and "$date" in ts:
        return ts["$date"]
    return str(ts)


def build_context(group: dict, chunks: dict) -> str:
    """
    Assemble a structured context block that will be fed to the
    prompt-generation LLM call.
    """
    rep   = group["representative"]
    lines = []

    def section(title):
        lines.append(f"\n{'─' * 60}")
        lines.append(f"# {title}")
        lines.append(f"{'─' * 60}")

    # ── Error summary ──────────────────────────────────────────────
    section("ERROR SUMMARY")
    lines.append(f"Service   : {group['service']}")
    lines.append(f"Level     : {group['level']}")
    lines.append(f"Occurrences: {group['count']} time(s) in the last window")

    valid_ts = [t for t in group["timestamps"] if t is not None]
    if valid_ts:
        lines.append(f"First seen: {fmt_ts(min(valid_ts))}")
        lines.append(f"Last seen : {fmt_ts(max(valid_ts))}")

    lines.append(f"\nError message:\n  {rep.get('message', '—')}")

    # ── Traceback ──────────────────────────────────────────────────
    tb = rep.get("traceback")
    if tb:
        section("TRACEBACK")
        lines.append(tb)

    # ── Extra payload samples ──────────────────────────────────────
    if group.get("extra_samples"):
        section("EXAMPLE REQUEST CONTEXT (from extra field)")
        for sample in group["extra_samples"]:
            lines.append(json.dumps(sample, default=str, indent=2))

    # ── Primary chunk ──────────────────────────────────────────────
    primary = chunks["primary"]
    if primary:
        m = primary["metadata"]
        section(f"FAILING FUNCTION: {m.get('name')} ({m.get('chunk_type')})")
        lines.append(f"File      : {m.get('file_path')}")
        lines.append(f"Lines     : {m.get('start_line')} – {m.get('end_line')}")
        lines.append(f"Signature : {m.get('signature') or '—'}")
        if m.get("docstring"):
            lines.append(f"Docstring : {m['docstring']}")
        if m.get("calls"):
            lines.append(f"Calls     : {', '.join(m['calls'])}")
        if m.get("called_by"):
            lines.append(f"Called by : {', '.join(m['called_by'])}")
        lines.append(f"\n```{m.get('language', '')}")
        lines.append(primary["document"])
        lines.append("```")

    # ── Caller chunks ──────────────────────────────────────────────
    if chunks["callers"]:
        section("CALLER FUNCTIONS (functions that invoke the failing function)")
        for chunk in chunks["callers"]:
            m = chunk["metadata"]
            lines.append(f"\n### {m.get('name')}()  —  {m.get('file_path')}:{m.get('start_line')}")
            lines.append(f"```{m.get('language', '')}")
            lines.append(chunk["document"])
            lines.append("```")

    # ── Semantic neighbours ────────────────────────────────────────
    if chunks["semantic"]:
        section("RELATED CODE (semantically similar chunks)")
        for chunk in chunks["semantic"]:
            m = chunk["metadata"]
            lines.append(f"\n### {m.get('name')}()  —  {m.get('file_path')}:{m.get('start_line')}")
            lines.append(f"```{m.get('language', '')}")
            lines.append(chunk["document"])
            lines.append("```")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompt generation via gemini  
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are a senior software engineer specialising in debugging production issues.

Your job is to read a structured error report (containing the error message,
traceback, failing source code, and calling context) and produce a
SINGLE, SELF-CONTAINED PROMPT that another LLM can use to fix the bug.

The prompt you write must strongly constrain the fixing model:
1. Name the primary failing file as the canonical target file.
2. Instruct the fixing model to use that exact file path verbatim.
3. Instruct the fixing model to default to changing only that one file.
4. Allow extra files only if the provided caller/related code proves they are necessary.
5. Require full-function or full-block replacements for any patch plan entries.
6. Forbid speculative extra fixes outside the evidence in the report.

The output prompt you write must:
1. State the exact bug clearly in one sentence.
2. Include the full failing code as a fenced code block.
3. Explain WHY the error occurs (root cause).
4. Explicitly identify the canonical primary file path and line number.
5. List every file and line number that needs to change, defaulting to the primary file only.
6. End with a "Verification" section describing how to confirm the fix works.

Output ONLY the prompt text. No preamble, no explanation, no markdown wrapper.
"""


def generate_fix_prompt(context: str, group: dict, provider: str = "GEMINI") -> str:
    """Call the selected LLM provider to convert context into a fix prompt."""

    user_message = (
        f"Here is the full error context for a production bug in the "
        f"'{group['service']}' service.\n\n"
        f"{context}\n\n"
        f"Generate the fix prompt now."
    )

    try:
        return generate_text(provider, SYSTEM_PROMPT + "\n\n" + user_message)
    except Exception as e:
        raise RuntimeError(f"{provider} API error: {e}")

   

   

    


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
def save_prompt(group: dict, prompt: str, index: int) -> str:
    """Save the generated prompt to debugger/generated_prompts/ and return the path."""
    os.makedirs(PROMPTS_DIR, exist_ok=True)
    safe_func    = group["function"].replace("/", "_").replace(" ", "_")
    safe_service = group["service"].replace("/", "_").replace(" ", "_")
    filename     = f"{index:02d}_{safe_service}__{safe_func}.txt"
    path         = os.path.join(PROMPTS_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(prompt)
    return path


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------
SEP  = "─" * 72
SEP2 = "═" * 72


def display_result(index: int, group: dict, chunks: dict, prompt: str, saved_path: str | None):
    print()
    print(SEP2)
    print(f"  ISSUE #{index}  |  {group['level']}  |  {group['service']}::{group['function']}  |  ×{group['count']}")
    print(SEP2)

    primary = chunks["primary"]
    if primary:
        m = primary["metadata"]
        print(f"  Primary chunk : {m.get('file_path')}:{m.get('start_line')}–{m.get('end_line')}")
        print(f"  Match type    : {primary['match_type']}")
    else:
        print("  Primary chunk : not found")

    print(f"  Caller chunks : {len(chunks['callers'])}")
    print(f"  Semantic hits : {len(chunks['semantic'])}")

    if saved_path:
        print(f"  Saved to      : {saved_path}")

    print()
    print("  [ GENERATED FIX PROMPT ]")
    print(SEP)
    for line in prompt.splitlines():
        print(f"  {line}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Map log groups to code chunks and generate fix prompts",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--uri",        default=DEFAULT_MONGO_URI)
    parser.add_argument("--db",         default=DEFAULT_DB_NAME)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--service",    default=None,           help="Filter logs by service")
    parser.add_argument("--hours",      default=DEFAULT_HOURS,  type=int)
    parser.add_argument("--top",        default=DEFAULT_TOP,    type=int,
                        help="Process top-N groups by occurrence count")
    parser.add_argument("--no-save",    action="store_true",    help="Don't save prompts to disk")
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Step 1: fetch + group logs ──────────────────────────────────
    print("\n  Loading fetch_logs module...")
    sys.path.insert(0, _BASE_DIR)
    try:
        from fetch_logs import fetch_logs, group_logs, DEFAULT_LEVELS
    except ImportError as e:
        print(f"  ERROR: could not import fetch_logs.py — {e}")
        sys.exit(1)

    print(f"  Fetching logs  (last {args.hours}h, levels: {', '.join(DEFAULT_LEVELS)})...")
    try:
        logs = fetch_logs(
            uri=args.uri,
            db_name=args.db,
            collection_name=args.collection,
            levels=DEFAULT_LEVELS,
            service=args.service,
            hours=args.hours,
        )
    except Exception as e:
        print(f"  ERROR fetching logs: {e}")
        sys.exit(1)

    if not logs:
        print("  No logs found. Nothing to process.\n")
        return

    groups = group_logs(logs)
    top_groups = groups[: args.top]

    print(f"  {len(logs)} log(s) → {len(groups)} group(s) → processing top {len(top_groups)}\n")

    # ── Step 2: load ChromaDB collection ───────────────────────────
    print("  Loading ChromaDB collection...")
    try:
        collection = get_collection()
        print(f"  Collection ready — {collection.count()} chunks indexed\n")
    except Exception as e:
        print(f"  ERROR loading ChromaDB: {e}")
        print(f"  Make sure ingest.py has been run first.")
        sys.exit(1)

    # ── Step 3: per-group — find chunks, build context, generate ───
    for i, group in enumerate(top_groups, 1):
        print(f"  [{i}/{len(top_groups)}] {group['service']}::{group['function']}  (×{group['count']})")

        # find relevant chunks
        chunks = find_chunks_for_group(collection, group)

        if not chunks["primary"] and not chunks["semantic"]:
            print(f"         No matching chunks found — skipping\n")
            continue

        # assemble context
        context = build_context(group, chunks)

        # call gemini   
        print(f"         Generating fix prompt via gemini   ...")
        try:
            prompt = generate_fix_prompt(context, group)
        except Exception as e:
            print(f"         ERROR generating prompt: {e}\n")
            continue

        # save
        saved_path = None
        if not args.no_save:
            saved_path = save_prompt(group, prompt, i)

        display_result(i, group, chunks, prompt, saved_path)

    print(f"\n  Done. Prompts saved to: {PROMPTS_DIR}\n")


if __name__ == "__main__":
    main()