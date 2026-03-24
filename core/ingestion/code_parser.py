import logging
from typing import Optional
from tree_sitter import Language, Parser
import tree_sitter_python as tspython
import tree_sitter_javascript as tsjavascript
import tree_sitter_typescript as tstypescript

logger = logging.getLogger(__name__)

# --- build Language objects from tree-sitter grammar bindings ---
PY_LANGUAGE = Language(tspython.language())
JS_LANGUAGE = Language(tsjavascript.language())
TS_LANGUAGE = Language(tstypescript.language_typescript())

PARSERS = {
    "python": None,
    "javascript": None,
    "typescript": None,
}


def get_parser(language: str) -> Parser:
    """Return a cached Parser for the given language."""
    if PARSERS[language] is None:
        parser = Parser()
        if language == "python":
            parser.language = PY_LANGUAGE
        elif language == "javascript":
            parser.language = JS_LANGUAGE
        elif language == "typescript":
            parser.language = TS_LANGUAGE
        PARSERS[language] = parser
    return PARSERS[language]


# node types that represent a callable block in each language
FUNCTION_NODE_TYPES = {
    "python": {"function_definition", "async_function_definition"},
    "javascript": {
        "function_declaration",
        "function_expression",
        "arrow_function",
        "method_definition",
    },
    "typescript": {
        "function_declaration",
        "function_expression",
        "arrow_function",
        "method_definition",
    },
}

CLASS_NODE_TYPES = {
    "python": {"class_definition"},
    "javascript": {"class_declaration", "class_expression"},
    "typescript": {"class_declaration", "class_expression"},
}

IMPORT_NODE_TYPES = {
    "python": {"import_statement", "import_from_statement"},
    "javascript": {"import_declaration", "import_statement"},
    "typescript": {"import_declaration", "import_statement"},
}


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------

def extract_node_name(node, source_bytes: bytes, language: str) -> Optional[str]:
    """Extract the name identifier from a function or class node."""
    for child in node.children:
        if child.type == "identifier":
            return source_bytes[child.start_byte:child.end_byte].decode("utf-8")
    return None


def get_parent_class(node, source_bytes: bytes, language: str) -> Optional[str]:
    """Walk up the AST to find the enclosing class name, if any."""
    cls_types = CLASS_NODE_TYPES.get(language, set())
    parent = node.parent
    while parent:
        if parent.type in cls_types:
            return extract_node_name(parent, source_bytes, language)
        parent = parent.parent
    return None


def get_signature(node, source_bytes: bytes) -> str:
    """Extract the raw parameter signature string from a function node."""
    for child in node.children:
        if child.type == "parameters":
            return source_bytes[child.start_byte:child.end_byte].decode("utf-8")
    return "()"


def get_docstring(node, source_bytes: bytes, language: str) -> Optional[str]:
    """
    Extract the docstring / leading comment from a function or class node.
    Python  — first string literal inside the body block.
    JS/TS   — block comment immediately before the node.
    """
    if language == "python":
        for child in node.children:
            if child.type == "block":
                for stmt in child.children:
                    if stmt.type == "expression_statement":
                        for inner in stmt.children:
                            if inner.type == "string":
                                raw = source_bytes[inner.start_byte:inner.end_byte].decode("utf-8")
                                return raw.strip("\"'").strip()
                        break
                    break
    elif language in ("javascript", "typescript"):
        prev = node.prev_sibling
        if prev and prev.type == "comment":
            raw = source_bytes[prev.start_byte:prev.end_byte].decode("utf-8")
            return raw.strip("/* \n").strip()
    return None


def get_calls(node, source_bytes: bytes) -> list[str]:
    """
    Recursively collect every function / method name called inside a node.

    Captures:
        plain calls   — get_user_by_id(...)  →  "get_user_by_id"
        method calls  — db.find_one(...)     →  "db.find_one"
                        Cart.get(...)        →  "Cart.get"

    Returns a deduplicated list preserving order of first appearance.
    Library / built-in calls are kept in the list — they are filtered out
    during the inversion step in build_called_by().
    """
    calls = []

    def walk(n):
        if n.type == "call":
            if n.children:
                callee = n.children[0]
                if callee.type == "identifier":
                    calls.append(
                        source_bytes[callee.start_byte:callee.end_byte].decode("utf-8")
                    )
                elif callee.type in ("attribute", "member_expression"):
                    calls.append(
                        source_bytes[callee.start_byte:callee.end_byte].decode("utf-8")
                    )
        for child in n.children:
            walk(child)

    walk(node)

    # deduplicate, preserve order
    seen = set()
    unique = []
    for c in calls:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return unique


def get_file_imports(tree, source_bytes: bytes, language: str) -> list[str]:
    """Collect all top-level import statements in the file."""
    import_types = IMPORT_NODE_TYPES.get(language, set())
    imports = []
    for node in tree.root_node.children:
        if node.type in import_types:
            imports.append(
                source_bytes[node.start_byte:node.end_byte].decode("utf-8").strip()
            )
    return imports


# ---------------------------------------------------------------------------
# Pass 1 — parse a single file, extract chunks with calls
# ---------------------------------------------------------------------------

def parse_file(file_path: str, content: str, language: str) -> list[dict]:
    """
    Parse one source file with Tree-sitter and return a list of chunks.

    Each chunk contains:
        chunk_id     — unique key  "<file_path>::<name>::<start_line>"
        file_path    — relative path to the file
        language     — python | javascript | typescript
        chunk_type   — function | class | file
        name         — function or class name
        start_line   — 1-indexed start line
        end_line     — 1-indexed end line
        code         — raw source text of this chunk
        context      — human-readable label used for embedding metadata
        parent_class — name of enclosing class if this is a method, else None
        signature    — parameter list string e.g. "(self, user_id, cart_id)"
        docstring    — extracted docstring or leading comment, or None
        imports      — file-level import lines shared across all chunks in file
        calls        — list of function/method names called inside this chunk
        called_by    — empty list here; populated in Pass 2 by build_called_by()
    """
    chunks = []

    try:
        parser = get_parser(language)
        source_bytes = content.encode("utf-8")
        tree = parser.parse(source_bytes)
    except Exception as e:
        logger.warning(f"Tree-sitter parse failed for {file_path}: {e}")
        return [make_fallback_chunk(file_path, content, language)]

    fn_types  = FUNCTION_NODE_TYPES.get(language, set())
    cls_types = CLASS_NODE_TYPES.get(language, set())
    file_imports = get_file_imports(tree, source_bytes, language)

    def visit(node):
        if node.type in fn_types:
            name       = extract_node_name(node, source_bytes, language)
            code       = source_bytes[node.start_byte:node.end_byte].decode("utf-8")
            start_line = node.start_point[0] + 1
            end_line   = node.end_point[0] + 1

            chunks.append({
                "chunk_id":     f"{file_path}::{name}::{start_line}",
                "file_path":    file_path,
                "language":     language,
                "chunk_type":   "function",
                "name":         name or "anonymous",
                "start_line":   start_line,
                "end_line":     end_line,
                "code":         code,
                "context":      f"{file_path} → {name}()",
                "parent_class": get_parent_class(node, source_bytes, language),
                "signature":    get_signature(node, source_bytes),
                "docstring":    get_docstring(node, source_bytes, language),
                "imports":      file_imports,
                "calls":        get_calls(node, source_bytes),
                "called_by":    [],  # populated in Pass 2
            })

        elif node.type in cls_types:
            name       = extract_node_name(node, source_bytes, language)
            code       = source_bytes[node.start_byte:node.end_byte].decode("utf-8")
            start_line = node.start_point[0] + 1
            end_line   = node.end_point[0] + 1

            chunks.append({
                "chunk_id":     f"{file_path}::{name}::{start_line}",
                "file_path":    file_path,
                "language":     language,
                "chunk_type":   "class",
                "name":         name or "anonymous",
                "start_line":   start_line,
                "end_line":     end_line,
                "code":         code,
                "context":      f"{file_path} → class {name}",
                "parent_class": None,
                "signature":    None,
                "docstring":    get_docstring(node, source_bytes, language),
                "imports":      file_imports,
                "calls":        get_calls(node, source_bytes),
                "called_by":    [],  # populated in Pass 2
            })

        for child in node.children:
            visit(child)

    visit(tree.root_node)

    if not chunks:
        return [make_fallback_chunk(file_path, content, language)]

    return chunks


# ---------------------------------------------------------------------------
# Pass 2 — invert the call graph to populate called_by on every chunk
# ---------------------------------------------------------------------------

def build_called_by(all_chunks: list[dict]) -> list[dict]:
    """
    Given ALL chunks across the entire repo, invert the calls graph so
    each chunk knows which other chunks call it.

    Modifies chunks in-place and returns the same list.

    How it works:
        1. Build  name → [chunks]  index from all chunks.
        2. For each chunk C and each callee name N in C["calls"]:
               - Try to resolve N to an internal chunk.
               - If found, append C["name"] to that chunk's called_by list.
               - If not found (library call, builtin), silently skip.

    Dotted names like "db.find_one" are tried both as-is AND by their
    last segment ("find_one"), so method calls on well-known internal
    objects still resolve correctly when the method name is unique.
    """
    # name → list of chunks  (list handles rare duplicate names across files)
    name_to_chunks: dict[str, list[dict]] = {}
    for chunk in all_chunks:
        name = chunk["name"]
        if name not in name_to_chunks:
            name_to_chunks[name] = []
        name_to_chunks[name].append(chunk)

    for chunk in all_chunks:
        caller_name = chunk["name"]

        for callee_name in chunk["calls"]:
            # build a set of candidate names to try resolving
            # "Cart.get"   → try "Cart.get" and "get"
            # "get_user"   → try "get_user"
            candidates: set[str] = {callee_name}
            if "." in callee_name:
                candidates.add(callee_name.split(".")[-1])

            for candidate in candidates:
                if candidate in name_to_chunks:
                    for callee_chunk in name_to_chunks[candidate]:
                        if caller_name not in callee_chunk["called_by"]:
                            callee_chunk["called_by"].append(caller_name)

    return all_chunks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_fallback_chunk(file_path: str, content: str, language: str) -> dict:
    """Fallback: treat the entire file as one chunk when parsing fails."""
    lines = content.splitlines()
    return {
        "chunk_id":     f"{file_path}::__file__::1",
        "file_path":    file_path,
        "language":     language,
        "chunk_type":   "file",
        "name":         "__file__",
        "start_line":   1,
        "end_line":     len(lines),
        "code":         content,
        "context":      f"{file_path} (full file)",
        "parent_class": None,
        "signature":    None,
        "docstring":    None,
        "imports":      [],
        "calls":        [],
        "called_by":    [],
    }


def find_chunk_at_line(chunks: list[dict], target_line: int) -> Optional[dict]:
    """
    Given a line number from a stack trace, return the chunk that contains it.
    """
    for chunk in chunks:
        if chunk["start_line"] <= target_line <= chunk["end_line"]:
            return chunk
    return None