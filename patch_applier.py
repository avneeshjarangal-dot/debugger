import difflib
from pathlib import Path, PurePosixPath

from core.ingestion.repo_cloner import get_local_path


class PatchApplicationError(Exception):
    pass


def _resolve_repo_root(repo_full_name: str) -> Path:
    repo_root = Path(get_local_path(repo_full_name)).resolve()
    if not repo_root.exists():
        raise PatchApplicationError(f"Cloned repo not found: {repo_root}")
    return repo_root


def _normalize_path_variants(relative_path: str, repo_root: Path) -> list[str]:
    raw = str(relative_path or "").strip().replace("\\", "/")
    if not raw:
        return []

    parts = [part for part in PurePosixPath(raw).parts if part not in {"/", "."}]
    if not parts:
        return []

    variants: list[str] = []

    def add_variant(candidate_parts: list[str]):
        cleaned = [part for part in candidate_parts if part and part not in {".", "/"}]
        if not cleaned:
            return
        candidate = "/".join(cleaned)
        if candidate not in variants:
            variants.append(candidate)

    add_variant(parts)

    if parts and parts[0] == "app":
        add_variant(parts[1:])

    repo_name = repo_root.name
    for index, part in enumerate(parts):
        if part == repo_name:
            add_variant(parts[index:])
            add_variant(parts[index + 1 :])

    repo_variants = {repo_name, repo_name.replace("-", "_"), repo_name.replace("_", "-")}
    for index, part in enumerate(parts):
        if part in repo_variants:
            add_variant([repo_name, *parts[index + 1 :]])
            add_variant(parts[index + 1 :])

    expanded: list[str] = []
    for candidate in list(variants):
        candidate_parts = candidate.split("/")
        expanded_parts = []
        for part in candidate_parts:
            expanded_parts.append([part])
            if "_" in part:
                expanded_parts[-1].append(part.replace("_", "-"))
            if "-" in part:
                expanded_parts[-1].append(part.replace("-", "_"))

        built = [""]
        for options in expanded_parts:
            next_built = []
            for prefix in built:
                for option in dict.fromkeys(options):
                    next_built.append(f"{prefix}/{option}" if prefix else option)
            built = next_built
        for item in built:
            if item not in expanded:
                expanded.append(item)

    return expanded or variants


def _safe_target_path(repo_root: Path, relative_path: str) -> Path:
    variants = _normalize_path_variants(relative_path, repo_root)

    for variant in variants:
        candidate = (repo_root / variant).resolve()
        if repo_root not in [candidate, *candidate.parents]:
            continue
        if candidate.exists():
            return candidate

    variant_tails = [variant.split("/") for variant in variants if variant]
    for file_path in repo_root.rglob("*"):
        if not file_path.is_file():
            continue
        relative_parts = file_path.relative_to(repo_root).as_posix().split("/")
        for tail in variant_tails:
            if len(tail) <= len(relative_parts) and relative_parts[-len(tail):] == tail:
                return file_path.resolve()

    raise PatchApplicationError(f"target_file_not_found:{relative_path}")




def _extract_python_symbol_name(target: str, before: str, after: str) -> str | None:
    candidates = [after, before]
    for snippet in candidates:
        for raw_line in str(snippet or '').splitlines():
            line = raw_line.strip()
            if line.startswith('async def '):
                return line[len('async def '):].split('(', 1)[0].strip() or None
            if line.startswith('def '):
                return line[len('def '):].split('(', 1)[0].strip() or None
            if line.startswith('class '):
                return line[len('class '):].split('(', 1)[0].split(':', 1)[0].strip() or None

    raw_target = str(target or '').strip()
    for suffix in (' function', ' method', ' class', ' block'):
        if raw_target.lower().endswith(suffix):
            raw_target = raw_target[: -len(suffix)].strip()
    if raw_target:
        symbol = raw_target.split()[-1].strip()
        return symbol or None
    return None


def _find_python_block(original: str, symbol_name: str) -> tuple[int, int] | None:
    lines = original.splitlines(keepends=True)
    patterns = (f'async def {symbol_name}(', f'def {symbol_name}(', f'class {symbol_name}(' , f'class {symbol_name}:')

    start_index = None
    indent = 0
    for index, line in enumerate(lines):
        stripped = line.lstrip()
        if any(stripped.startswith(pattern) for pattern in patterns):
            start_index = index
            indent = len(line) - len(stripped)
            break

    if start_index is None:
        return None

    end_index = len(lines)
    for index in range(start_index + 1, len(lines)):
        stripped = lines[index].lstrip()
        if not stripped.strip():
            continue
        current_indent = len(lines[index]) - len(stripped)
        if current_indent <= indent and not stripped.startswith(('#', '@')):
            end_index = index
            break

    start_offset = sum(len(line) for line in lines[:start_index])
    end_offset = sum(len(line) for line in lines[:end_index])
    return start_offset, end_offset


def _replace_with_python_target(original: str, target: str, before: str, after: str, relative_path: str) -> tuple[str, str] | None:
    if not relative_path.endswith('.py'):
        return None
    if not after.strip():
        return None

    symbol_name = _extract_python_symbol_name(target, before, after)
    if not symbol_name:
        return None

    block_range = _find_python_block(original, symbol_name)
    if not block_range:
        return None

    start_offset, end_offset = block_range
    existing_block = original[start_offset:end_offset]

    normalized_before = ''.join(str(before or '').split())
    normalized_existing = ''.join(existing_block.split())
    normalized_after = ''.join(str(after or '').split())
    if normalized_before and normalized_before not in normalized_existing and normalized_existing == normalized_after:
        return None

    replacement = after
    if not replacement.endswith('\n'):
        replacement += '\n'

    updated = original[:start_offset] + replacement + original[end_offset:]
    return updated, 'python_target_match'



def _extract_js_symbol_name(target: str, before: str, after: str) -> str | None:
    import re

    candidates = [str(after or ''), str(before or '')]
    patterns = (
        r"function\s+([A-Za-z_$][\w$]*)\s*\(",
        r"(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s+)?function\s*\(",
        r"(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s+)?\([^)]*\)\s*=>",
    )
    for snippet in candidates:
        for pattern in patterns:
            match = re.search(pattern, snippet)
            if match:
                return match.group(1)

    raw_target = str(target or '').strip()
    for suffix in (' function', ' method', ' handler', ' middleware', ' block'):
        if raw_target.lower().endswith(suffix):
            raw_target = raw_target[: -len(suffix)].strip()
    if raw_target:
        symbol = raw_target.split()[-1].strip()
        return symbol or None
    return None


def _find_js_block(original: str, symbol_name: str) -> tuple[int, int] | None:
    import re

    patterns = (
        re.compile(rf"(^|\n)[ \t]*function\s+{re.escape(symbol_name)}\s*\(", re.MULTILINE),
        re.compile(rf"(^|\n)[ \t]*(?:const|let|var)\s+{re.escape(symbol_name)}\s*=\s*(?:async\s+)?function\s*\(", re.MULTILINE),
        re.compile(rf"(^|\n)[ \t]*(?:const|let|var)\s+{re.escape(symbol_name)}\s*=\s*(?:async\s+)?\([^)]*\)\s*=>", re.MULTILINE),
    )

    match = None
    for pattern in patterns:
        match = pattern.search(original)
        if match:
            break
    if not match:
        return None

    start_offset = match.start()
    open_brace = original.find('{', match.end())
    if open_brace == -1:
        return None

    depth = 0
    in_single = False
    in_double = False
    in_template = False
    escaped = False
    i = open_brace
    while i < len(original):
        ch = original[i]
        if escaped:
            escaped = False
        elif ch == '\\':
            escaped = True
        elif in_single:
            if ch == "'":
                in_single = False
        elif in_double:
            if ch == '"':
                in_double = False
        elif in_template:
            if ch == '`':
                in_template = False
        else:
            if ch == "'":
                in_single = True
            elif ch == '"':
                in_double = True
            elif ch == '`':
                in_template = True
            elif ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    end_offset = i + 1
                    while end_offset < len(original) and original[end_offset] in '\r\n; ':
                        end_offset += 1
                    return start_offset, end_offset
        i += 1
    return None


def _replace_with_js_target(original: str, target: str, before: str, after: str, relative_path: str) -> tuple[str, str] | None:
    if not relative_path.endswith(('.js', '.jsx', '.ts', '.tsx', '.mjs', '.cjs')):
        return None
    if not after.strip():
        return None

    symbol_name = _extract_js_symbol_name(target, before, after)
    if not symbol_name:
        return None

    block_range = _find_js_block(original, symbol_name)
    if not block_range:
        return None

    start_offset, end_offset = block_range
    existing_block = original[start_offset:end_offset]

    normalized_before = ''.join(str(before or '').split())
    normalized_existing = ''.join(existing_block.split())
    normalized_after = ''.join(str(after or '').split())
    if normalized_before and normalized_before not in normalized_existing and normalized_existing == normalized_after:
        return None

    replacement = after
    if not replacement.endswith('\n'):
        replacement += '\n'

    updated = original[:start_offset] + replacement + original[end_offset:]
    return updated, 'js_target_match'


def _prepare_file_update(repo_root: Path, entry: dict) -> dict:
    rel_path = str(entry.get("path") or "").strip()
    action = str(entry.get("action") or "update").strip().lower()
    before = entry.get("before") or ""
    after = entry.get("after") or ""
    target = entry.get("target") or ""

    file_result = {
        "path": rel_path,
        "target": target,
        "action": action,
        "status": "pending",
        "reason": None,
        "diff": "",
        "matched": False,
        "written": False,
    }

    if not rel_path:
        raise PatchApplicationError("missing_file_path")
    if action != "update":
        raise PatchApplicationError(f"unsupported_action:{action}")

    target_path = _safe_target_path(repo_root, rel_path)
    original = target_path.read_text(encoding="utf-8")
    if not before:
        raise PatchApplicationError("missing_before_snippet")
    if not after:
        raise PatchApplicationError("missing_after_snippet")

    match_strategy = "exact_snippet"
    if before in original:
        updated = original.replace(before, after, 1)
    else:
        relative_path = target_path.relative_to(repo_root).as_posix()
        fallback = _replace_with_python_target(
            original,
            target,
            before,
            after,
            relative_path,
        )
        if not fallback:
            fallback = _replace_with_js_target(
                original,
                target,
                before,
                after,
                relative_path,
            )
        if not fallback:
            raise PatchApplicationError("before_snippet_not_found")
        updated, match_strategy = fallback
    diff = "".join(
        difflib.unified_diff(
            original.splitlines(keepends=True),
            updated.splitlines(keepends=True),
            fromfile=target_path.relative_to(repo_root).as_posix(),
            tofile=target_path.relative_to(repo_root).as_posix(),
        )
    )
    file_result.update({
        "path": target_path.relative_to(repo_root).as_posix(),
        "status": "ready",
        "matched": True,
        "reason": match_strategy,
        "diff": diff,
    })
    return {
        "target_path": target_path,
        "updated": updated,
        "result": file_result,
    }


def build_dry_run_patch(repo_full_name: str, patch_plan: dict | None) -> dict:
    if not patch_plan:
        return {"ok": False, "reason": "missing_patch_plan", "files": []}

    repo_root = _resolve_repo_root(repo_full_name)
    files = patch_plan.get("files") or []
    results = []
    applied_count = 0

    for entry in files:
        try:
            prepared = _prepare_file_update(repo_root, entry)
            results.append(prepared["result"])
            applied_count += 1
        except Exception as exc:
            results.append({
                "path": str(entry.get("path") or "").strip(),
                "target": entry.get("target") or "",
                "action": str(entry.get("action") or "update").strip().lower(),
                "status": "manual_review",
                "reason": str(exc),
                "diff": "",
                "matched": False,
                "written": False,
            })

    return {
        "ok": applied_count > 0 and all(item["status"] == "ready" for item in results),
        "repo_root": str(repo_root),
        "files": results,
        "applied_count": applied_count,
        "total_files": len(results),
    }


def apply_patch_plan(repo_full_name: str, patch_plan: dict | None) -> dict:
    if not patch_plan:
        raise PatchApplicationError("missing_patch_plan")

    repo_root = _resolve_repo_root(repo_full_name)
    files = patch_plan.get("files") or []
    prepared_updates = []
    failures = []

    for entry in files:
        try:
            prepared_updates.append(_prepare_file_update(repo_root, entry))
        except Exception as exc:
            failures.append({
                "path": str(entry.get("path") or "").strip(),
                "reason": str(exc),
            })

    if failures:
        return {
            "ok": False,
            "repo_root": str(repo_root),
            "applied_count": 0,
            "total_files": len(files),
            "files": failures,
        }

    written_files = []
    for prepared in prepared_updates:
        prepared["target_path"].write_text(prepared["updated"], encoding="utf-8")
        result = dict(prepared["result"])
        result["written"] = True
        written_files.append(result)

    return {
        "ok": True,
        "repo_root": str(repo_root),
        "applied_count": len(written_files),
        "total_files": len(written_files),
        "files": written_files,
    }
