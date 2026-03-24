import os
import logging
from pathlib import Path
from typing import Generator

logger = logging.getLogger(__name__)
 
# file extensions we care about — expand this as needed
SUPPORTED_EXTENSIONS = {
    ".py",
    ".js",
    ".ts",
    ".jsx",
    ".tsx",
}

# directories to always skip
SKIP_DIRS = {
    "node_modules",
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "env",
    "dist",
    "build",
    ".next",
    "coverage",
    ".pytest_cache",
    "migrations",
}

# files to always skip
SKIP_FILES = {
    "package-lock.json",
    "yarn.lock",
    "poetry.lock",
    ".DS_Store",
}

# max file size to index — skip huge generated files
MAX_FILE_SIZE_BYTES = 200 * 1024  # 200 KB


def should_skip_dir(dir_name: str) -> bool:
    return dir_name in SKIP_DIRS or dir_name.startswith(".")


def should_skip_file(file_path: str) -> bool:
    file_name = os.path.basename(file_path)
    if file_name in SKIP_FILES:
        return True
    if os.path.getsize(file_path) > MAX_FILE_SIZE_BYTES:
        logger.debug(f"Skipping large file: {file_path}")
        return True
    return False


def walk_repo(repo_path: str) -> Generator[dict, None, None]:
    """
    Walk a cloned repo directory and yield metadata for each
    indexable source file.

    Yields dicts with:
        path         — absolute path to the file
        relative_path — path relative to repo root
        extension    — file extension e.g. '.py'
        language     — 'python' | 'javascript' | 'typescript'
        size_bytes   — file size
    """
    repo_path = os.path.abspath(repo_path)

    for root, dirs, files in os.walk(repo_path):
        # prune skip dirs in-place so os.walk doesn't descend into them
        dirs[:] = [d for d in dirs if not should_skip_dir(d)]

        for file_name in files:
            file_path = os.path.join(root, file_name)
            ext = Path(file_name).suffix.lower()

            if ext not in SUPPORTED_EXTENSIONS:
                continue

            if should_skip_file(file_path):
                continue

            relative_path = os.path.relpath(file_path, repo_path)

            yield {
                "path": file_path,
                "relative_path": relative_path,
                "extension": ext,
                "language": get_language(ext),
                "size_bytes": os.path.getsize(file_path),
            }


def get_language(extension: str) -> str:
    mapping = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".jsx": "javascript",
        ".tsx": "typescript",
    }
    return mapping.get(extension, "unknown")


def get_file_content(file_path: str) -> str:
    """Read file content, handling encoding issues gracefully."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        # fallback for files with mixed encoding
        with open(file_path, "r", encoding="latin-1") as f:
            return f.read()


def count_files(repo_path: str) -> dict:
    """Count indexable files per language for logging."""
    counts = {}
    for file_info in walk_repo(repo_path):
        lang = file_info["language"]
        counts[lang] = counts.get(lang, 0) + 1
    return counts
