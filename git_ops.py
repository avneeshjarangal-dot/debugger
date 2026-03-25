import re
import subprocess
from pathlib import Path

from core.ingestion.repo_cloner import get_local_path


def _repo_root(repo_full_name: str) -> Path:
    repo_root = Path(get_local_path(repo_full_name)).resolve()
    if not repo_root.exists():
        raise RuntimeError(f"Cloned repo not found: {repo_root}")
    return repo_root


def _run_git(repo_root: Path, args: list[str]) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout).strip() or f"git {' '.join(args)} failed")
    return result.stdout.strip()


def _sanitize_branch_name(issue_id: str, summary: str | None = None) -> str:
    raw = summary or issue_id
    slug = re.sub(r"[^a-zA-Z0-9._/-]+", "-", raw).strip("-./").lower()
    slug = re.sub(r"-+", "-", slug)[:40] or issue_id
    return f"proctor/{issue_id}-{slug}"[:80]

def _detect_default_branch(repo_root: Path) -> str:
    try:
        head_ref = _run_git(repo_root, ["symbolic-ref", "--short", "refs/remotes/origin/HEAD"])
        if head_ref.startswith("origin/"):
            return head_ref.split("/", 1)[1]
    except Exception:
        pass

    for candidate in ("main", "master"):
        try:
            _run_git(repo_root, ["rev-parse", "--verify", f"origin/{candidate}"])
            return candidate
        except Exception:
            continue

    raise RuntimeError("Unable to determine repository default branch")


def ensure_clean_base_checkout(repo_full_name: str) -> dict:
    repo_root = _repo_root(repo_full_name)
    _run_git(repo_root, ["fetch", "origin", "--prune"])
    base_branch = _detect_default_branch(repo_root)
    _run_git(repo_root, ["checkout", "-B", base_branch, f"origin/{base_branch}"])
    _run_git(repo_root, ["reset", "--hard", f"origin/{base_branch}"])
    _run_git(repo_root, ["clean", "-fd"])
    return {
        "ok": True,
        "repo_root": str(repo_root),
        "base_branch": base_branch,
    }


def create_branch_commit_push(repo_full_name: str, issue_id: str, summary: str | None = None) -> dict:
    repo_root = _repo_root(repo_full_name)
    branch_name = _sanitize_branch_name(issue_id, summary)

    _run_git(repo_root, ["checkout", "-B", branch_name])

    status = _run_git(repo_root, ["status", "--porcelain"])
    if not status:
        raise RuntimeError("No local changes available to commit")

    _run_git(repo_root, ["add", "."])
    commit_message = f"Proctor fix for issue {issue_id}"
    _run_git(
        repo_root,
        [
            "-c", "user.name=Proctor",
            "-c", "user.email=proctor@local",
            "commit", "-m", commit_message,
        ],
    )

    push_args = ["push", "-u", "origin", branch_name]
    if branch_name.startswith("proctor/"):
        # Proctor owns these automation branches. If the branch already exists on the
        # remote, replace it; otherwise allow the first push to create it normally.
        try:
            _run_git(repo_root, ["fetch", "origin", branch_name])
            push_args.insert(1, "--force")
        except RuntimeError as exc:
            if "couldn't find remote ref" not in str(exc):
                raise
    _run_git(repo_root, push_args)

    commit_sha = _run_git(repo_root, ["rev-parse", "HEAD"])
    return {
        "ok": True,
        "repo_root": str(repo_root),
        "branch_name": branch_name,
        "commit_sha": commit_sha,
        "commit_message": commit_message,
    }
