import os
import shutil
import logging
from urllib.parse import urlparse
import subprocess

logger = logging.getLogger(__name__)

# walk up: ingestion/ -> core/ -> debugger/
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CLONE_BASE_DIR = os.environ.get("CLONE_BASE_DIR", os.path.join(_BASE_DIR, "debugger_repos"))


def parse_repo_url(github_url: str) -> dict:
    """
    Parse a GitHub URL into its components.
    Accepts:
      - https://github.com/owner/repo
      - https://github.com/owner/repo.git
      - github.com/owner/repo
    """
    if not github_url.startswith("http"):
        github_url = "https://" + github_url

    parsed = urlparse(github_url)
    path_parts = parsed.path.strip("/").replace(".git", "").split("/")

    if len(path_parts) < 2:
        raise ValueError(f"Invalid GitHub URL: {github_url}")

    owner = path_parts[0]
    repo  = path_parts[1]

    return {
        "owner":     owner,
        "repo":      repo,
        "full_name": f"{owner}/{repo}",
        "clone_url": f"https://github.com/{owner}/{repo}.git",
    }


def get_local_path(repo_full_name: str) -> str:
    """Return the local directory path where this repo will be cloned."""
    safe_name = repo_full_name.replace("/", "__")
    return os.path.join(CLONE_BASE_DIR, safe_name)


def clone_repo(github_url: str, github_token: str, force: bool = False) -> str:
    """
    Clone a GitHub repo to local disk using a Personal Access Token.

    Args:
        github_url:   Full GitHub URL e.g. https://github.com/owner/repo
        github_token: GitHub PAT for private repo access
        force:        If True, delete and re-clone even if already present

    Returns:
        Local path where the repo was cloned
    """
    repo_info  = parse_repo_url(github_url)
    local_path = get_local_path(repo_info["full_name"])

    if os.path.exists(local_path):
        if force:
            logger.info(f"Force re-clone: removing {local_path}")
            shutil.rmtree(local_path)
        else:
            logger.info(f"Repo already cloned at {local_path} — skipping clone")
            return local_path

    os.makedirs(CLONE_BASE_DIR, exist_ok=True)

    # Inject PAT into the clone URL for authentication.
    # Format: https://<token>@github.com/owner/repo.git
    # NOTE: auth_url is used in the subprocess call (not the bare clone_url).
    auth_url = repo_info["clone_url"].replace(
        "https://", f"https://{github_token}@"
    )

    logger.info(f"Cloning {repo_info['full_name']} into {local_path} ...")

    result = subprocess.run(
        ["git", "clone", "--depth=1", auth_url, local_path],  # ← auth_url, not clone_url
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        # Scrub token from error message before logging
        error_msg = result.stderr.replace(github_token, "***")
        raise RuntimeError(f"Git clone failed:\n{error_msg}")

    logger.info(f"Clone complete → {local_path}")
    return local_path


def delete_repo(github_url: str):
    """Remove a cloned repo from disk."""
    repo_info  = parse_repo_url(github_url)
    local_path = get_local_path(repo_info["full_name"])

    if os.path.exists(local_path):
        shutil.rmtree(local_path)
        logger.info(f"Deleted repo at {local_path}")