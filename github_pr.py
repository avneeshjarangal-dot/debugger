import httpx


def _headers(github_token: str) -> dict:
    return {
        "Authorization": f"Bearer {github_token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _format_pr(data: dict, *, existing: bool = False) -> dict:
    return {
        "ok": True,
        "number": data.get("number"),
        "url": data.get("html_url"),
        "title": data.get("title"),
        "existing": existing,
    }


def _find_existing_pull_request(*, github_token: str, repo_full_name: str, head_branch: str, base_branch: str) -> dict | None:
    owner = repo_full_name.split("/", 1)[0]
    response = httpx.get(
        f"https://api.github.com/repos/{repo_full_name}/pulls",
        headers=_headers(github_token),
        params={
            "state": "open",
            "head": f"{owner}:{head_branch}",
            "base": base_branch,
        },
        timeout=30.0,
    )
    response.raise_for_status()
    pulls = response.json()
    if pulls:
        return _format_pr(pulls[0], existing=True)
    return None


def create_pull_request(*, github_token: str, repo_full_name: str, head_branch: str, base_branch: str, title: str, body: str) -> dict:
    existing = _find_existing_pull_request(
        github_token=github_token,
        repo_full_name=repo_full_name,
        head_branch=head_branch,
        base_branch=base_branch,
    )
    if existing:
        return existing

    url = f"https://api.github.com/repos/{repo_full_name}/pulls"
    payload = {
        "title": title,
        "head": head_branch,
        "base": base_branch,
        "body": body,
    }
    response = httpx.post(url, headers=_headers(github_token), json=payload, timeout=30.0)
    if response.status_code == 422:
        existing = _find_existing_pull_request(
            github_token=github_token,
            repo_full_name=repo_full_name,
            head_branch=head_branch,
            base_branch=base_branch,
        )
        if existing:
            return existing
    if response.status_code >= 400:
        raise RuntimeError(response.text)
    data = response.json()
    return _format_pr(data)
