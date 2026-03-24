import os
from dataclasses import dataclass

from dotenv import load_dotenv

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_BASE_DIR, ".env"), override=False)


def _clean_env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name, default)
    if value is None:
        return None
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        value = value[1:-1]
    return value


def require_env(name: str) -> str:
    value = _clean_env(name)
    if not value:
        raise RuntimeError(f"{name} is not set.")
    return value


@dataclass(frozen=True)
class DebuggerConfig:
    github_token: str | None
    mongo_uri: str | None
    mongo_db: str
    mongo_collection: str
    repo_full_name: str | None
    chroma_dir: str | None
    clone_base_dir: str | None
    gemini_api_key: str | None
    openai_api_key: str | None
    openrouter_api_key: str | None
    groq_api_key: str | None
    anthropic_api_key: str | None
    allow_external_ai: bool
    allow_frontend_secrets: bool


def load_config() -> DebuggerConfig:
    return DebuggerConfig(
        github_token=_clean_env("GITHUB_TOKEN"),
        mongo_uri=_clean_env("MONGO_URI"),
        mongo_db=_clean_env("MONGO_DB", "log") or "log",
        mongo_collection=_clean_env("MONGO_COLLECTION", "logs") or "logs",
        repo_full_name=_clean_env("REPO_FULL_NAME"),
        chroma_dir=_clean_env("CHROMA_DIR"),
        clone_base_dir=_clean_env("CLONE_BASE_DIR"),
        gemini_api_key=_clean_env("GEMINI_API_KEY"),
        openai_api_key=_clean_env("OPENAI_API_KEY") or _clean_env("OPEN_AI_API_KEY"),
        openrouter_api_key=_clean_env("OPENROUTER_API_KEY"),
        groq_api_key=_clean_env("GROQ_API_KEY"),
        anthropic_api_key=_clean_env("ANTHROPIC_API_KEY"),
        allow_external_ai=(_clean_env("ALLOW_EXTERNAL_AI", "false") or "false").lower() in {"1", "true", "yes"},
        allow_frontend_secrets=(_clean_env("ALLOW_FRONTEND_SECRETS", "false") or "false").lower() in {"1", "true", "yes"},
    )
