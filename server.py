"""
server.py  —  FastAPI wrapper that orchestrates the 4-step debugger pipeline.

Exposes:
    POST /api/analysis          Run full pipeline (ingest → logs → prompts → fix)
    GET  /api/analysis/status   Check status of a running analysis
    GET  /api/logs              Fetch + group logs directly (fast, no AI)
    GET  /health                Health check

Env vars required on the server (NOT sent from frontend):
    GEMINI_API_KEY   AIza...

Run with:
    uvicorn server:app --host 0.0.0.0 --port 8000 --reload

All four scripts (ingest.py, fetch_logs.py, generate_prompts.py, apply_fix.py)
must live in the same directory as this file.
"""

import os
import re
import sys
import uuid
import asyncio
import logging
from datetime import datetime, timezone
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import load_config, require_env

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("server")

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Debugger API",
    description="AI-powered autonomous bug detection and fix pipeline",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in prod
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# In-memory job store  (replace with Redis/DB for multi-process deployments)
# ---------------------------------------------------------------------------
jobs: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------
class AnalysisRequest(BaseModel):
    github_url:       str = Field(..., example="https://github.com/owner/repo")
    github_token:     Optional[str] = Field(None, repr=False)
    mongo_uri:        Optional[str] = Field(None, repr=False)
    mongo_db:         Optional[str] = Field(None)
    mongo_collection: Optional[str] = Field(None)
    service:          Optional[str] = Field(None, description="Optionally scope analysis to one service")
    llm_provider:     Literal["GEMINI", "OPEN_AI", "OPENROUTER", "GROQ"] = Field("GEMINI", description="Default LLM provider for on-demand root cause and fixes")
    log_limit:        int = Field(200, ge=1, le=500, description="Maximum raw logs to fetch")
    cubeapm_url:      Optional[str] = Field(None)
    cubeapm_key:      Optional[str] = Field(None)
    hours:            int  = Field(120,   ge=1, le=720, description="Log window in hours")
    top:              int  = Field(5,     ge=1, le=20,  description="Top N grouped issues to display")
    force_ingest:     bool = Field(False, description="Re-clone and re-index even if cached")


class JobStatus(BaseModel):
    job_id:      str
    analysis_id: str          # alias of job_id — used by frontend
    status:      str          # queued | running | done | failed
    step:        Optional[str]
    started_at:  Optional[str]
    ended_at:    Optional[str]
    result:      Optional[dict]
    error:       Optional[str]


class ApplyLocalPatchRequest(BaseModel):
    job_id: str
    issue_id: str


class GenerateIssueFixRequest(BaseModel):
    job_id: str
    issue_id: str
    llm_provider: Optional[Literal["GEMINI", "OPEN_AI", "OPENROUTER", "GROQ"]] = Field(None)


class CreatePullRequestRequest(BaseModel):
    job_id: str
    issue_id: str
    base_branch: str = Field("main")
    title: Optional[str] = Field(None)
    body: Optional[str] = Field(None)
    github_token: Optional[str] = Field(None, repr=False)


# ---------------------------------------------------------------------------
# Pipeline helpers — thin wrappers that import from your existing scripts
# ---------------------------------------------------------------------------
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _BASE_DIR)


def _resolved_repo_full_name(req: AnalysisRequest) -> str:
    cfg = load_config()
    if cfg.repo_full_name:
        return cfg.repo_full_name

    from core.ingestion.repo_cloner import parse_repo_url

    return parse_repo_url(req.github_url)["full_name"]


def _external_ai_enabled() -> bool:
    return load_config().allow_external_ai


def _resolved_github_token(req: AnalysisRequest) -> str:
    cfg = load_config()
    if cfg.allow_frontend_secrets and req.github_token:
        return req.github_token.strip()
    return require_env("GITHUB_TOKEN")


def _resolved_mongo_uri(req: AnalysisRequest) -> str:
    cfg = load_config()
    if cfg.allow_frontend_secrets and req.mongo_uri:
        return req.mongo_uri.strip()
    return require_env("MONGO_URI")


def _resolved_runtime_github_token(frontend_token: Optional[str] = None) -> str:
    cfg = load_config()
    if cfg.allow_frontend_secrets and frontend_token:
        return frontend_token.strip()
    return require_env("GITHUB_TOKEN")


def _build_pr_title(issue: dict) -> str:
    summary = (((issue.get("fix") or {}).get("summary")) or "").strip()
    return summary or f"Proctor fix for issue {issue.get('issue_id')}"


def _build_pr_body(issue: dict) -> str:
    fix = issue.get("fix") or {}
    root_cause = (fix.get("root_cause") or "Not provided").strip()
    what_changed = fix.get("what_changed") or []
    verification = fix.get("verification") or "Not provided"
    changed_lines = "\n".join(f"- {item}" for item in what_changed) if what_changed else "- Not provided"
    return (
        f"## Root Cause\n{root_cause}\n\n"
        f"## What Changed\n{changed_lines}\n\n"
        f"## Verification\n{verification}\n"
    )



_REDACTION_PATTERNS = [
    (re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"), "[REDACTED_EMAIL]"),
    (re.compile(r"\b(?:\d[ -]?){10,16}\b"), "[REDACTED_NUMBER]"),
    (re.compile(r"\b[0-9a-fA-F]{24,}\b"), "[REDACTED_HEX]"),
    (re.compile(r"(?i)(bearer|token|apikey|api_key|authorization)[=: ]+[^\s,;]+"), "[REDACTED_SECRET]"),
]


def _redact_value(value):
    if isinstance(value, str):
        redacted = value
        for pattern, replacement in _REDACTION_PATTERNS:
            redacted = pattern.sub(replacement, redacted)
        return redacted
    if isinstance(value, list):
        return [_redact_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _redact_value(val) for key, val in value.items()}
    return value


def _sanitize_groups_for_ai(groups: list[dict]) -> list[dict]:
    return _redact_value(groups)


def _patch_env(req: AnalysisRequest):
    """Inject server-side configuration into env so legacy scripts can read it."""
    cfg = load_config()
    os.environ["GITHUB_TOKEN"] = _resolved_github_token(req)
    os.environ["MONGO_URI"] = _resolved_mongo_uri(req)
    os.environ["MONGO_DB"] = req.mongo_db or cfg.mongo_db
    os.environ["MONGO_COLLECTION"] = req.mongo_collection or cfg.mongo_collection
    os.environ["REPO_FULL_NAME"] = _resolved_repo_full_name(req)
    if cfg.gemini_api_key:
        os.environ["GEMINI_API_KEY"] = cfg.gemini_api_key


# ── Step 1: Ingest ──────────────────────────────────────────────────────────
def run_ingest(req: AnalysisRequest) -> dict:
    from core.ingestion.indexer import index_repo, is_repo_indexed

    if not req.force_ingest and is_repo_indexed(req.github_url):
        logger.info("Repo already indexed — skipping ingest")
        return {"repo": _resolved_repo_full_name(req), "skipped": True, "reason": "already_indexed"}

    summary = index_repo(req.github_url, _resolved_github_token(req), force=req.force_ingest)
    return {
        "repo":                 _resolved_repo_full_name(req),
        "skipped":              False,
        "files_walked":         summary["total_files_walked"],
        "files_skipped":        summary["total_files_skipped"],
        "chunks_indexed":       summary["total_chunks_indexed"],
        "chunks_with_callers":  summary["chunks_with_callers"],
        "collection_name":      summary["collection_name"],
    }


# ── Step 2: Fetch + group logs ──────────────────────────────────────────────
def run_fetch_logs(req: AnalysisRequest) -> dict:
    from fetch_logs import fetch_logs, group_logs, DEFAULT_LEVELS

    raw_logs = fetch_logs(
        uri=_resolved_mongo_uri(req),
        db_name=req.mongo_db or load_config().mongo_db,
        collection_name=req.mongo_collection or load_config().mongo_collection,
        levels=DEFAULT_LEVELS,
        service=req.service,
        hours=req.hours,
        limit=req.log_limit,
    )

    groups = group_logs(raw_logs)

    # Serialise for JSON — convert sets / datetimes
    serialised_groups = []
    for g in groups:
        rep = g["representative"]
        import hashlib
        _sig = f"{g['service']}::{g['function']}::{g['message_sig']}"
        issue_id = hashlib.md5(_sig.encode()).hexdigest()[:12]
        serialised_groups.append({
            "issue_id":     issue_id,
            "service":      g["service"],
            "function":     g["function"],
            "level":        g["level"],
            "count":        g["count"],
            "message_sig":  g["message_sig"],
            "first_seen":   _fmt_ts(min(g["timestamps"], default=None)),
            "last_seen":    _fmt_ts(max(g["timestamps"], default=None)),
            "locations":    g["locations"],
            "extra_samples":g["extra_samples"],
            "representative": {
                "message":   rep.get("message"),
                "file":      rep.get("file"),
                "line":      rep.get("line"),
                "traceback": rep.get("traceback"),
            },
        })

    return {
        "total_raw_logs":    len(raw_logs),
        "total_groups":      len(groups),
        "groups":            serialised_groups,
    }


def _fmt_ts(ts) -> Optional[str]:
    if ts is None:
        return None
    if hasattr(ts, "isoformat"):
        return ts.replace(tzinfo=timezone.utc).isoformat()
    return str(ts)


# ── Step 3: Generate fix prompts ─────────────────────────────────────────────
def run_generate_prompts(req: AnalysisRequest, groups_raw: list) -> list[dict]:
    """
    Reimplements generate_prompts.main() inline so we can pass dynamic config
    (mongo_uri, api_key, etc.) without relying on env / argparse.
    """
    import generate_prompts as gp

    # Patch globals so the module picks up the correct Gemini key + Mongo URI
    gp.DEFAULT_MONGO_URI  = _resolved_mongo_uri(req)
    gp.DEFAULT_DB_NAME    = req.mongo_db
    gp.DEFAULT_COLLECTION = req.mongo_collection

    # Patch api_key used inside generate_fix_prompt
    import google.genai as _genai_module   # noqa: F401 — ensure importable

    # GEMINI_API_KEY already in env from _patch_env

    # Load ChromaDB collection
    collection = gp.get_collection(_resolved_repo_full_name(req))
    logger.info(f"ChromaDB ready — {collection.count()} chunks")

    from fetch_logs import group_logs
    # Re-group from the serialised groups (already done upstream, just re-use)
    # We need the original group dicts with timestamps as datetime objects,
    # so we re-fetch here (cheaply — results are cached in mongo cursor)
    from fetch_logs import fetch_logs, DEFAULT_LEVELS
    raw_logs = fetch_logs(
        uri=_resolved_mongo_uri(req),
        db_name=req.mongo_db or load_config().mongo_db,
        collection_name=req.mongo_collection or load_config().mongo_collection,
        levels=DEFAULT_LEVELS,
        service=req.service,
        hours=req.hours,
        limit=req.log_limit,
    )
    groups = group_logs(raw_logs)[: req.top]

    results = []
    for i, group in enumerate(groups, 1):
        logger.info(f"  Generating prompt {i}/{len(groups)}: {group['service']}::{group['function']}")

        chunks = gp.find_chunks_for_group(collection, group)

        if not chunks["primary"] and not chunks["semantic"]:
            logger.warning(f"  No chunks found for group {i} — skipping")
            results.append({
                "index":    i,
                "service":  group["service"],
                "function": group["function"],
                "skipped":  True,
                "reason":   "no_chunks_found",
            })
            continue

        context = gp.build_context(group, chunks)

        try:
            prompt_text = gp.generate_fix_prompt(context, group, provider=req.llm_provider)
        except Exception as e:
            logger.error(f"  Prompt generation failed: {e}")
            results.append({
                "index":    i,
                "service":  group["service"],
                "function": group["function"],
                "skipped":  True,
                "reason":   str(e),
            })
            continue

        # Save to disk (same as CLI behaviour)
        saved_path = gp.save_prompt(group, prompt_text, i)
        logger.info(f"  Saved prompt → {saved_path}")

        results.append({
            "index":        i,
            "service":      group["service"],
            "function":     group["function"],
            "level":        group["level"],
            "count":        group["count"],
            "llm_provider": req.llm_provider,
            "skipped":      False,
            "prompt_path":  saved_path,
            "prompt_text":  prompt_text,
            "chunks": {
                "primary_found":   chunks["primary"] is not None,
                "primary_file":    chunks["primary"]["metadata"].get("file_path") if chunks["primary"] else None,
                "callers_count":   len(chunks["callers"]),
                "semantic_count":  len(chunks["semantic"]),
            },
        })

    return results


# ── Step 4: Apply fixes ──────────────────────────────────────────────────────
def run_apply_fixes(prompt_results: list[dict]) -> list[dict]:
    from apply_fix import get_fixes, save_fixes

    fix_results = []
    for pr in prompt_results:
        if pr.get("skipped"):
            fix_results.append({**pr, "fix": None})
            continue

        prompt_text = pr["prompt_text"]
        prompt_path = pr["prompt_path"]

        provider = pr.get("llm_provider", "GEMINI")
        provider_key = provider.lower()
        logger.info(f"  Calling {provider} for fix on: {pr['service']}::{pr['function']}")
        fixes = get_fixes(prompt_text, provider=provider)

        saved = save_fixes(prompt_path, fixes)

        provider_result = fixes.get(provider_key, {})
        fix_results.append({
            **pr,
            "fix": {
                "ok":         provider_result.get("ok", False),
                "text":       provider_result.get("text"),
                "saved_path": saved.get(provider_key),
                "provider":   provider,
            },
        })

    return fix_results


def _issue_id_for_group(group: dict) -> str:
    import hashlib

    sig = f"{group['service']}::{group['function']}::{group['message_sig']}"
    return hashlib.md5(sig.encode()).hexdigest()[:12]


def _request_from_job(job: dict) -> AnalysisRequest:
    payload = (job.get("metadata") or {}).get("request_payload")
    if not payload:
        raise RuntimeError("Missing request payload for analysis job")
    return AnalysisRequest(**payload)


def _upsert_indexed_result(existing: list[dict], item: dict) -> list[dict]:
    filtered = [entry for entry in existing if entry.get("index") != item.get("index")]
    filtered.append(item)
    return sorted(filtered, key=lambda entry: entry.get("index", 0))


def run_generate_fix_for_issue(req: AnalysisRequest, issue_id: str) -> tuple[dict, dict]:
    if not _external_ai_enabled():
        raise RuntimeError("External AI calls are disabled. Set ALLOW_EXTERNAL_AI=true to enable on-demand fixes.")

    import generate_prompts as gp
    from fetch_logs import fetch_logs, group_logs, DEFAULT_LEVELS

    gp.DEFAULT_MONGO_URI = _resolved_mongo_uri(req)
    gp.DEFAULT_DB_NAME = req.mongo_db
    gp.DEFAULT_COLLECTION = req.mongo_collection

    collection = gp.get_collection(_resolved_repo_full_name(req))
    logger.info(f"ChromaDB ready — {collection.count()} chunks")

    raw_logs = fetch_logs(
        uri=_resolved_mongo_uri(req),
        db_name=req.mongo_db or load_config().mongo_db,
        collection_name=req.mongo_collection or load_config().mongo_collection,
        levels=DEFAULT_LEVELS,
        service=req.service,
        hours=req.hours,
        limit=req.log_limit,
    )
    groups = group_logs(raw_logs)

    selected_index = None
    selected_group = None
    for index, group in enumerate(groups, 1):
        if _issue_id_for_group(group) == issue_id:
            selected_index = index
            selected_group = group
            break

    if selected_group is None or selected_index is None:
        raise RuntimeError(f"Issue {issue_id} not found in grouped logs")

    chunks = gp.find_chunks_for_group(collection, selected_group)
    if not chunks["primary"] and not chunks["semantic"]:
        prompt_result = {
            "index": selected_index,
            "service": selected_group["service"],
            "function": selected_group["function"],
            "skipped": True,
            "reason": "no_chunks_found",
        }
        return prompt_result, {**prompt_result, "fix": None}

    context = gp.build_context(selected_group, chunks)
    prompt_text = gp.generate_fix_prompt(context, selected_group, provider=req.llm_provider)
    saved_path = gp.save_prompt(selected_group, prompt_text, selected_index)

    prompt_result = {
        "index": selected_index,
        "service": selected_group["service"],
        "function": selected_group["function"],
        "level": selected_group["level"],
        "count": selected_group["count"],
        "llm_provider": req.llm_provider,
        "skipped": False,
        "prompt_path": saved_path,
        "prompt_text": prompt_text,
        "chunks": {
            "primary_found": chunks["primary"] is not None,
            "primary_file": chunks["primary"]["metadata"].get("file_path") if chunks["primary"] else None,
            "callers_count": len(chunks["callers"]),
            "semantic_count": len(chunks["semantic"]),
        },
    }
    fix_result = run_apply_fixes([prompt_result])[0]
    return prompt_result, fix_result


# ---------------------------------------------------------------------------
# Background pipeline runner
# ---------------------------------------------------------------------------
async def run_pipeline(job_id: str, req: AnalysisRequest):
    job = jobs[job_id]
    job["status"]     = "running"
    job["started_at"] = datetime.now(tz=timezone.utc).isoformat()

    def update_step(step: str):
        job["step"] = step
        logger.info(f"[{job_id}] Step: {step}")

    try:
        _patch_env(req)

        # ── 1. Ingest ──
        update_step("ingest")
        ingest_result = await asyncio.get_event_loop().run_in_executor(
            None, run_ingest, req
        )
        job["result"] = {
            "ingest": ingest_result,
            "logs": {"total_raw_logs": 0, "total_groups": 0, "groups": []},
            "prompts": [],
            "fixes": [],
            "issues": [],
            "summary": _build_summary(ingest_result, {"total_raw_logs": 0, "total_groups": 0}, [], req),
        }

        # ── 2. Fetch logs ──
        update_step("fetch_logs")
        logs_result = await asyncio.get_event_loop().run_in_executor(
            None, run_fetch_logs, req
        )
        job["result"] = {
            "ingest": ingest_result,
            "logs": {
                "total_raw_logs": logs_result["total_raw_logs"],
                "total_groups": logs_result["total_groups"],
                "groups": logs_result["groups"],
            },
            "prompts": [],
            "fixes": [],
            "issues": _build_issues_map(logs_result["groups"], [], []),
            "summary": _build_summary(ingest_result, logs_result, [], req),
        }

        if logs_result["total_raw_logs"] == 0:
            job["status"]   = "done"
            job["ended_at"] = datetime.now(tz=timezone.utc).isoformat()
            job["result"]   = {
                "ingest":   ingest_result,
                "logs":     {
                    "total_raw_logs": logs_result["total_raw_logs"],
                    "total_groups": logs_result["total_groups"],
                    "groups": logs_result["groups"],
                },
                "prompts":  [],
                "fixes":    [],
                "issues":   [],
                "summary":  _build_summary(ingest_result, logs_result, [], req),
                "message":  "No logs found in the given window.",
            }
            return

        prompt_results = []
        fix_results = []
        logger.info(f"[{job_id}] Initial analysis completed without AI generation; issue-level fixes are generated on demand")

        # ── Done ──
        job["status"]   = "done"
        job["ended_at"] = datetime.now(tz=timezone.utc).isoformat()
        issues = _build_issues_map(
            logs_result["groups"],
            prompt_results,
            fix_results,
        )
        issues = _attach_dry_run_previews(issues, req)

        job["result"]   = {
            "ingest":  ingest_result,
            "logs": {
                "total_raw_logs": logs_result["total_raw_logs"],
                "total_groups":   logs_result["total_groups"],
                "groups":         logs_result["groups"],
            },
            "prompts": prompt_results,
            "fixes":   fix_results,
            "issues":  issues,          # ← log + prompt + fix zipped per issue
            "summary": _build_summary(ingest_result, logs_result, fix_results, req),
        }

    except Exception as exc:
        logger.exception(f"[{job_id}] Pipeline failed")
        job["status"]   = "failed"
        job["ended_at"] = datetime.now(tz=timezone.utc).isoformat()
        job["error"]    = str(exc)



def _normalize_repo_path(path_value: str | None) -> str | None:
    if not path_value:
        return None
    normalized = str(path_value).strip().replace('\\', '/')
    while normalized.startswith('/'):
        normalized = normalized[1:]
    for prefix in ('app/', 'workspace/', 'srv/app/'):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):]
    return normalized or None


def _path_tail(path_value: str | None) -> str:
    normalized = _normalize_repo_path(path_value) or ''
    return normalized.replace('-', '').replace('_', '').lower()


def _patch_plan_with_repo_paths(patch_plan: dict | None, chunk_info: dict | None, log_info: dict | None) -> dict | None:
    if not isinstance(patch_plan, dict):
        return patch_plan

    files = patch_plan.get('files')
    if not isinstance(files, list) or not files:
        return patch_plan

    canonical_path = None
    if isinstance(chunk_info, dict):
        canonical_path = chunk_info.get('primary_file')
    if not canonical_path and isinstance(log_info, dict):
        canonical_path = log_info.get('file')
    canonical_path = _normalize_repo_path(canonical_path)
    if not canonical_path:
        return patch_plan

    canonical_tail = _path_tail(canonical_path)
    updated_files = []
    for file_entry in files:
        if not isinstance(file_entry, dict):
            updated_files.append(file_entry)
            continue

        current_path = file_entry.get('path')
        normalized_current = _normalize_repo_path(current_path)
        current_tail = _path_tail(normalized_current)

        if len(files) == 1 or not normalized_current or current_tail == canonical_tail or canonical_tail.endswith(current_tail) or current_tail.endswith(canonical_tail):
            next_entry = {**file_entry, 'path': canonical_path}
        else:
            next_entry = {**file_entry, 'path': normalized_current}

        updated_files.append(next_entry)

    return {**patch_plan, 'files': updated_files}


def _attach_dry_run_previews(issues: list, req: AnalysisRequest) -> list:
    if not issues:
        return issues

    from patch_applier import build_dry_run_patch

    repo_full_name = _resolved_repo_full_name(req)
    for issue in issues:
        fix = issue.get("fix")
        if not fix:
            continue
        patch_plan = fix.get("patch_plan")
        try:
            fix["dry_run_patch"] = build_dry_run_patch(repo_full_name, patch_plan)
        except Exception as exc:
            fix["dry_run_patch"] = {
                "ok": False,
                "reason": str(exc),
                "files": [],
            }
    return issues


def _build_issues_map(groups: list, prompt_results: list, fix_results: list) -> list:
    """
    Zip log groups + prompts + fixes into one list, each item being a complete
    record of a single issue that the frontend can directly render.

    Shape of each item:
    {
        "issue_id":   str,          # stable 12-char key
        "index":      int,          # 1-based position
        "log": {                    # everything from the log group
            "service":    str,
            "function":   str,
            "level":      str,
            "count":      int,
            "message":    str,
            "file":       str,
            "line":       int | None,
            "traceback":  str | None,
            "first_seen": str | None,
            "last_seen":  str | None,
            "locations":  list,
            "extra_samples": list,
        },
        "prompt": {                 # the intermediate fix-prompt sent to Gemini
            "text":       str | None,
            "saved_path": str | None,
            "skipped":    bool,
            "skip_reason":str | None,
            "chunks": {
                "primary_found": bool,
                "primary_file":  str | None,
                "callers_count": int,
                "semantic_count":int,
            } | None,
        },
        "fix": {                    # the final provider fix output
            "ok":         bool,
            "raw_text":   str | None,   # full markdown from the selected LLM
            "saved_path": str | None,
            # parsed sections — ready for frontend to render directly
            "summary":      str,
            "root_cause":   str,
            "fixed_code":   str,        # fenced code block(s) as-is
            "what_changed": list[str],  # bullet list items
            "verification": str,
            # extracted before/after for CodeDiff component
            "before": str,
            "after":  str,
            "patch_plan": dict | None,
            "dry_run_patch": dict | None,
        } | None,
        "skipped": bool,
        "skip_reason": str | None,
    }
    """
    import hashlib
    import json
    import re

    def _parse_sections(text: str) -> dict:
        out = {}
        hits = list(re.finditer(r"^## (.+)$", text, re.MULTILINE))
        for i, m in enumerate(hits):
            title = m.group(1).strip()
            start = m.end()
            end   = hits[i + 1].start() if i + 1 < len(hits) else len(text)
            out[title] = text[start:end].strip()
        return out

    def _extract_code_blocks(section: str):
        blocks = re.findall(r"```[\w]*\n([\s\S]*?)```", section)
        blocks = [b.strip() for b in blocks]
        if not blocks:
            return "", section
        if len(blocks) == 1:
            return "", blocks[0]
        has_original = "original" in section.lower()
        before = blocks[0] if has_original else ""
        after  = blocks[1] if has_original else blocks[0]
        return before, after

    def _extract_patch_plan(text: str):
        match = re.search(r"```json\s*([\s\S]*?)```", text, re.IGNORECASE)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            return None
        if not isinstance(parsed, dict):
            return None
        files = parsed.get("files")
        if files is not None and not isinstance(files, list):
            parsed["files"] = []
        return parsed

    def _issue_id(service, function, message_sig):
        sig = f"{service}::{function}::{message_sig}"
        return hashlib.md5(sig.encode()).hexdigest()[:12]

    # Build lookup: index → prompt_result, fix_result
    # prompt_results and fix_results are ordered by group index (1-based)
    prompt_by_index = {pr["index"]: pr for pr in prompt_results if "index" in pr}
    fix_by_index    = {fr["index"]: fr for fr in fix_results    if "index" in fr}

    issues = []
    for pos, group in enumerate(groups, 1):
        rep    = group.get("representative", {})
        pr     = prompt_by_index.get(pos, {})
        fr     = fix_by_index.get(pos, {})
        skipped = pr.get("skipped", True)

        fix_data = None
        if not skipped:
            raw_fix = fr.get("fix") or {}
            fix_ok  = raw_fix.get("ok", False)
            raw_txt = raw_fix.get("text") or ""

            sections     = _parse_sections(raw_txt) if raw_txt else {}
            fixed_code   = sections.get("Fixed code", sections.get("Fix", ""))
            before, after = _extract_code_blocks(fixed_code) if fixed_code else ("", "")

            what_changed_raw = sections.get("What changed", "")
            what_changed = [
                line.lstrip("-* ").strip()
                for line in what_changed_raw.splitlines()
                if line.strip()
            ]

            patch_plan = _extract_patch_plan(raw_txt) if raw_txt else None
            patch_plan = _patch_plan_with_repo_paths(patch_plan, pr.get("chunks"), {"file": rep.get("file")})
            verification_text = sections.get("Verification", "")
            if patch_plan and isinstance(patch_plan.get("verification"), list):
                verification_text = "\n".join(str(step) for step in patch_plan.get("verification", []))

            fix_data = {
                "ok":           fix_ok,
                "raw_text":     raw_txt or None,
                "saved_path":   raw_fix.get("saved_path"),
                "summary":      sections.get("Summary", "") or (patch_plan or {}).get("summary", ""),
                "root_cause":   sections.get("Root cause", sections.get("Root Cause", "")) or (patch_plan or {}).get("root_cause", ""),
                "fixed_code":   fixed_code,
                "what_changed": what_changed or (patch_plan or {}).get("what_changed", []),
                "verification": verification_text,
                "before":       before,
                "after":        after,
                "patch_plan":   patch_plan,
                "dry_run_patch": None,
            }

        issues.append({
            "issue_id":    _issue_id(
                group.get("service", ""),
                group.get("function", ""),
                group.get("message_sig", ""),
            ),
            "index":       pos,
            "log": {
                "service":       group.get("service"),
                "function":      group.get("function"),
                "level":         group.get("level"),
                "count":         group.get("count"),
                "message":       rep.get("message"),
                "file":          rep.get("file"),
                "line":          rep.get("line"),
                "traceback":     rep.get("traceback"),
                "first_seen":    group.get("first_seen"),
                "last_seen":     group.get("last_seen"),
                "locations":     group.get("locations", []),
                "extra_samples": group.get("extra_samples", []),
            },
            "prompt": {
                "text":       pr.get("prompt_text"),
                "saved_path": pr.get("prompt_path"),
                "skipped":    skipped,
                "skip_reason":pr.get("reason"),
                "chunks":     pr.get("chunks"),
            },
            "fix":         fix_data,
            "skipped":     skipped,
            "skip_reason": pr.get("reason"),
        })

    return issues

def _build_summary(ingest, logs, fixes, req: AnalysisRequest) -> dict:
    successful_fixes = [f for f in fixes if not f.get("skipped") and f.get("fix", {}).get("ok")]
    return {
        "llm_provider":     req.llm_provider,
        "ai_generation_mode": "on_demand",
        "external_ai_enabled": _external_ai_enabled(),
        "service_filter":   req.service,
        "ai_input_redacted": True,
        "chunks_indexed":   ingest.get("chunks_indexed", "cached"),
        "raw_logs":         logs["total_raw_logs"],
        "log_groups":       logs["total_groups"],
        "issues_processed": len(fixes),
        "fixes_generated":  len(successful_fixes),
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.post("/api/analysis", response_model=JobStatus, status_code=202)
async def start_analysis(req: AnalysisRequest, background_tasks: BackgroundTasks):
    """
    Kick off the full pipeline asynchronously.
    Returns a job_id — poll GET /api/analysis/status?job_id=<id> for results.
    """
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "job_id":      job_id,
        "analysis_id": job_id,   # frontend uses this name
        "status":      "queued",
        "step":        None,
        "started_at":  None,
        "ended_at":    None,
        "result":      None,
        "error":       None,
        "metadata": {
            "github_url": req.github_url,
            "repo_full_name": _resolved_repo_full_name(req),
            "llm_provider": req.llm_provider,
            "request_payload": req.model_dump(),
        },
    }
    background_tasks.add_task(run_pipeline, job_id, req)
    logger.info(f"Queued job {job_id} for {req.github_url} with provider={req.llm_provider} allow_external_ai={load_config().allow_external_ai}")
    return jobs[job_id]


@app.get("/api/analysis/status", response_model=JobStatus)
async def get_status(job_id: str):
    """Poll for pipeline progress and results."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return jobs[job_id]


@app.post("/api/analysis/generate-fix")
async def generate_fix_for_issue(req: GenerateIssueFixRequest):
    if req.job_id not in jobs:
        raise HTTPException(status_code=404, detail=f"Job {req.job_id} not found")

    job = jobs[req.job_id]
    if job.get("status") != "done" or not job.get("result"):
        raise HTTPException(status_code=409, detail="Analysis job is not complete yet")
    if not _external_ai_enabled():
        raise HTTPException(status_code=403, detail="External AI calls are disabled on the backend")

    issue = next((item for item in job.get("result", {}).get("issues", []) if item.get("issue_id") == req.issue_id), None)
    if not issue:
        raise HTTPException(status_code=404, detail=f"Issue {req.issue_id} not found")

    job_request = _request_from_job(job)
    if req.llm_provider:
        payload = job_request.model_dump()
        payload["llm_provider"] = req.llm_provider
        job_request = AnalysisRequest(**payload)

    _patch_env(job_request)

    prompt_result, fix_result = await asyncio.get_event_loop().run_in_executor(
        None, run_generate_fix_for_issue, job_request, req.issue_id
    )

    existing_prompts = job.get("result", {}).get("prompts", []) or []
    existing_fixes = job.get("result", {}).get("fixes", []) or []
    prompts = _upsert_indexed_result(existing_prompts, prompt_result)
    fixes = _upsert_indexed_result(existing_fixes, fix_result)
    issues = _build_issues_map(job.get("result", {}).get("logs", {}).get("groups", []), prompts, fixes)
    issues = _attach_dry_run_previews(issues, job_request)

    job["result"]["prompts"] = prompts
    job["result"]["fixes"] = fixes
    job["result"]["issues"] = issues
    job["result"]["summary"] = _build_summary(job.get("result", {}).get("ingest", {}), job.get("result", {}).get("logs", {}), fixes, job_request)

    updated_issue = next((item for item in issues if item.get("issue_id") == req.issue_id), None)
    return {
        "job_id": req.job_id,
        "issue_id": req.issue_id,
        "issue": updated_issue,
    }


@app.post("/api/analysis/apply-local")
async def apply_local_patch(req: ApplyLocalPatchRequest):
    if req.job_id not in jobs:
        raise HTTPException(status_code=404, detail=f"Job {req.job_id} not found")

    job = jobs[req.job_id]
    if job.get("status") != "done" or not job.get("result"):
        raise HTTPException(status_code=409, detail="Analysis job is not complete yet")

    issues = job.get("result", {}).get("issues", [])
    issue = next((item for item in issues if item.get("issue_id") == req.issue_id), None)
    if not issue:
        raise HTTPException(status_code=404, detail=f"Issue {req.issue_id} not found")

    fix = issue.get("fix")
    if not fix or not fix.get("patch_plan"):
        raise HTTPException(status_code=400, detail="Issue does not contain an applyable patch plan")

    from patch_applier import apply_patch_plan

    repo_full_name = job.get("metadata", {}).get("repo_full_name")
    if not repo_full_name:
        raise HTTPException(status_code=500, detail="Missing repo metadata for job")

    result = apply_patch_plan(repo_full_name, fix.get("patch_plan"))
    fix["local_apply_result"] = result
    return {
        "job_id": req.job_id,
        "issue_id": req.issue_id,
        "repo_full_name": repo_full_name,
        "result": result,
    }


@app.post("/api/analysis/create-pr")
async def create_pr(req: CreatePullRequestRequest):
    if req.job_id not in jobs:
        raise HTTPException(status_code=404, detail=f"Job {req.job_id} not found")

    job = jobs[req.job_id]
    if job.get("status") != "done" or not job.get("result"):
        raise HTTPException(status_code=409, detail="Analysis job is not complete yet")

    issues = job.get("result", {}).get("issues", [])
    issue = next((item for item in issues if item.get("issue_id") == req.issue_id), None)
    if not issue:
        raise HTTPException(status_code=404, detail=f"Issue {req.issue_id} not found")

    fix = issue.get("fix")
    if not fix or not fix.get("patch_plan"):
        raise HTTPException(status_code=400, detail="Issue does not contain an applyable patch plan")

    from patch_applier import apply_patch_plan
    from git_ops import create_branch_commit_push
    from github_pr import create_pull_request

    repo_full_name = job.get("metadata", {}).get("repo_full_name")
    if not repo_full_name:
        raise HTTPException(status_code=500, detail="Missing repo metadata for job")

    local_apply_result = fix.get("local_apply_result")
    if not local_apply_result or not local_apply_result.get("ok"):
        local_apply_result = apply_patch_plan(repo_full_name, fix.get("patch_plan"))
        fix["local_apply_result"] = local_apply_result

    if not local_apply_result.get("ok"):
        raise HTTPException(status_code=400, detail="Local patch application failed; PR not created")

    commit_result = create_branch_commit_push(
        repo_full_name,
        req.issue_id,
        (fix.get("summary") or issue.get("log", {}).get("message") or req.issue_id),
    )
    fix["git_result"] = commit_result

    pr_result = create_pull_request(
        github_token=_resolved_runtime_github_token(req.github_token),
        repo_full_name=repo_full_name,
        head_branch=commit_result["branch_name"],
        base_branch=req.base_branch,
        title=req.title or _build_pr_title(issue),
        body=req.body or _build_pr_body(issue),
    )
    fix["pr_result"] = pr_result

    return {
        "job_id": req.job_id,
        "issue_id": req.issue_id,
        "repo_full_name": repo_full_name,
        "local_apply_result": local_apply_result,
        "git_result": commit_result,
        "pr_result": pr_result,
    }


@app.get("/api/logs")
async def get_logs(
    mongo_db:         Optional[str] = None,
    mongo_collection: Optional[str] = None,
    hours:            int = 120,
    top:              int = 5,
    log_limit:        int = 200,
    level:            Optional[str] = None,   # ERROR | WARNING | INFO | all
    service:          Optional[str] = None,
    group:            bool = True,            # False → return raw flat logs
):
    """
    Fetch and group logs from MongoDB using the default config.
    All params are optional — defaults match the debugger config.

    Query params:
        mongo_db         Database name          (default: log)
        mongo_collection Collection name        (default: logs)
        hours            Rolling window         (default: 120)
        top              Return top-N groups    (default: 5)
        level            ERROR | WARNING | all  (default: ERROR+WARNING)
        service          Filter by service name (default: all)
        group            true = grouped, false = raw flat logs
    """
    from fetch_logs import fetch_logs, group_logs, DEFAULT_LEVELS

    # resolve levels
    if level is None:
        levels = DEFAULT_LEVELS           # ["ERROR", "WARNING"]
    elif level.lower() == "all":
        levels = None
    else:
        levels = [level.upper()]

    try:
        cfg = load_config()
        raw_logs = fetch_logs(
            uri=require_env("MONGO_URI") if cfg.allow_frontend_secrets else require_env("MONGO_URI"),
            db_name=mongo_db or cfg.mongo_db,
            collection_name=mongo_collection or cfg.mongo_collection,
            levels=levels,
            service=service,
            hours=hours,
            limit=log_limit,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MongoDB error: {e}")

    if not group:
        # flat raw logs — just serialise ObjectId / datetime fields
        flat = []
        for log in raw_logs:
            flat.append({
                "timestamp": _fmt_ts(log.get("timestamp")),
                "level":     log.get("level"),
                "service":   log.get("service"),
                "function":  log.get("function"),
                "message":   log.get("message"),
                "file":      log.get("file"),
                "line":      log.get("line"),
                "traceback": log.get("traceback"),
                "extra":     log.get("extra"),
            })
        return {
            "total": len(flat),
            "logs":  flat,
        }

    # grouped (default)
    groups = group_logs(raw_logs)
    top_groups = groups[:top]

    serialised = []
    for g in top_groups:
        rep = g["representative"]
        serialised.append({
            "service":      g["service"],
            "function":     g["function"],
            "level":        g["level"],
            "count":        g["count"],
            "message_sig":  g["message_sig"],
            "first_seen":   _fmt_ts(min(g["timestamps"], default=None)),
            "last_seen":    _fmt_ts(max(g["timestamps"], default=None)),
            "locations":    g["locations"],
            "extra_samples":g["extra_samples"],
            "representative": {
                "message":   rep.get("message"),
                "file":      rep.get("file"),
                "line":      rep.get("line"),
                "traceback": rep.get("traceback"),
            },
        })

    return {
        "total_raw_logs": len(raw_logs),
        "total_groups":   len(groups),
        "showing_top":    len(serialised),
        "groups":         serialised,
    }


@app.get("/health")
async def health():
    return {"status": "ok", "jobs_in_memory": len(jobs)}


# ---------------------------------------------------------------------------
# Dev entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)