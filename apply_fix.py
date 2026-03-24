"""
apply_fix.py  —  Step 4: Send a fix prompt to Gemini, get a recommended fix.

What this does:
  1. Reads a prompt from  debugger/generated_prompts/  (or stdin / --prompt-file)
  2. Sends it to Gemini 1.5 Pro (or Flash)
  3. Prints the fix with a clear diff-friendly format
  4. Saves the fix to  debugger/fixes/<prompt_stem>/gemini.md

Usage (run from inside debugger/):
    # pick a specific prompt file
    python apply_fix.py --prompt-file generated_prompts/01_order-service__create_order.txt

    # auto-pick the most recent prompt in generated_prompts/
    python apply_fix.py

    # process all prompt files in generated_prompts/
    python apply_fix.py --all

    # skip saving to disk
    python apply_fix.py --no-save

Env vars:
    GEMINI_API_KEY      AIza...
"""

import os
import sys
import time
import argparse
from datetime import datetime, timezone
from pathlib import Path

from llm_client import GEMINI_MODEL, GROQ_MODEL, OPENAI_MODEL, OPENROUTER_MODEL, generate_text

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
_BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
PROMPTS_DIR  = os.path.join(_BASE_DIR, "generated_prompts")
FIXES_DIR    = os.path.join(_BASE_DIR, "fixes")

# ---------------------------------------------------------------------------
# System prompt — what we want from the fixing LLM
# ---------------------------------------------------------------------------
FIX_SYSTEM_PROMPT = """\
You are a senior software engineer. You will be given a detailed bug report that describes a production error, includes the failing source code, root cause analysis, and specific fix instructions.

Your job is to produce a complete, ready-to-apply fix. Be conservative and prefer the smallest safe change.

Hard requirements:
1. Treat the primary failing file from the prompt as canonical. Do not rename,    normalize, shorten, or invent file paths.
2. By default, modify exactly one file: the primary failing file.
3. Only include additional files if the prompt explicitly proves they must    change too. If you include them, justify that in the summary and what-changed.
4. For updates, `before` and `after` must be full function/class/block    replacements, not one-line substitutions.
5. Do not propose speculative refactors or unrelated cleanups.

Structure your response exactly as follows:

## Summary
One sentence: what the bug is and what the fix does.

## Root cause
2-3 sentences explaining why the error occurs.

## Fixed code
For every file that needs to change, provide a fenced code block with the complete fixed function or class — never a partial snippet. Label each block with the exact canonical file path from the prompt.

## What changed
A bullet list of every specific line-level change made, in the format:
  - <file>:<line> — <what changed and why>

## Verification
Step-by-step instructions to confirm the fix works locally and in production.

## Patch Plan JSON
Return a fenced ```json``` block containing a machine-readable patch plan with this shape:
{
  "summary": "short summary",
  "root_cause": "short root cause",
  "files": [
    {
      "path": "exact canonical relative/file/path from prompt",
      "action": "update",
      "target": "function/class/block being changed",
      "before": "exact full existing function/class/block",
      "after": "exact full replacement function/class/block"
    }
  ],
  "what_changed": ["bullet item"],
  "verification": ["step 1", "step 2"]
}
If you cannot produce an exact full-block replacement for a file, omit that file from the JSON.
Keep the JSON valid.
"""


# ---------------------------------------------------------------------------
# LLM calls
# ---------------------------------------------------------------------------
def get_fixes(prompt: str, provider: str = "GEMINI") -> dict:
    """Send prompt to the selected LLM provider."""
    provider_key = provider.lower()
    results = {}
    try:
        # Keep a small delay to avoid bursty prototype traffic.
        time.sleep(1)
        results[provider_key] = {"ok": True, "text": generate_text(provider, prompt, system_instruction=FIX_SYSTEM_PROMPT)}
    except Exception as e:
        results[provider_key] = {"ok": False, "text": str(e)}

    return results


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
def save_fixes(prompt_path: str, fixes: dict) -> dict[str, str]:
    """
    Save each fix under  debugger/fixes/<prompt_stem>/gemini.md
    Returns {model_name: saved_path}.
    """
    stem      = Path(prompt_path).stem          # e.g. 01_order-service__create_order
    fixes_dir = os.path.join(FIXES_DIR, stem)
    os.makedirs(fixes_dir, exist_ok=True)

    saved = {}
    for model_name, result in fixes.items():
        filename = f"{model_name}.md"
        path     = os.path.join(fixes_dir, filename)
        content  = result["text"] if result["ok"] else f"ERROR: {result['text']}"
        with open(path, "w", encoding="utf-8") as f:
            # header so the file is self-contained
            f.write(f"# Fix recommendation — {model_name.upper()}\n")
            f.write(f"# Prompt: {os.path.basename(prompt_path)}\n")
            f.write(f"# Generated: {datetime.now(tz=timezone.utc).isoformat()}\n\n")
            f.write(content)
        saved[model_name] = path

    return saved


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------
SEP  = "─" * 72
SEP2 = "═" * 72

MODEL_LABELS = {
    "gemini": f"Gemini  ({GEMINI_MODEL})",
    "open_ai": f"OpenAI  ({OPENAI_MODEL})",
    "openrouter": f"OpenRouter  ({OPENROUTER_MODEL})",
    "groq": f"Groq  ({GROQ_MODEL})",
}


def display_fixes(prompt_path: str, fixes: dict, saved: dict):
    stem = Path(prompt_path).stem

    print()
    print(SEP2)
    print(f"  FIX RECOMMENDATION  —  {stem}")
    print(SEP2)

    for model_name in fixes.keys():
        result = fixes.get(model_name, {"ok": False, "text": "Did not run"})
        label  = MODEL_LABELS.get(model_name, model_name.upper())

        print()
        print(SEP)
        print(f"  [ {label} ]")
        if saved.get(model_name):
            print(f"  Saved → {saved[model_name]}")
        print(SEP)

        if not result["ok"]:
            print(f"\n  ERROR: {result['text']}\n")
            continue

        for line in result["text"].splitlines():
            print(f"  {line}")

        print()

    print(SEP2)
    print()


# ---------------------------------------------------------------------------
# Prompt file resolution
# ---------------------------------------------------------------------------
def resolve_prompt_files(args) -> list[str]:
    """Return list of prompt file paths to process."""
    if args.prompt_file:
        p = args.prompt_file
        if not os.path.isabs(p):
            p = os.path.join(_BASE_DIR, p)
        if not os.path.exists(p):
            print(f"\n  ERROR: prompt file not found: {p}\n")
            sys.exit(1)
        return [p]

    if not os.path.isdir(PROMPTS_DIR):
        print(f"\n  ERROR: {PROMPTS_DIR} does not exist.")
        print("  Run generate_prompts.py first.\n")
        sys.exit(1)

    txt_files = sorted(Path(PROMPTS_DIR).glob("*.txt"))
    if not txt_files:
        print(f"\n  No .txt files found in {PROMPTS_DIR}\n")
        sys.exit(1)

    if args.all:
        return [str(p) for p in txt_files]

    # default — most recently modified file
    latest = max(txt_files, key=lambda p: p.stat().st_mtime)
    return [str(latest)]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Send a fix prompt to Gemini and get a fix recommendation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--prompt-file", default=None,
        help="Path to a specific prompt .txt file (relative to debugger/ is fine)"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Process every .txt file in generated_prompts/"
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Don't save fixes to disk"
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args         = parse_args()
    prompt_files = resolve_prompt_files(args)

    print(f"\n  {len(prompt_files)} prompt file(s) to process")
    print(f"  Model: {GEMINI_MODEL}\n")

    for i, prompt_path in enumerate(prompt_files, 1):
        print(f"  [{i}/{len(prompt_files)}] {os.path.basename(prompt_path)}")

        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt = f.read().strip()

        if not prompt:
            print("  Skipping — file is empty\n")
            continue

        print("  Calling Gemini...")
        fixes = get_fixes(prompt)

        gemini_ok = fixes.get("gemini", {}).get("ok", False)
        print(f"  Gemini : {'ok' if gemini_ok else 'FAILED'}")

        saved = {}
        if not args.no_save:
            saved = save_fixes(prompt_path, fixes)
            for model_name, path in saved.items():
                print(f"  Saved  : {path}")

        display_fixes(prompt_path, fixes, saved)

    print(f"  Done. Fixes saved under: {FIXES_DIR}\n")


if __name__ == "__main__":
    main()