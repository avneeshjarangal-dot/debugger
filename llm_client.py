import os

import httpx

from config import load_config

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3-flash-preview")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4.1-mini")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")


def _ensure_ai_enabled():
    if not load_config().allow_external_ai:
        raise RuntimeError("External AI calls are disabled. Set ALLOW_EXTERNAL_AI=true to enable LLM requests.")


def _generate_with_gemini(prompt: str, system_instruction: str | None = None) -> str:
    cfg = load_config()
    if not cfg.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY not set.")

    try:
        from google import genai
        from google.genai import types
    except ImportError:
        raise RuntimeError("google-genai not installed. Run: pip install google-genai")

    client = genai.Client(api_key=cfg.gemini_api_key)
    config = types.GenerateContentConfig(system_instruction=system_instruction) if system_instruction else None
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=config,
    )
    return (response.text or "").strip()


def _chat_completion(base_url: str, api_key: str, model: str, prompt: str, system_instruction: str | None = None, extra_headers: dict | None = None) -> str:
    messages = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})
    messages.append({"role": "user", "content": prompt})

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if extra_headers:
        headers.update(extra_headers)

    response = httpx.post(
        base_url,
        headers=headers,
        json={
            "model": model,
            "messages": messages,
            "temperature": 0.2,
        },
        timeout=60.0,
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


def _generate_with_openai(prompt: str, system_instruction: str | None = None) -> str:
    cfg = load_config()
    if not cfg.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY not set.")
    return _chat_completion(
        "https://api.openai.com/v1/chat/completions",
        cfg.openai_api_key,
        OPENAI_MODEL,
        prompt,
        system_instruction=system_instruction,
    )


def _generate_with_openrouter(prompt: str, system_instruction: str | None = None) -> str:
    cfg = load_config()
    if not cfg.openrouter_api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set.")
    return _chat_completion(
        "https://openrouter.ai/api/v1/chat/completions",
        cfg.openrouter_api_key,
        OPENROUTER_MODEL,
        prompt,
        system_instruction=system_instruction,
        extra_headers={
            "HTTP-Referer": os.environ.get("OPENROUTER_REFERER", "http://localhost:3000"),
            "X-Title": os.environ.get("OPENROUTER_APP_NAME", "Proctor Debugger"),
        },
    )


def _generate_with_groq(prompt: str, system_instruction: str | None = None) -> str:
    cfg = load_config()
    if not cfg.groq_api_key:
        raise RuntimeError("GROQ_API_KEY not set.")
    return _chat_completion(
        "https://api.groq.com/openai/v1/chat/completions",
        cfg.groq_api_key,
        GROQ_MODEL,
        prompt,
        system_instruction=system_instruction,
    )


def generate_text(provider: str, prompt: str, system_instruction: str | None = None) -> str:
    _ensure_ai_enabled()
    provider = (provider or "GEMINI").upper()
    if provider == "GEMINI":
        return _generate_with_gemini(prompt, system_instruction=system_instruction)
    if provider == "OPEN_AI":
        return _generate_with_openai(prompt, system_instruction=system_instruction)
    if provider == "OPENROUTER":
        return _generate_with_openrouter(prompt, system_instruction=system_instruction)
    if provider == "GROQ":
        return _generate_with_groq(prompt, system_instruction=system_instruction)
    raise RuntimeError(f"Unsupported llm provider: {provider}")
