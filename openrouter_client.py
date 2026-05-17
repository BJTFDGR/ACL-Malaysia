"""
openrouter_client.py
--------------------
Thin wrapper around the OpenRouter API (OpenAI-compatible endpoint).
Supports any model on https://openrouter.ai/models — GPT, Claude, Gemini,
DeepSeek, Llama, SEA-LION, Nemotron, and more, all via one API key.

Usage
-----
    python openrouter_client.py                        # smoke test DeepSeek + Nemotron
    python openrouter_client.py --model openai/gpt-5   # single model
    python openrouter_client.py --list                 # print top text models

Environment
-----------
    OPENROUTER_API_KEY   required

    Any model constant can be overridden via its matching env var, e.g.:
        GPT_STRONG_MODEL, CLAUDE_WEAK_MODEL, GEMINI_STRONG_MODEL, ...
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap
from typing import Optional

import requests

# Load .env if present (requires python-dotenv, silently skipped if missing)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

OPENROUTER_BASE = "https://openrouter.ai/api/v1"

# ---------------------------------------------------------------------------
# Model catalog — all routed through OpenRouter with a single API key.
# Override any entry via environment variables.
# ---------------------------------------------------------------------------

# OpenAI / GPT
GPT_STRONG_MODEL      = os.getenv("GPT_STRONG_MODEL",      "openai/gpt-5")
GPT_WEAK_MODEL        = os.getenv("GPT_WEAK_MODEL",        "openai/gpt-5.4-mini")

# Anthropic / Claude
CLAUDE_STRONG_MODEL   = os.getenv("CLAUDE_STRONG_MODEL",   "anthropic/claude-sonnet-4-6")
CLAUDE_WEAK_MODEL     = os.getenv("CLAUDE_WEAK_MODEL",     "anthropic/claude-haiku-4-5")

# Google / Gemini
GEMINI_STRONG_MODEL   = os.getenv("GEMINI_STRONG_MODEL",   "google/gemini-3.1-pro-preview")
GEMINI_WEAK_MODEL     = os.getenv("GEMINI_WEAK_MODEL",     "google/gemini-3.1-flash-lite")
GEMINI_STRONG_FALLBACK_MODELS = [
    m.strip()
    for m in os.getenv("GEMINI_STRONG_FALLBACK_MODELS", "google/gemini-3.1-flash-lite").split(",")
    if m.strip()
]

# DeepSeek
DEEPSEEK_STRONG_MODEL = os.getenv("DEEPSEEK_STRONG_MODEL", "deepseek/deepseek-v4-pro")
DEEPSEEK_WEAK_MODEL   = os.getenv("DEEPSEEK_WEAK_MODEL",   "deepseek/deepseek-v4-flash")

# Meta / Llama
LLAMA_STRONG_MODEL    = os.getenv("LLAMA_STRONG_MODEL",    "meta-llama/llama-3.3-70b-instruct")
LLAMA_WEAK_MODEL      = os.getenv("LLAMA_WEAK_MODEL",      "meta-llama/llama-3.1-8b-instruct")

# AI Singapore / SEA-LION
# Not hosted on OpenRouter — use SEA_LION_BASE_URL = https://api.sea-lion.ai/v1 directly.
# These constants are kept for direct-API callers; excluded from --all smoke test.
SEA_LION_STRONG_MODEL = os.getenv("SEA_LION_STRONG_MODEL", "aisingapore/Llama-SEA-LION-v3.5-70B-R")
SEA_LION_WEAK_MODEL   = os.getenv("SEA_LION_WEAK_MODEL",   "aisingapore/Gemma-SEA-LION-v4-27B-IT")
SEA_LION_BASE_URL     = os.getenv("SEA_LION_BASE_URL",     "https://api.sea-lion.ai/v1")

# NVIDIA
NEMOTRON_MODEL        = os.getenv("NEMOTRON_MODEL",        "nvidia/nemotron-3-super-120b-a12b:free")

# Convenience alias kept for backwards compatibility
DEEPSEEK_MODEL        = DEEPSEEK_STRONG_MODEL

SMOKE_PROMPT     = "Reply with exactly one sentence: what is 2 + 2?"


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        sys.exit(
            "ERROR: OPENROUTER_API_KEY environment variable is not set.\n"
            "Get a key at https://openrouter.ai/settings/keys"
        )
    return key


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {_api_key()}",
        "Content-Type": "application/json",
        # Optional attribution headers
        "HTTP-Referer": "https://github.com/xitongzhang/maylie",
        "X-Title": "Maylie",
    }


def chat(
    model: str,
    messages: list[dict],
    *,
    temperature: float = 0.7,
    max_tokens: int = 512,
    timeout: int = 60,
) -> str:
    """Send a chat request to OpenRouter; return the assistant text."""
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    resp = requests.post(
        f"{OPENROUTER_BASE}/chat/completions",
        headers=_headers(),
        json=payload,
        timeout=timeout,
    )
    try:
        resp.raise_for_status()
    except requests.HTTPError as exc:
        body = resp.text[:400]
        raise RuntimeError(f"OpenRouter API error {resp.status_code}: {body}") from exc

    data = resp.json()
    msg = data["choices"][0]["message"]
    # Reasoning models (e.g. gpt-5, deepseek-v4-pro) may return content=None
    # with the actual reply inside the 'reasoning' field.
    return msg.get("content") or msg.get("reasoning") or ""


def list_models(
    *,
    output_modality: str = "text",
    order: str = "most-popular",
    limit: int = 20,
) -> list[dict]:
    """
    Return models from https://openrouter.ai/models
    filtered by output_modality (default: text) and sorted by `order`.
    """
    params: dict = {}
    if output_modality:
        params["output_modalities"] = output_modality
    if order:
        params["order"] = order

    resp = requests.get(
        f"{OPENROUTER_BASE}/models",
        headers=_headers(),
        params=params,
        timeout=30,
    )
    resp.raise_for_status()
    models = resp.json().get("data", [])
    return models[:limit]


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def _smoke(model: str) -> bool:
    print(f"\n{'='*60}")
    print(f"  Model : {model}")
    print(f"  Prompt: {SMOKE_PROMPT}")
    print(f"{'='*60}")
    try:
        reply = chat(
            model,
            [{"role": "user", "content": SMOKE_PROMPT}],
            temperature=0.0,
            max_tokens=64,
        )
        print(f"  Reply : {reply.strip()}")
        print("  STATUS: PASS")
        return True
    except Exception as exc:
        print(f"  ERROR : {exc}")
        print("  STATUS: FAIL")
        return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="OpenRouter client — smoke test & model listing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Examples:
              python openrouter_client.py                         # smoke test DeepSeek + Nemotron
              python openrouter_client.py --model deepseek/deepseek-chat-v3-0324:free
              python openrouter_client.py --list --limit 30
        """),
    )
    parser.add_argument("--model", help="Model ID to smoke-test (overrides defaults)")
    parser.add_argument("--all", action="store_true", help="Smoke test every model in the catalog")
    parser.add_argument("--list", action="store_true", help="List top text models and exit")
    parser.add_argument("--limit", type=int, default=20, help="Max models to list (default 20)")
    args = parser.parse_args()

    if args.list:
        print(f"Top {args.limit} text models by popularity on OpenRouter:\n")
        models = list_models(limit=args.limit)
        for i, m in enumerate(models, 1):
            pricing = m.get("pricing", {})
            prompt_price  = pricing.get("prompt", "?")
            print(f"  {i:>3}. {m['id']:<55}  prompt=${prompt_price}/tok")
        return

    if args.model:
        ok = _smoke(args.model)
        sys.exit(0 if ok else 1)

    all_models = [
        ("GPT strong",        GPT_STRONG_MODEL),
        ("GPT weak",          GPT_WEAK_MODEL),
        ("Claude strong",     CLAUDE_STRONG_MODEL),
        ("Claude weak",       CLAUDE_WEAK_MODEL),
        ("Gemini strong",     GEMINI_STRONG_MODEL),
        ("Gemini weak",       GEMINI_WEAK_MODEL),
        ("DeepSeek strong",   DEEPSEEK_STRONG_MODEL),
        ("DeepSeek weak",     DEEPSEEK_WEAK_MODEL),
        ("Llama strong",      LLAMA_STRONG_MODEL),
        ("Llama weak",        LLAMA_WEAK_MODEL),
        ("Nemotron",          NEMOTRON_MODEL),
        # SEA-LION excluded: not on OpenRouter, requires SEA_LION_BASE_URL directly
    ]

    if args.all:
        targets = all_models
    else:
        # Default: DeepSeek + Nemotron
        targets = [("DeepSeek strong", DEEPSEEK_STRONG_MODEL), ("Nemotron", NEMOTRON_MODEL)]

    results = []
    for label, model in targets:
        print(f"\n[{label}]")
        results.append(_smoke(model))

    print(f"\n{'='*60}")
    passed = sum(results)
    print(f"  {passed}/{len(results)} models passed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
