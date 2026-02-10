"""
Centralized LLM Provider Configuration.

Provides runtime-resolved LLM configuration based on AI_PROVIDER env var.
All agents should call get_llm_config() at request time (NOT at module import time)
to support per-request provider switching from the dashboard.

Supported providers:
- "openrouter" (default): Routes through OpenRouter API
- "claude_code_max": Routes through Claude Code Max proxy pod (K8s internal, no API key needed)

Usage:
    from llm_provider import get_llm_config, get_llm_headers, get_llm_base_url

    config = get_llm_config()
    response = await client.post(config["base_url"], headers=config["headers"], json=payload)

Environment Variables:
    AI_PROVIDER: "openrouter" or "claude_code_max" (default: "openrouter")
    CLAUDE_CODE_PROXY_URL: Proxy pod URL (default: "http://claude-code-proxy.nexus.svc.cluster.local:3100")
    OPENROUTER_API_KEY: OpenRouter API key (required when AI_PROVIDER=openrouter)
"""

import os
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Default proxy pod URL (K8s service DNS)
DEFAULT_PROXY_URL = "http://claude-code-proxy.nexus.svc.cluster.local:3100"


def get_ai_provider() -> str:
    """Get the current AI provider from environment."""
    return os.environ.get("AI_PROVIDER", "openrouter")


def get_llm_base_url(chat_completions: bool = True) -> str:
    """
    Get the LLM API base URL for the current provider.

    Args:
        chat_completions: If True, returns the full /chat/completions endpoint.
                         If False, returns the base /v1 URL.

    Returns:
        The appropriate API URL.
    """
    provider = get_ai_provider()

    if provider == "claude_code_max":
        proxy_url = os.environ.get("CLAUDE_CODE_PROXY_URL", DEFAULT_PROXY_URL)
        if chat_completions:
            return f"{proxy_url}/v1/chat/completions"
        return f"{proxy_url}/v1"
    else:
        if chat_completions:
            return "https://openrouter.ai/api/v1/chat/completions"
        return "https://openrouter.ai/api/v1"


def get_llm_api_key() -> str:
    """
    Get the API key for the current provider.

    Returns empty string for claude_code_max (no auth needed for internal proxy).
    """
    provider = get_ai_provider()

    if provider == "claude_code_max":
        return ""
    return os.environ.get("OPENROUTER_API_KEY", "")


def get_llm_headers() -> Dict[str, str]:
    """
    Get the appropriate HTTP headers for the current provider.

    For OpenRouter: includes Authorization, HTTP-Referer, X-Title
    For Claude Code Max proxy: only Content-Type (no auth needed internally)
    """
    provider = get_ai_provider()
    headers = {"Content-Type": "application/json"}

    if provider == "claude_code_max":
        # No auth or OpenRouter-specific headers for internal proxy
        pass
    else:
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        headers["HTTP-Referer"] = "https://adverant.ai"
        headers["X-Title"] = "Nexus EE Design"

    return headers


def get_llm_config() -> Dict[str, object]:
    """
    Get complete LLM configuration for the current provider.

    Returns a dict with: provider, base_url, api_key, headers, needs_auth
    """
    provider = get_ai_provider()
    return {
        "provider": provider,
        "base_url": get_llm_base_url(chat_completions=True),
        "base_url_v1": get_llm_base_url(chat_completions=False),
        "api_key": get_llm_api_key(),
        "headers": get_llm_headers(),
        "needs_auth": provider != "claude_code_max",
    }


def check_llm_available() -> bool:
    """Check if LLM provider is properly configured."""
    provider = get_ai_provider()
    if provider == "claude_code_max":
        return True  # Proxy pod handles auth internally
    return bool(os.environ.get("OPENROUTER_API_KEY"))


def log_provider_info(agent_name: str = "Unknown") -> None:
    """Log the current LLM provider configuration."""
    provider = get_ai_provider()
    if provider == "claude_code_max":
        proxy_url = os.environ.get("CLAUDE_CODE_PROXY_URL", DEFAULT_PROXY_URL)
        logger.info(f"[{agent_name}] LLM Provider: Claude Code Max proxy at {proxy_url}")
    else:
        has_key = bool(os.environ.get("OPENROUTER_API_KEY"))
        logger.info(f"[{agent_name}] LLM Provider: OpenRouter (API key: {'present' if has_key else 'MISSING'})")
