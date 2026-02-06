#!/usr/bin/env python3
"""
OpenRouter Client for Visual Validation

Python client for OpenRouter API, following the same pattern as MageAgent's
TypeScript OpenRouterClient. Uses Claude Opus 4.6 for visual validation tasks.

Environment Variables:
    OPENROUTER_API_KEY: Required - Your OpenRouter API key

Also checks these .env file locations for API key:
    - ~/.env
    - /Users/don/Adverant/Adverant-Nexus/docker/.env.unified-nexus
    - ./.env

Usage:
    from openrouter_client import OpenRouterClient

    client = OpenRouterClient()
    response = client.create_vision_completion(
        image_path="board.png",
        prompt="Analyze this PCB layout",
        system_prompt="You are a PCB expert..."
    )
"""

import base64
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

from validation_exceptions import MissingDependencyFailure


# Model configuration - using Claude Opus 4.6 as specified
# OpenRouter uses simplified model IDs (not the full Anthropic model names)
CLAUDE_OPUS_MODEL = "anthropic/claude-opus-4.6"
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"

# Common .env file locations to check for API key
ENV_FILE_LOCATIONS = [
    Path.home() / '.env',
    Path('/Users/don/Adverant/Adverant-Nexus/docker/.env.unified-nexus'),
    Path.cwd() / '.env',
]


def _load_api_key_from_env_files() -> Optional[str]:
    """Try to load OPENROUTER_API_KEY from common .env file locations."""
    for env_path in ENV_FILE_LOCATIONS:
        if env_path.exists():
            try:
                with open(env_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith('OPENROUTER_API_KEY='):
                            value = line.split('=', 1)[1].strip()
                            # Remove quotes if present
                            if value.startswith('"') and value.endswith('"'):
                                value = value[1:-1]
                            elif value.startswith("'") and value.endswith("'"):
                                value = value[1:-1]
                            if value and value != 'your_openrouter_api_key_here':
                                return value
            except Exception:
                continue
    return None


@dataclass
class CompletionUsage:
    """Token usage information from completion."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class CompletionResponse:
    """Response from OpenRouter completion."""
    id: str
    content: str
    model: str
    usage: CompletionUsage
    finish_reason: str
    raw_response: Dict[str, Any] = field(default_factory=dict)


class OpenRouterError(Exception):
    """Error from OpenRouter API."""
    def __init__(self, message: str, status_code: int = 0, response_data: dict = None):
        self.message = message
        self.status_code = status_code
        self.response_data = response_data or {}
        super().__init__(f"OpenRouter Error ({status_code}): {message}")


class OpenRouterClient:
    """
    OpenRouter API client for visual validation.

    Mirrors the MageAgent OpenRouterClient pattern:
    - Uses OPENROUTER_API_KEY environment variable
    - Supports vision/multimodal requests
    - Only uses paid models (Claude Opus 4.6)
    - Includes retry logic with exponential backoff
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = OPENROUTER_API_BASE,
        model: str = CLAUDE_OPUS_MODEL,
        timeout: int = 120,
        max_retries: int = 3
    ):
        """
        Initialize OpenRouter client.

        Args:
            api_key: OpenRouter API key (defaults to OPENROUTER_API_KEY env var)
            base_url: OpenRouter API base URL
            model: Model to use (defaults to Claude Opus 4.6)
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts

        Raises:
            MissingDependencyFailure: If requests package not installed or API key not set
        """
        if not HAS_REQUESTS:
            raise MissingDependencyFailure(
                "requests package required for OpenRouter client",
                dependency_name="requests",
                install_instructions="pip install requests"
            )

        # Try to get API key from: 1) argument, 2) env var, 3) .env files
        self.api_key = api_key or os.environ.get('OPENROUTER_API_KEY') or _load_api_key_from_env_files()
        if not self.api_key:
            raise MissingDependencyFailure(
                "OPENROUTER_API_KEY environment variable not set",
                dependency_name="OPENROUTER_API_KEY",
                install_instructions=(
                    "1. Get API key from https://openrouter.ai/keys\n"
                    "2. export OPENROUTER_API_KEY=your-key-here\n"
                    "3. Or add to ~/.env or /Users/don/Adverant/Adverant-Nexus/docker/.env.unified-nexus\n"
                    "4. Re-run this script"
                )
            )

        self.base_url = base_url.rstrip('/')
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries

        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {self.api_key}',
            'HTTP-Referer': 'https://adverant.ai',
            'X-Title': 'EE Design Visual Validator',
            'Content-Type': 'application/json'
        })

    def _load_image_base64(self, image_path: str) -> str:
        """Load image file and convert to base64."""
        with open(image_path, 'rb') as f:
            return base64.standard_b64encode(f.read()).decode('utf-8')

    def _get_media_type(self, image_path: str) -> str:
        """Determine media type from file extension."""
        ext = Path(image_path).suffix.lower()
        media_types = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
            '.webp': 'image/webp'
        }
        return media_types.get(ext, 'image/png')

    def _make_request(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int = 4096,
        temperature: float = 0.1
    ) -> CompletionResponse:
        """
        Make a completion request to OpenRouter.

        Args:
            messages: List of message dicts (role, content)
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (low for factual analysis)

        Returns:
            CompletionResponse with model output

        Raises:
            OpenRouterError: On API errors
        """
        payload = {
            'model': self.model,
            'messages': messages,
            'max_tokens': max_tokens,
            'temperature': temperature
        }

        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self.session.post(
                    f'{self.base_url}/chat/completions',
                    json=payload,
                    timeout=self.timeout
                )

                if response.status_code == 200:
                    data = response.json()

                    # Extract response content
                    choice = data.get('choices', [{}])[0]
                    message = choice.get('message', {})
                    content = message.get('content', '')

                    # Extract usage
                    usage_data = data.get('usage', {})
                    usage = CompletionUsage(
                        prompt_tokens=usage_data.get('prompt_tokens', 0),
                        completion_tokens=usage_data.get('completion_tokens', 0),
                        total_tokens=usage_data.get('total_tokens', 0)
                    )

                    return CompletionResponse(
                        id=data.get('id', ''),
                        content=content,
                        model=data.get('model', self.model),
                        usage=usage,
                        finish_reason=choice.get('finish_reason', 'stop'),
                        raw_response=data
                    )

                # Handle retryable errors
                if response.status_code in [429, 502, 503, 504]:
                    wait_time = (2 ** attempt) + 1
                    print(f"  Retrying in {wait_time}s (status {response.status_code})...")
                    time.sleep(wait_time)
                    last_error = OpenRouterError(
                        f"HTTP {response.status_code}",
                        response.status_code,
                        response.json() if response.text else {}
                    )
                    continue

                # Non-retryable error
                error_data = response.json() if response.text else {}
                error_msg = error_data.get('error', {}).get('message', response.text)

                # Provide more helpful message for common errors
                if response.status_code == 401:
                    if 'User not found' in error_msg:
                        error_msg = (
                            f"OpenRouter API key is invalid or expired. "
                            f"Please get a new key from https://openrouter.ai/keys "
                            f"Original error: {error_msg}"
                        )

                raise OpenRouterError(error_msg, response.status_code, error_data)

            except requests.RequestException as e:
                wait_time = (2 ** attempt) + 1
                print(f"  Network error, retrying in {wait_time}s: {e}")
                time.sleep(wait_time)
                last_error = OpenRouterError(str(e), 0)

        raise last_error or OpenRouterError("Max retries exceeded", 0)

    def create_completion(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.1
    ) -> CompletionResponse:
        """
        Create a text completion.

        Args:
            prompt: User prompt
            system_prompt: Optional system instructions
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature

        Returns:
            CompletionResponse with model output
        """
        messages = []

        if system_prompt:
            messages.append({
                'role': 'system',
                'content': system_prompt
            })

        messages.append({
            'role': 'user',
            'content': prompt
        })

        return self._make_request(messages, max_tokens, temperature)

    def create_vision_completion(
        self,
        image_path: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.1,
        image_detail: str = 'high'
    ) -> CompletionResponse:
        """
        Create a vision/multimodal completion with an image.

        Args:
            image_path: Path to image file
            prompt: User prompt about the image
            system_prompt: Optional system instructions
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (low for analysis)
            image_detail: Image detail level ('auto', 'low', 'high')

        Returns:
            CompletionResponse with model analysis
        """
        # Load and encode image
        image_data = self._load_image_base64(image_path)
        media_type = self._get_media_type(image_path)

        messages = []

        if system_prompt:
            messages.append({
                'role': 'system',
                'content': system_prompt
            })

        # Build multimodal content (text + image)
        user_content = [
            {
                'type': 'text',
                'text': prompt
            },
            {
                'type': 'image_url',
                'image_url': {
                    'url': f'data:{media_type};base64,{image_data}',
                    'detail': image_detail
                }
            }
        ]

        messages.append({
            'role': 'user',
            'content': user_content
        })

        return self._make_request(messages, max_tokens, temperature)

    def test_connection(self) -> bool:
        """
        Test connection to OpenRouter API.

        Returns:
            True if connection successful

        Raises:
            OpenRouterError: If connection fails
        """
        try:
            response = self.session.get(
                f'{self.base_url}/models',
                timeout=10
            )

            if response.status_code == 200:
                models = response.json().get('data', [])
                print(f"OpenRouter connection verified: {len(models)} models available")
                return True

            raise OpenRouterError(
                f"Connection test failed: {response.text}",
                response.status_code
            )

        except requests.RequestException as e:
            raise OpenRouterError(f"Connection test failed: {e}", 0)

    def get_model_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the configured model."""
        try:
            response = self.session.get(
                f'{self.base_url}/models',
                timeout=10
            )

            if response.status_code == 200:
                models = response.json().get('data', [])
                for model in models:
                    if model.get('id') == self.model:
                        return model
            return None

        except requests.RequestException:
            return None


def get_openrouter_client(
    api_key: Optional[str] = None,
    model: str = CLAUDE_OPUS_MODEL
) -> OpenRouterClient:
    """
    Get an OpenRouter client instance.

    Args:
        api_key: Optional API key (defaults to env var)
        model: Model to use (defaults to Claude Opus 4.6)

    Returns:
        Configured OpenRouterClient

    Raises:
        MissingDependencyFailure: If dependencies not available
    """
    return OpenRouterClient(api_key=api_key, model=model)


# For backwards compatibility - adapter to match anthropic SDK interface
class OpenRouterAnthropicAdapter:
    """
    Adapter class that provides an interface similar to anthropic.Anthropic
    but uses OpenRouter as the backend.

    This allows existing code using the Anthropic SDK to switch to OpenRouter
    with minimal changes.
    """

    def __init__(self, api_key: Optional[str] = None):
        """Initialize adapter with OpenRouter client."""
        self.client = OpenRouterClient(api_key=api_key)
        self.messages = self  # For messages.create() syntax

    def create(
        self,
        model: str,
        max_tokens: int,
        system: str,
        messages: List[Dict[str, Any]]
    ) -> 'AdapterResponse':
        """
        Create a completion matching Anthropic SDK interface.

        Args:
            model: Model name (ignored, uses configured OpenRouter model)
            max_tokens: Maximum tokens
            system: System prompt
            messages: List of message dicts

        Returns:
            AdapterResponse mimicking Anthropic response structure
        """
        # Extract the user message content
        user_message = messages[0] if messages else {'content': ''}
        content = user_message.get('content', '')

        # Check if content is multimodal (list with image)
        if isinstance(content, list):
            # Find image and text parts
            image_part = None
            text_part = None

            for part in content:
                if part.get('type') == 'image':
                    image_part = part
                elif part.get('type') == 'text':
                    text_part = part.get('text', '')

            if image_part:
                # Handle base64 image from Anthropic format
                source = image_part.get('source', {})
                image_data = source.get('data', '')
                media_type = source.get('media_type', 'image/png')

                # Build OpenRouter multimodal message
                openrouter_messages = [
                    {'role': 'system', 'content': system}
                ]

                openrouter_content = [
                    {'type': 'text', 'text': text_part or ''},
                    {
                        'type': 'image_url',
                        'image_url': {
                            'url': f'data:{media_type};base64,{image_data}',
                            'detail': 'high'
                        }
                    }
                ]

                openrouter_messages.append({
                    'role': 'user',
                    'content': openrouter_content
                })

                response = self.client._make_request(
                    openrouter_messages,
                    max_tokens=max_tokens
                )

                return AdapterResponse(response.content)

        # Simple text completion
        response = self.client.create_completion(
            prompt=content if isinstance(content, str) else str(content),
            system_prompt=system,
            max_tokens=max_tokens
        )

        return AdapterResponse(response.content)


@dataclass
class AdapterResponse:
    """Response object matching Anthropic SDK structure."""
    _content: str

    @property
    def content(self) -> List['ContentBlock']:
        """Return content as list of content blocks."""
        return [ContentBlock(self._content)]


@dataclass
class ContentBlock:
    """Content block matching Anthropic SDK structure."""
    text: str
    type: str = 'text'


if __name__ == '__main__':
    # Test the client
    import sys

    print("Testing OpenRouter Client...")
    print(f"Model: {CLAUDE_OPUS_MODEL}")

    try:
        client = get_openrouter_client()

        # Test connection
        client.test_connection()

        # Test simple completion
        print("\nTesting text completion...")
        response = client.create_completion(
            prompt="What is 2+2? Answer in one word.",
            system_prompt="You are a helpful assistant. Be concise."
        )
        print(f"Response: {response.content}")
        print(f"Tokens: {response.usage.total_tokens}")

        print("\nOpenRouter client working correctly!")

    except MissingDependencyFailure as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)
    except OpenRouterError as e:
        print(f"API ERROR: {e}", file=sys.stderr)
        sys.exit(1)
