"""HTTP client for proxying requests to upstream APIs using OpenRouter SDK."""
from typing import Any, Dict, Optional

from openrouter import OpenRouter

from app.core.config import get_settings


class ProxyClient:
    """HTTP client for proxying requests to BASE_URL using OpenRouter SDK."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self._client: Optional[OpenRouter] = None

    def _get_client(self) -> OpenRouter:
        """Get or create the OpenRouter client."""
        if self._client is None:
            self._client = OpenRouter(
                api_key=self.settings.api_key or "",
                server_url=self.settings.base_url,
            )
        return self._client

    def post_chat(
        self,
        messages: list,
        model: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Send a chat completion request to the proxy target.
        
        Args:
            messages: List of message dicts with role and content
            model: Model identifier
            **kwargs: Additional OpenAI-compatible parameters
            
        Returns:
            Chat completion response dict
        """
        client = self._get_client()
        return client.chat.send(
            model=model,
            messages=messages,
            **kwargs,
        )

    def get_models(self) -> Dict[str, Any]:
        """Get list of available models."""
        client = self._get_client()
        return client.models.list()


_proxy_client: Optional[ProxyClient] = None


def get_proxy_client() -> ProxyClient:
    """Get the global proxy client instance."""
    global _proxy_client
    if _proxy_client is None:
        _proxy_client = ProxyClient()
    return _proxy_client


def reset_proxy_client() -> None:
    """Reset the global proxy client."""
    global _proxy_client
    _proxy_client = None