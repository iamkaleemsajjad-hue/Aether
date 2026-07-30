"""
Opt-in authentication utilities for Hub API access.

Provides a simple token-based auth flow. In production this would integrate
with OAuth2, API key rotation, and scoped access tokens.
"""

from __future__ import annotations

import hmac
import json
import time
from dataclasses import dataclass
from typing import Any

from aether.core.exceptions import AuthenticationError
from aether.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class AuthCredentials:
    """Credentials for Huv authentication."""

    access_key: str
    secret_key: str

    def sign_request(self, method: str, path: str, body: bytes | None = None) -> str:
        """Create an HMAC-SHA256 signature for a request.

        Args:
            method: HTTP method (GET, POST, etc.).
            path: Request path.
            body: Optional request body bytes.

        Returns:
            Hex-encoded signature string.
        """
        timestamp = str(int(time.time()))
        message_parts = [method.upper(), path, timestamp]
        if body:
            message_parts.append(body.hex())
        message = "\n".join(message_parts)
        signature = hmac.new(self.secret_key.encode(), message.encode(), "sha256").hexdigest()
        return f"{self.access_key}:{timestamp}:{signature}"

    def to_dict(self) -> dict[str, str]:
        return {"access_key": self.access_key}

    @staticmethod
    def from_env() -> AuthCredentials | None:
        """Load credentials from environment variables."""
        import os

        access_key = os.environ.get("AETHER_ACCESS_KEY", "")
        secret_key = os.environ.get("AETHER_SECRET_KEY", "")
        if access_key and secret_key:
            return AuthCredentials(access_key=access_key, secret_key=secret_key)
        return None


class TokenManager:
    """Manages API tokens for Hub authentication."""

    def __init__(self, token_file: str | None = None) -> None:
        self._token_file = None
        if token_file:
            self._token_file = __import__("pathlib").Path(token_file)
        self._tokens: dict[str, dict[str, Any]] = {}

    def add_token(self, name: str, token: str, scopes: list[str] | None = None) -> None:
        """Register a token."""
        self._tokens[name] = {"token": token, "scopes": scopes or ["read"], "created_at": time.time()}

    def validate_token(self, token: str) -> str | None:
        """Validate a token and return its name if valid."""
        for name, info in self._tokens.items():
            if hmac.compare_digest(info["token"], token):
                return name
        return None

    def revoke_token(self, name: str) -> bool:
        """Revoke a token by name."""
        return self._tokens.pop(name, None) is not None

    def list_tokens(self) -> list[dict[str, Any]]:
        """Return a summary of all registered tokens (without secret values)."""
        return [
            {"name": name, "scopes": info["scopes"], "created_at": info["created_at"]}
            for name, info in self._tokens.items()
        ]

    def __repr__(self) -> str:
        return f"TokenManager(tokens={len(self._tokens)})"
