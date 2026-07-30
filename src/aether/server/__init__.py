"""
Aether REST server package.

Provides OpenAI-compatible routes (`/v1/chat`, `/v1/generate`, etc.) and
Prometheus-compatible metrics.
"""

from __future__ import annotations

from aether.server.app import create_app

__all__ = ["create_app"]
