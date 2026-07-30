"""Example: Serve a model and call it with the OpenAI SDK."""

from __future__ import annotations

from aether import RuntimeConfig
from aether.server.app import create_app


def main() -> None:
    config = RuntimeConfig(
        optimize_for="latency",
        speculative_decoding=True,
    )
    app = create_app(config)
    print("Aether server starting at http://localhost:11434")
    print("Try: curl http://localhost:11434/v1/health")
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=11434)


if __name__ == "__main__":
    main()
