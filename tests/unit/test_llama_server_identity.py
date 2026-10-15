"""Local HTTP lifecycle contracts; these do not validate GGUF model inference."""

from __future__ import annotations

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import Mock

import pytest

from aether.backends.llamacpp_backend import LlamaCppBackend
from aether.core.exceptions import BackendError


def test_occupied_explicit_port_is_rejected_before_launch(monkeypatch):
    backend = LlamaCppBackend()
    monkeypatch.setattr(backend, "_find_llama_binary", lambda: "llama-server")
    launch = Mock()
    monkeypatch.setattr("aether.backends.llamacpp_backend.subprocess.Popen", launch)
    with socket.socket() as occupied:
        occupied.bind(("127.0.0.1", 0))
        occupied.listen()
        with pytest.raises(BackendError, match="port is unavailable"):
            backend._load_subprocess("first.gguf", server_port=occupied.getsockname()[1])
    launch.assert_not_called()
    assert not backend._server_urls


@pytest.mark.parametrize("wrong_identity", [False, True])
def test_ephemeral_server_identity_and_cleanup(monkeypatch, wrong_identity):
    backend = LlamaCppBackend()
    monkeypatch.setattr(backend, "_find_llama_binary", lambda: "llama-server")
    requests = []
    processes = []

    def launch(arguments, **kwargs):
        port = int(arguments[arguments.index("--port") + 1])
        alias = arguments[arguments.index("--alias") + 1]

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                requests.append(self.path)
                body = {"data": [{"id": "another-model" if wrong_identity else alias}]}
                encoded = json.dumps(body).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        class Process:
            returncode = None

            def poll(self):
                return self.returncode

            def terminate(self):
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)
                self.returncode = 0

            def wait(self, timeout):
                return self.returncode

        process = Process()
        processes.append(process)
        return process

    monkeypatch.setattr("aether.backends.llamacpp_backend.subprocess.Popen", launch)
    try:
        if wrong_identity:
            with pytest.raises(BackendError, match="identity does not match"):
                backend._load_subprocess("first.gguf")
            assert not backend._server_urls and not backend._server_procs
        else:
            backend._load_subprocess("first.gguf")
            backend._load_subprocess("second.gguf")
            assert backend._server_urls["first.gguf"] != backend._server_urls["second.gguf"]
        assert "/health" in requests and "/v1/models" in requests
    finally:
        backend.shutdown()
    assert all(process.poll() == 0 for process in processes)
