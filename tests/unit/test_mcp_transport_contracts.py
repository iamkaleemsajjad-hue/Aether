"""Actual local MCP transports, quoted commands, and failed lifecycle cleanup."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from aether.runtime.r6_mcp_integration import MCPClient, MCPIntegrationLayer, MCPToolRegistry


SERVER = r'''
import json, sys
initialized = False
for line in sys.stdin:
    message = json.loads(line)
    method = message.get("method")
    if method == "notifications/initialized":
        initialized = True
        continue
    if method == "initialize":
        result = {"protocolVersion": message["params"]["protocolVersion"], "capabilities": {"tools": {}}, "serverInfo": {"name": "local-regression", "version": "1"}}
    elif not initialized:
        raise RuntimeError("missing initialized notification")
    elif method == "tools/list":
        result = {"tools": [{"name": "echo", "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}]}
    elif method == "tools/call":
        result = {"content": [{"type": "text", "text": message["params"]["arguments"]["text"]}]}
    else:
        result = {}
    print(json.dumps({"jsonrpc": "2.0", "method": "notifications/message", "params": {"level": "info", "data": "real local process"}}), flush=True)
    print(json.dumps({"jsonrpc": "2.0", "id": message["id"], "result": result}), flush=True)
'''


@pytest.mark.parametrize("as_string", [False, True])
def test_real_stdio_handles_quoted_paths_notifications_and_concurrent_calls(tmp_path, as_string):
    script = tmp_path / "server with spaces.py"
    script.write_text(SERVER, encoding="utf-8")
    argv = [sys.executable, "-u", str(script)]
    command = subprocess.list2cmdline(argv) if as_string else argv
    client = MCPClient("local", command=command, timeout_s=3)
    try:
        assert client.connect()
        assert client.list_tools()[0]["name"] == "echo"
        process = client._process
        with ThreadPoolExecutor(max_workers=3) as pool:
            replies = list(pool.map(lambda text: client.call_tool("echo", {"text": text})["content"][0]["text"], ["one", "two", "three"]))
        assert replies == ["one", "two", "three"]
        assert client.stats == {"calls": 3, "errors": 0}
        assert client._loop_thread is None
    finally:
        client.disconnect()
    assert not client.is_connected and client._process is None
    assert process.poll() is not None


def test_unresponsive_stdio_is_reaped_without_background_loop(tmp_path):
    script = tmp_path / "no response.py"
    script.write_text("import time; time.sleep(20)", encoding="utf-8")
    client = MCPClient("slow", command=[sys.executable, str(script)], timeout_s=0.1)
    assert not client.connect()
    assert client._process is None and not client.is_connected
    assert client._loop is None and client._loop_thread is None


def test_missing_executable_does_not_claim_connection(tmp_path):
    client = MCPClient("missing", command=[str(tmp_path / "absent.exe")], timeout_s=0.1)
    assert not client.connect()
    assert client._process is None


@pytest.mark.parametrize("sse", [False, True])
def test_real_http_mcp_cli_handshake_session_and_tool(tmp_path, minimal_aeg_package, sse):
    from click.testing import CliRunner
    from aether.cli import cli

    events = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            message = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            events.append(message["method"])
            assert "text/event-stream" in self.headers["Accept"]
            if message["method"] == "initialize":
                result = {"protocolVersion": message["params"]["protocolVersion"], "capabilities": {"tools": {}}, "serverInfo": {"name": "local-http", "version": "1"}}
            else:
                assert self.headers["MCP-Session-Id"] == "local-session"
                if message["method"] == "notifications/initialized":
                    self.send_response(202)
                    self.end_headers()
                    return
                if message["method"] == "tools/list":
                    result = {"tools": [{"name": "echo", "inputSchema": {"type": "object"}}]}
                else:
                    result = {"content": [{"type": "text", "text": message["params"]["arguments"]["text"]}]}
            response = json.dumps({"jsonrpc": "2.0", "id": message["id"], "result": result})
            body = ("data: " + response + "\n\n" if sse else response).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream" if sse else "application/json")
            self.send_header("MCP-Session-Id", "local-session")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = MCPClient("local", "http", f"http://127.0.0.1:{server.server_port}/mcp", timeout_s=3)
    try:
        assert client.connect()
        assert client.call_tool("echo", {"text": "actual local HTTP"})["content"][0]["text"] == "actual local HTTP"
        runner = CliRunner()
        added = runner.invoke(cli, ["mcp", "add", str(minimal_aeg_package.root), "--server", "local", "--transport", "http", "--endpoint", client.endpoint])
        assert added.exit_code == 0, added.output
        reply = runner.invoke(cli, ["mcp", "test", str(minimal_aeg_package.root), "--tool", "local/echo", "--call", '{"text":"CLI"}'])
        assert reply.exit_code == 0, reply.output
        assert "CLI" in reply.output
        assert "notifications/initialized" in events
    finally:
        client.disconnect()
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_websocket_alias_is_normalized_and_invalid_transport_has_no_loop():
    assert MCPClient("alias", "websocket").transport == "ws"
    client = MCPClient("invalid", "unsupported")
    assert not client.connect()
    assert client._loop_thread is None


def test_duplicate_tool_names_require_server_qualification():
    registry = MCPToolRegistry()
    registry.register_server_tools("first", [{"name": "echo", "description": "first"}])
    registry.register_server_tools("second", [{"name": "echo", "description": "second"}])
    assert registry.lookup("echo") == (None, None)
    assert registry.lookup("first/echo")[0]["description"] == "first"
    assert registry.lookup("second/echo")[0]["description"] == "second"
    assert registry.tool_count() == 2


def test_qualified_tools_send_actual_advertised_names_to_real_stdio_servers(tmp_path):
    # A strict server rejects an Aether-local namespace on the MCP wire.
    strict_server = SERVER.replace(
        'elif method == "tools/call":',
        'elif method == "tools/call":\n        assert message["params"]["name"] == "echo", "unknown wire tool name"',
    )
    script = tmp_path / "strict echo server.py"
    script.write_text(strict_server, encoding="utf-8")
    layer = MCPIntegrationLayer(timeout_s=3)
    try:
        for server in ("first", "second"):
            assert layer.add_server(server, command=[sys.executable, "-u", str(script)])
        assert layer.call_tool("echo", {"text": "ambiguous"})["isError"]
        assert layer.call_tool("first/echo", {"text": "first"})["content"][0]["text"] == "first"
        assert layer.call_tool("second/echo", {"text": "second"})["content"][0]["text"] == "second"
        assert layer._clients["first"]._process.pid != layer._clients["second"]._process.pid
    finally:
        for client in layer._clients.values():
            client.disconnect()


def test_real_stdio_eof_invalidates_connection_and_reconnects_fresh(tmp_path):
    script = tmp_path / "close on tool call.py"
    closing_server = SERVER.replace(
        'elif method == "tools/call":',
        'elif method == "tools/call":\n        raise SystemExit(0)',
    )
    script.write_text(closing_server, encoding="utf-8")
    client = MCPClient("eof", command=[sys.executable, "-u", str(script)], timeout_s=3)
    try:
        assert client.connect()
        first_process = client._process
        with pytest.raises(EOFError, match="closed stdout"):
            client.call_tool("echo", {"text": "close"})
        assert not client.is_connected
        assert client._process is None
        assert first_process.poll() is not None
        assert client.connect()
        assert client._process.pid != first_process.pid
        assert client.list_tools()[0]["name"] == "echo"
    finally:
        client.disconnect()

