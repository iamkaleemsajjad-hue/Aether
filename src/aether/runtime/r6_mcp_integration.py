"""
R6 — MCP Native Integration Layer.

Model Context Protocol (MCP, Anthropic 2024) is the emerging standard for
connecting LLMs to external tools, databases, and services via a unified
JSON-RPC 2.0 interface.  Aether Runtime R6 provides *native* MCP integration
— tool calls are intercepted at the runtime level, reducing round-trip latency
vs application-layer MCP clients by eliminating an HTTP proxy hop.

Architecture:
  - ``MCPClient``: JSON-RPC 2.0 client that maintains persistent connections
    to MCP servers via stdio / WebSocket / HTTP.
  - ``MCPToolRegistry``: Discovers available tools from connected servers and
    caches their JSON schemas.
  - ``MCPCallInterceptor``: Hooks into the token stream, detects tool call
    JSON patterns, and dispatches to the appropriate MCP server without
    interrupting the generation loop.
  - ``MCPResultInjector``: Injects MCP results back into the context as
    structured tool result tokens.

Tool call detection strategies:
  1. **FSM detection**: Use Pass 11 FSA to detect ``{"tool":`` JSON prefix.
  2. **Token pattern**: Detect ``<tool_call>`` / ``<function_call>`` delimiters.
  3. **LLM signal**: Model emits a special TOOL_CALL token ID (MCP-aware models).

Research basis:
  - MCP Specification (Anthropic 2024): JSON-RPC 2.0 tool protocol.
  - ReAct (Yao et al. 2023): reasoning + acting framework.
  - ToolLLM (2023): tool learning for LLMs.
  - Gorilla (2023): API-calling LLM.
  - AutoGPT / OpenAI Function Calling: production tool use patterns.
"""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import subprocess
import threading
import time
import uuid
import urllib.request
from typing import Any, Callable, Coroutine

from aether.utils.logging import get_logger

logger = get_logger(__name__)

# MCP protocol version supported.
_MCP_PROTOCOL_VERSION = "2025-11-25"

# JSON-RPC error codes.
_ERR_PARSE = -32700
_ERR_INVALID_REQUEST = -32600
_ERR_METHOD_NOT_FOUND = -32601
_ERR_INVALID_PARAMS = -32602
_ERR_INTERNAL = -32603
_ERR_TOOL_NOT_FOUND = -32001
_ERR_TOOL_TIMEOUT = -32002


class _MCPProtocolError(RuntimeError):
    """A remote JSON-RPC error leaves an otherwise live transport usable."""


class MCPToolRegistry:
    """Registry of MCP tools discovered from connected servers."""

    def __init__(self) -> None:
        self._tools: dict[str, dict[str, Any]] = {}  # tool_name → tool_schema
        self._server_map: dict[str, str] = {}  # tool_name → server_id
        self._qualified_tools: dict[tuple[str, str], dict[str, Any]] = {}
        self._lock = threading.RLock()

    def register_server_tools(self, server_id: str, tools: list[dict]) -> int:
        """Register tools from an MCP server's tools/list response.

        Args:
            server_id: Unique server identifier.
            tools: List of tool descriptors from the MCP server.

        Returns:
            Number of tools registered.
        """
        count = 0
        with self._lock:
            self._qualified_tools = {key: tool for key, tool in self._qualified_tools.items() if key[0] != server_id}
            for tool in tools:
                name = tool.get("name", "")
                if not name:
                    continue
                self._qualified_tools[(server_id, name)] = tool
                count += 1
            self._tools.clear()
            self._server_map.clear()
            for (server, name), tool in self._qualified_tools.items():
                owners = [owner for owner, other in self._qualified_tools if other == name]
                if len(owners) == 1:
                    self._tools[name] = tool
                    self._server_map[name] = server
        logger.debug("R6: Registered %d tools from server %r.", count, server_id)
        return count

    def lookup(self, tool_name: str) -> tuple[dict | None, str | None]:
        """Look up a tool schema and its server ID."""
        with self._lock:
            schema = self._tools.get(tool_name)
            server = self._server_map.get(tool_name)
            if schema is not None:
                return schema, server
            # REST and MCP clients commonly address tools as
            # ``server_name/tool_name``. Resolve the qualified form while
            # retaining the server identity for isolation checks.
            if "/" in tool_name:
                server_name, unqualified = tool_name.split("/", 1)
                schema = self._qualified_tools.get((server_name, unqualified))
                if schema is not None:
                    return schema, server_name
            return None, None

    def all_tools(self) -> list[dict]:
        """Return all registered tool schemas."""
        with self._lock:
            return list(self._qualified_tools.values())

    def tool_count(self) -> int:
        with self._lock:
            return len(self._qualified_tools)


class MCPClient:
    """JSON-RPC 2.0 MCP client for a single server connection.

    Supports synchronous call dispatch with async-under-the-hood execution
    via a dedicated event loop thread.
    """

    def __init__(
        self,
        server_id: str,
        transport: str = "stdio",
        endpoint: str | None = None,
        command: str | list[str] | None = None,
        timeout_s: float = 30.0,
    ) -> None:
        self.server_id = server_id
        self.transport = "ws" if transport == "websocket" else transport
        self.endpoint = endpoint
        self.command = command
        self.timeout_s = timeout_s
        self._connected = False
        self._call_count = 0
        self._error_count = 0
        self._pending_calls: dict[str, asyncio.Future] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None
        self._process: subprocess.Popen[str] | None = None
        self._rpc_lock = threading.RLock()
        self._ws_connection: Any = None
        self._session_id: str | None = None
        self._protocol_version = _MCP_PROTOCOL_VERSION

    def connect(self) -> bool:
        """Establish connection to the MCP server.

        Returns True if connection was successful.
        For stdio transport, spawns a subprocess.
        For http/ws transport, opens a persistent connection.
        """
        if self._connected:
            return True
        try:
            if self.transport == "stdio":
                command = self.command or self.server_id
                if not command:
                    raise ValueError("stdio MCP transport requires a server command")
                if isinstance(command, list):
                    if not command or any(not isinstance(item, str) for item in command):
                        raise ValueError("MCP command argv must contain strings")
                    arguments = command
                else:
                    # Windows CreateProcess already parses a quoted command
                    # line; shlex(posix=False) incorrectly retains those quotes.
                    arguments = command if os.name == "nt" else shlex.split(command)
                self._process = subprocess.Popen(
                    arguments, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL, text=True, encoding="utf-8", bufsize=1,
                )
            elif self.transport in {"http", "ws"} and not self.endpoint:
                raise ValueError(f"{self.transport} MCP transport requires an endpoint")
            elif self.transport not in {"http", "ws", "stdio"}:
                raise ValueError(f"Unsupported MCP transport: {self.transport}")

            # Send MCP initialize handshake.
            init_result = self._rpc_call_sync(
                method="initialize",
                params={
                    "protocolVersion": _MCP_PROTOCOL_VERSION,
                    "clientInfo": {"name": "aether-runtime", "version": "4.0"},
                    "capabilities": {},
                },
            )

            self._connected = isinstance(init_result, dict) and isinstance(init_result.get("capabilities"), dict) and isinstance(init_result.get("serverInfo"), dict) and isinstance(init_result.get("protocolVersion"), str)
            if self._connected:
                version = init_result.get("protocolVersion", _MCP_PROTOCOL_VERSION)
                if version not in {"2024-11-05", "2025-03-26", "2025-06-18", "2025-11-25"}:
                    raise ValueError(f"Unsupported MCP protocol version: {version}")
                self._protocol_version = version
                self._notify("notifications/initialized")
                logger.info("R6: Connected to MCP server %r (%s).", self.server_id, self.transport)
            else:
                self.disconnect()
            return self._connected

        except Exception as exc:  # noqa: BLE001
            logger.warning("R6: Failed to connect to %r: %s", self.server_id, exc)
            self.disconnect()
            return False

    def list_tools(self) -> list[dict]:
        """Fetch available tools from the server."""
        result = self._rpc_call_sync("tools/list", params={})
        if result and "tools" in result:
            return result["tools"]
        return []

    def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Call an MCP tool and return its result.

        Args:
            tool_name: Name of the tool to call.
            arguments: Tool arguments as a dict.

        Returns:
            Tool result dict with ``content`` key.
        """
        start = time.perf_counter()
        result = self._rpc_call_sync(
            method="tools/call",
            params={"name": tool_name, "arguments": arguments},
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        self._call_count += 1

        if result is None:
            self._error_count += 1
            return {"content": [{"type": "text", "text": f"Error: tool call failed"}], "isError": True}

        logger.debug("R6: Tool %r completed in %.1f ms.", tool_name, elapsed_ms)
        return result

    def _rpc_call_sync(self, method: str, params: dict) -> dict | None:
        """Execute a JSON-RPC call synchronously (blocks up to timeout_s).

        The request is sent over the configured real transport.  There is no
        local-success simulation: absent or malformed servers return an error.
        """
        with self._rpc_lock:
            try:
                return self._rpc_exchange(method, params)
            except _MCPProtocolError:
                raise
            except Exception:
                # EOF, broken pipes and closed sockets invalidate connection
                # state. Reconnecting must create a fresh initialized transport.
                try:
                    self.disconnect()
                except Exception as cleanup_error:  # noqa: BLE001
                    logger.warning("R6: Failed to clean up disconnected transport: %s", cleanup_error)
                raise

    def _http_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream", "MCP-Protocol-Version": self._protocol_version}
        if self._session_id:
            headers["MCP-Session-Id"] = self._session_id
        return headers

    def _notify(self, method: str) -> None:
        payload = {"jsonrpc": "2.0", "method": method}
        with self._rpc_lock:
            if self.transport == "stdio":
                if self._process is None or self._process.stdin is None:
                    raise RuntimeError("MCP stdio client is not connected")
                self._process.stdin.write(json.dumps(payload) + "\n")
                self._process.stdin.flush()
            elif self.transport == "http":
                request = urllib.request.Request(self.endpoint or "", data=json.dumps(payload).encode("utf-8"), headers=self._http_headers(), method="POST")
                with urllib.request.urlopen(request, timeout=self.timeout_s):
                    pass
            elif self.transport == "ws":
                self._ws_connection.send(json.dumps(payload))

    @staticmethod
    def _matching_response(body: Any, call_id: str) -> bool:
        return isinstance(body, dict) and body.get("id") == call_id and ("result" in body or "error" in body)

    def _rpc_exchange(self, method: str, params: dict) -> dict | None:
        call_id = str(uuid.uuid4())
        payload = {
            "jsonrpc": "2.0",
            "id": call_id,
            "method": method,
            "params": params,
        }

        if self.transport == "http":
            request = urllib.request.Request(
                self.endpoint or "", data=json.dumps(payload).encode("utf-8"),
                headers=self._http_headers(), method="POST",
            )
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                session_id = response.headers.get("MCP-Session-Id")
                if session_id:
                    self._session_id = session_id
                if "text/event-stream" in response.headers.get("Content-Type", ""):
                    body = None
                    deadline = time.monotonic() + self.timeout_s
                    for line in response:
                        if time.monotonic() > deadline:
                            raise TimeoutError("MCP HTTP event stream timed out")
                        line = line.decode("utf-8").strip()
                        if line.startswith("data:") and line[5:].strip():
                            candidate = json.loads(line[5:].strip())
                            if self._matching_response(candidate, call_id):
                                body = candidate
                                break
                    if body is None:
                        raise RuntimeError("MCP HTTP stream ended without a matching response")
                else:
                    body = json.loads(response.read().decode("utf-8"))
        elif self.transport == "ws":
            try:
                import websocket  # type: ignore[import]
            except ImportError as exc:
                raise RuntimeError("WebSocket MCP transport requires websocket-client") from exc
            if self._ws_connection is None:
                self._ws_connection = websocket.create_connection(self.endpoint or "", timeout=self.timeout_s)
            self._ws_connection.send(json.dumps(payload))
            deadline = time.monotonic() + self.timeout_s
            while True:
                body = json.loads(self._ws_connection.recv())
                if self._matching_response(body, call_id):
                    break
                if time.monotonic() > deadline:
                    raise TimeoutError("MCP WebSocket response timed out")
        elif self.transport == "stdio":
            process = self._process
            if process is None or process.stdin is None or process.stdout is None:
                raise RuntimeError("MCP stdio client is not connected")
            process.stdin.write(json.dumps(payload) + "\n")
            process.stdin.flush()
            result_holder: list[Any] = []
            errors: list[Exception] = []

            def read_response() -> None:
                try:
                    while True:
                        line = process.stdout.readline()
                        if not line:
                            raise EOFError("MCP server closed stdout")
                        candidate = json.loads(line)
                        if self._matching_response(candidate, call_id):
                            result_holder.append(candidate)
                            return
                        if "method" in candidate and "id" in candidate:
                            reply = {"jsonrpc": "2.0", "id": candidate["id"], "error": {"code": _ERR_METHOD_NOT_FOUND, "message": "Client callbacks are not supported"}}
                            process.stdin.write(json.dumps(reply) + "\n")
                            process.stdin.flush()
                except Exception as exc:
                    errors.append(exc)

            reader = threading.Thread(target=read_response, daemon=True)
            reader.start()
            reader.join(self.timeout_s)
            if reader.is_alive():
                # A blocked reader cannot safely share stdout with the next
                # call. Terminate the transport and reap it before returning.
                self.disconnect()
                reader.join(timeout=1)
                raise TimeoutError(f"MCP server {self.server_id!r} did not answer within {self.timeout_s}s")
            if errors:
                raise errors[0]
            body = result_holder[0]
        else:
            raise ValueError(f"Unsupported MCP transport: {self.transport}")

        if not self._matching_response(body, call_id):
            raise RuntimeError("MCP response has no matching JSON-RPC request id")
        if "error" in body:
            raise _MCPProtocolError(f"MCP JSON-RPC error: {body['error']}")
        return body.get("result")

    def disconnect(self) -> None:
        """Disconnect from the MCP server."""
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        self._connected = False
        if self._process is not None:
            process = self._process
            self._process = None
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=1)
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream is not None:
                    stream.close()
        if self._ws_connection is not None:
            self._ws_connection.close()
            self._ws_connection = None
        if self._loop_thread is not None:
            self._loop_thread.join(timeout=1)
            self._loop_thread = None
        if self._loop is not None and not self._loop.is_running():
            self._loop.close()
            self._loop = None
        self._session_id = None

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def stats(self) -> dict[str, int]:
        return {"calls": self._call_count, "errors": self._error_count}


class MCPIntegrationLayer:
    """Runtime R6: MCP Native Integration Layer.

    Manages multiple MCP server connections, discovers tools, and provides
    a unified tool dispatch interface for the inference loop.
    """

    def __init__(
        self,
        timeout_s: float = 30.0,
        max_concurrent_calls: int = 16,
    ) -> None:
        self.timeout_s = timeout_s
        self.max_concurrent_calls = max_concurrent_calls
        self._clients: dict[str, MCPClient] = {}
        self._registry = MCPToolRegistry()
        self._semaphore = threading.Semaphore(max_concurrent_calls)
        self._lock = threading.RLock()
        self._total_calls = 0
        self._total_errors = 0

    def add_server(
        self,
        server_id: str,
        transport: str = "stdio",
        endpoint: str | None = None,
        command: str | list[str] | None = None,
    ) -> bool:
        """Connect to an MCP server and register its tools.

        Args:
            server_id: Unique name for this server.
            transport: "stdio" | "http" | "ws".
            endpoint: Server URL for http/ws transport.

        Returns:
            True if connected and tools registered successfully.
        """
        client = MCPClient(
            server_id=server_id,
            transport=transport,
            endpoint=endpoint,
            command=command,
            timeout_s=self.timeout_s,
        )
        if not client.connect():
            return False

        try:
            tools = client.list_tools()
        except Exception:
            client.disconnect()
            raise

        with self._lock:
            old = self._clients.get(server_id)
            if old is not None:
                old.disconnect()
            self._registry.register_server_tools(server_id, tools)
            self._clients[server_id] = client

        logger.info(
            "R6: Server %r connected — %d tools available.", server_id, len(tools)
        )
        return True

    def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Dispatch a tool call to the appropriate MCP server.

        Args:
            tool_name: Tool name as registered in the registry.
            arguments: Tool arguments.

        Returns:
            Tool result dict.
        """
        schema, server_id = self._registry.lookup(tool_name)
        if server_id is None:
            logger.warning("R6: Tool %r not found in registry.", tool_name)
            self._total_errors += 1
            return {
                "content": [{"type": "text", "text": f"Tool {tool_name!r} not found."}],
                "isError": True,
            }

        input_schema = None
        if isinstance(schema, dict):
            input_schema = (
                schema.get("inputSchema")
                or schema.get("input_schema")
                or schema.get("parameters")
            )
        if input_schema is not None:
            if not isinstance(input_schema, dict):
                self._total_errors += 1
                return {
                    "content": [{"type": "text", "text": "MCP tool schema is malformed."}],
                    "isError": True,
                }
            try:
                from jsonschema import Draft202012Validator

                errors = sorted(Draft202012Validator(input_schema).iter_errors(arguments), key=str)
            except Exception as exc:  # noqa: BLE001 - schema errors fail closed
                self._total_errors += 1
                return {
                    "content": [{"type": "text", "text": f"MCP tool arguments failed schema validation: {exc}"}],
                    "isError": True,
                }
            if errors:
                self._total_errors += 1
                return {
                    "content": [{"type": "text", "text": f"MCP tool arguments failed schema validation: {errors[0].message}"}],
                    "isError": True,
                }

        with self._lock:
            client = self._clients.get(server_id)
        if client is None or not client.is_connected:
            self._total_errors += 1
            return {
                "content": [{"type": "text", "text": f"Server {server_id!r} not connected."}],
                "isError": True,
            }

        self._semaphore.acquire()
        try:
            # The registry namespace belongs to Aether. The server expects its
            # advertised tool name, without an added "server/" prefix.
            server_tool_name = schema.get("name", tool_name) if isinstance(schema, dict) else tool_name
            result = client.call_tool(server_tool_name, arguments)
            self._total_calls += 1
            return result
        finally:
            self._semaphore.release()

    def detect_tool_call(self, token_stream: str) -> dict[str, Any] | None:
        """Detect and parse a tool call from the generated token stream.

        Supports three detection strategies:
          1. JSON pattern: detect ``{"tool": "name", "arguments": {...}}``
          2. XML tags: detect ``<tool_call>...</tool_call>``
          3. Special token: detect ``[TOOL_CALL]`` marker.

        Returns:
            Parsed tool call dict or None if no tool call detected.
        """
        import re

        # Strategy 1: JSON pattern detection.
        json_match = re.search(
            r'\{[^}]*"tool"\s*:\s*"([^"]+)"[^}]*"arguments"\s*:\s*(\{[^}]*\})',
            token_stream,
            re.DOTALL,
        )
        if json_match:
            tool_name = json_match.group(1)
            try:
                arguments = json.loads(json_match.group(2))
                return {"tool": tool_name, "arguments": arguments, "detection": "json_pattern"}
            except json.JSONDecodeError:
                pass

        # Strategy 2: XML-style tool call tags.
        xml_match = re.search(
            r"<tool_call>(.*?)</tool_call>",
            token_stream,
            re.DOTALL,
        )
        if xml_match:
            try:
                payload = json.loads(xml_match.group(1))
                return {"tool": payload.get("name", ""), "arguments": payload.get("arguments", {}), "detection": "xml_tag"}
            except (json.JSONDecodeError, AttributeError):
                pass

        # Strategy 3: Function call notation.
        func_match = re.search(
            r'<function_calls>\s*<invoke name="([^"]+)">(.*?)</invoke>',
            token_stream,
            re.DOTALL,
        )
        if func_match:
            tool_name = func_match.group(1)
            try:
                args_xml = func_match.group(2)
                # Simple parameter extraction.
                params = {}
                for pm in re.finditer(r'<parameter name="([^"]+)">(.*?)</parameter>', args_xml, re.DOTALL):
                    params[pm.group(1)] = pm.group(2).strip()
                return {"tool": tool_name, "arguments": params, "detection": "function_call_xml"}
            except Exception:  # noqa: BLE001
                pass

        return None

    @property
    def tool_count(self) -> int:
        return self._registry.tool_count()

    def list_tools(self) -> list[dict[str, Any]]:
        """Return the authenticated, schema-bearing tools visible to R6."""
        return self._registry.all_tools()

    @property
    def connected_servers(self) -> list[str]:
        with self._lock:
            return [sid for sid, c in self._clients.items() if c.is_connected]

    def disconnect_all(self) -> None:
        """Disconnect all MCP servers."""
        with self._lock:
            for client in self._clients.values():
                client.disconnect()
            self._clients.clear()
        logger.info("R6: All MCP servers disconnected.")

    def summary(self) -> dict[str, Any]:
        return {
            "connected_servers": self.connected_servers,
            "total_tools": self.tool_count,
            "total_calls": self._total_calls,
            "total_errors": self._total_errors,
        }
