"""Connects to MCP servers (stdio or HTTP, with OAuth) and exposes their tools to the LLM.

mcp_servers.json:
{
  "servers": {
    "stockpot": {"type": "http", "url": "https://stockpot.xxx.workers.dev/mcp", "oauth": true},
    "other":    {"type": "http", "url": "http://...", "headers": {"Authorization": "Bearer ..."}},
    "local":    {"type": "stdio", "command": "uvx", "args": ["my-mcp-server"], "env": {}}
  }
}

Tool results are read by the model, so the messages added around them stay in French.
"""

from __future__ import annotations

import asyncio
import json
import threading
import webbrowser
from contextlib import AsyncExitStack
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from mcp import ClientSession, StdioServerParameters, stdio_client
from mcp.client.auth import OAuthClientProvider, TokenStorage
from mcp.client.streamable_http import streamable_http_client
from mcp.shared._httpx_utils import create_mcp_http_client
from mcp.shared.auth import (
    AuthorizationCodeResult,
    OAuthClientInformationFull,
    OAuthClientMetadata,
    OAuthToken,
)

from . import web
from .aio import in_thread
from .config import Settings


class FileTokenStorage(TokenStorage):
    """Keeps OAuth tokens on disk: log in once on the PC, then copy the folder to the box."""

    def __init__(self, path: Path):
        self.path = path

    def _read(self) -> dict:
        return json.loads(self.path.read_text("utf-8")) if self.path.exists() else {}

    def _write(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=2), "utf-8")

    async def get_tokens(self) -> OAuthToken | None:
        raw = self._read().get("tokens")
        return OAuthToken.model_validate(raw) if raw else None

    async def set_tokens(self, tokens: OAuthToken) -> None:
        data = self._read()
        data["tokens"] = tokens.model_dump(mode="json", exclude_none=True)
        self._write(data)

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        raw = self._read().get("client_info")
        return OAuthClientInformationFull.model_validate(raw) if raw else None

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        data = self._read()
        data["client_info"] = client_info.model_dump(mode="json", exclude_none=True)
        self._write(data)


def _wait_for_oauth_callback(port: int, timeout: float = 300) -> AuthorizationCodeResult:
    """Small local HTTP server that catches the OAuth redirect (?code=...&state=...)."""
    result: dict[str, str] = {}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            query = parse_qs(urlparse(self.path).query)
            if "code" in query:
                result.update({k: v[0] for k, v in query.items()})
                body = "Logged in, you can close this tab."
            else:
                body = "No authorization code in the redirect."
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(body.encode())

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", port), Handler)
    server.timeout = 1
    deadline = threading.Event()
    threading.Timer(timeout, deadline.set).start()
    while "code" not in result and not deadline.is_set():
        server.handle_request()
    server.server_close()
    if "code" not in result:
        raise TimeoutError("No OAuth authorization received")
    return AuthorizationCodeResult(code=result["code"], state=result.get("state"), iss=result.get("iss"))


def _oauth_provider(name: str, url: str, settings: Settings) -> OAuthClientProvider:
    redirect_uri = f"http://localhost:{settings.oauth_callback_port}/callback"

    async def redirect_handler(auth_url: str) -> None:
        print(f'\n🔐 Authorize access to "{name}" in your browser:\n{auth_url}\n')
        webbrowser.open(auth_url)

    async def callback_handler() -> AuthorizationCodeResult:
        return await in_thread(_wait_for_oauth_callback, settings.oauth_callback_port)

    return OAuthClientProvider(
        server_url=url,
        client_metadata=OAuthClientMetadata(
            client_name="Prométhée",
            redirect_uris=[redirect_uri],
            grant_types=["authorization_code", "refresh_token"],
            response_types=["code"],
            token_endpoint_auth_method="none",
        ),
        storage=FileTokenStorage(settings.token_dir / f"{name}.json"),
        redirect_handler=redirect_handler,
        callback_handler=callback_handler,
    )


@dataclass
class _ToolRef:
    server: str
    session: ClientSession
    name: str  # original name on the server side


class McpHub:
    """Opens every MCP session at startup and keeps them open."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._stack = AsyncExitStack()
        self._tools: dict[str, _ToolRef] = {}
        self.ollama_tools: list[dict] = []
        self.instructions: dict[str, str] = {}
        self._local: set[str] = set()  # built-in tools (web search…), called without MCP

    def _load_config(self) -> dict[str, dict]:
        path = self.settings.mcp_config
        if not path.exists():
            print(f"⚠️  No {path.name}: running without tools.")
            return {}
        return json.loads(path.read_text("utf-8")).get("servers", {})

    async def _open(self, name: str, cfg: dict, stack: AsyncExitStack) -> ClientSession:
        kind = cfg.get("type", "stdio")
        if kind == "stdio":
            params = StdioServerParameters(
                command=cfg["command"], args=cfg.get("args", []), env=cfg.get("env") or None
            )
            read, write = await stack.enter_async_context(stdio_client(params))
        elif kind == "http":
            auth = _oauth_provider(name, cfg["url"], self.settings) if cfg.get("oauth") else None
            client = create_mcp_http_client(headers=cfg.get("headers"), auth=auth)
            await stack.enter_async_context(client)
            read, write = await stack.enter_async_context(
                streamable_http_client(cfg["url"], http_client=client)
            )
        else:
            raise ValueError(f"Unknown MCP server type: {kind}")
        session = await stack.enter_async_context(ClientSession(read, write, read_timeout_seconds=30))
        init = await session.initialize()
        if init.instructions:
            self.instructions[name] = init.instructions
        return session

    async def _connect(self, name: str, cfg: dict):
        """Isolated connection: if it fails, this server is closed cleanly without touching the others."""
        stack = AsyncExitStack()
        try:
            session = await self._open(name, cfg, stack)
            tools = (await session.list_tools()).tools
        except BaseException as e:  # anyio may raise an ExceptionGroup or a cancellation
            try:
                await stack.aclose()
            except BaseException:
                pass
            if isinstance(e, KeyboardInterrupt):
                raise
            cause = e.exceptions[0] if isinstance(e, BaseExceptionGroup) else e
            print(f'⚠️  MCP "{name}" unavailable: {cause!r}')
            return None
        await self._stack.enter_async_context(stack)
        return session, tools

    async def __aenter__(self) -> McpHub:
        if self.settings.web_search:
            for tool in web.TOOLS:
                self._local.add(tool["function"]["name"])
                self.ollama_tools.append(tool)
            print(f"🌐 Web search: {', '.join(sorted(self._local))}")
        for name, cfg in self._load_config().items():
            if cfg.get("enabled", True) is False:
                continue
            connected = await self._connect(name, cfg)
            if connected is None:
                continue
            session, tools = connected
            for tool in tools:
                taken = tool.name in self._tools or tool.name in self._local
                exposed = f"{name}__{tool.name}" if taken else tool.name
                self._tools[exposed] = _ToolRef(name, session, tool.name)
                self.ollama_tools.append({
                    "type": "function",
                    "function": {
                        "name": exposed,
                        "description": tool.description or "",
                        "parameters": tool.input_schema,
                    },
                })
            print(f'🔌 MCP "{name}": {len(tools)} tools ({", ".join(t.name for t in tools)})')
        return self

    async def __aexit__(self, *exc) -> None:
        await self._stack.aclose()

    async def call(self, name: str, arguments: dict[str, Any] | None) -> str:
        if name in self._local:
            return await web.call(name, arguments or {})
        ref = self._tools.get(name)
        if ref is None:
            return json.dumps({"error": f"Outil inconnu : {name}"}, ensure_ascii=False)
        try:
            result = await ref.session.call_tool(ref.name, arguments or {})
        except Exception as e:
            return json.dumps({"error": repr(e)})
        parts = [c.text for c in result.content if getattr(c, "type", None) == "text"]
        if not parts and result.structured_content is not None:
            parts = [json.dumps(result.structured_content, ensure_ascii=False)]
        text = "\n".join(parts) or "(aucun contenu)"
        return f"ERREUR : {text}" if result.is_error else text
