from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class McpServerSpec:
    command: str
    args: list[str]


class McpToolClient:
    """Thin async stdio MCP adapter.

    This is intentionally a boundary class. In production, replace the stub
    methods with official `mcp` session calls and keep the orchestrator stable.
    """

    def __init__(self, servers: dict[str, McpServerSpec]):
        self.servers = servers
        self._started = False

    async def start(self) -> None:
        # Placeholder for mcp.client.stdio.stdio_client + ClientSession setup.
        self._started = True

    async def close(self) -> None:
        self._started = False

    async def read_file(self, path: str) -> str:
        self._require_started()
        # Development fallback. Swap for filesystem MCP `read_file`.
        return await asyncio.to_thread(Path(path).read_text)

    async def write_file(self, path: str, content: str) -> None:
        self._require_started()
        # Development fallback. Swap for filesystem MCP `write_file`.
        await asyncio.to_thread(Path(path).write_text, content)

    async def run_command(self, command: str, cwd: str | None = None) -> dict[str, Any]:
        self._require_started()
        proc = await asyncio.create_subprocess_shell(
            command,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        return {
            "command": command,
            "exit_code": proc.returncode,
            "stdout": stdout.decode(errors="replace"),
            "stderr": stderr.decode(errors="replace"),
        }

    def _require_started(self) -> None:
        if not self._started:
            raise RuntimeError("MCP client is not started")
