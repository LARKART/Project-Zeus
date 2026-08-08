"""A minimal MCP client speaking JSON-RPC 2.0 over a subprocess's stdio.

WHY NOT THE OFFICIAL SDK. Global Constraints forbid new third-party
dependencies, and stdio MCP is a small protocol: newline-delimited JSON-RPC,
an `initialize` handshake, `tools/list`, `tools/call`. That is what this
file is. It deliberately implements no resources, prompts, sampling or
notifications -- ZEUS calls tools, and a client that pretended to more
surface than it supports would be a lie in the codebase.

EVERY CALL IS BOUNDED. This is the lesson the voice loop already paid for
four times: a read with no deadline is a daemon that stops for good. An MCP
server is an arbitrary third-party process that can hang, wedge on network
I/O, or exit mid-request, and it is called from the middle of a
conversation -- so every read here carries a timeout, and a timeout kills
the process rather than leaving it parked holding a pipe.

STDOUT IS THE PROTOCOL; STDERR IS NOISE. Servers log freely to stderr, and
mixing the two corrupts the stream, so stderr is drained on its own thread
into the log and never parsed.
"""
from __future__ import annotations

import json
import logging
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)

PROTOCOL_VERSION = "2025-06-18"
CLIENT_INFO = {"name": "zeus", "version": "0.1.0"}

# How long any single request may take. Generous because a first `npx` run
# downloads the server package, and mean because the caller is a voice
# conversation with a human waiting through it.
DEFAULT_TIMEOUT = 30.0
# The handshake gets its own, larger budget for exactly that reason.
STARTUP_TIMEOUT = 90.0


class MCPError(RuntimeError):
    """A server failed to start, answer, or answer sanely."""


@dataclass(frozen=True)
class MCPTool:
    """One tool a server offers, already in the shape the Anthropic API wants."""

    server: str
    name: str
    description: str
    input_schema: dict[str, Any]

    @property
    def qualified_name(self) -> str:
        """`gmail__search` — namespaced so two servers may both offer `search`.

        Double underscore, not a dot: the Messages API restricts tool names
        to [a-zA-Z0-9_-], so a dotted name is rejected outright.
        """
        return f"{self.server}__{self.name}"


class MCPServer:
    """One MCP server subprocess, and the tools it offers."""

    def __init__(self, name: str, command: list[str], env: dict | None = None,
                 timeout: float = DEFAULT_TIMEOUT) -> None:
        self.name = name
        self._command = command
        self._env = env
        self._timeout = timeout
        self._process: subprocess.Popen | None = None
        self._next_id = 0
        # One lock around the whole request/response pair. The pipe is a
        # single stream with no multiplexing: two threads interleaving
        # writes, or both reading, would hand each other the wrong reply.
        # ZEUS reaches this from the scheduler thread and the wake thread.
        self._lock = threading.Lock()
        self.tools: list[MCPTool] = []

    # -- lifecycle -------------------------------------------------------
    def start(self) -> None:
        import os

        environment = {**os.environ, **(self._env or {})}
        try:
            self._process = subprocess.Popen(
                self._command,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, env=environment, text=True, bufsize=1,
            )
        except (OSError, ValueError) as problem:
            raise MCPError(f"could not start {self.name}: {problem}") from problem

        threading.Thread(target=self._drain_stderr, daemon=True,
                         name=f"mcp-{self.name}-stderr").start()
        self._handshake()
        self.tools = self._discover()
        log.info("mcp: %s ready with %d tool(s): %s", self.name, len(self.tools),
                 ", ".join(t.name for t in self.tools) or "none")

    def stop(self) -> None:
        process = self._process
        if process is None:
            return
        self._process = None
        for step in (process.terminate, process.kill):
            try:
                step()
                process.wait(timeout=5)
                return
            except subprocess.TimeoutExpired:
                continue          # escalate to kill
            except Exception:
                return

    def _drain_stderr(self) -> None:
        process = self._process
        if process is None or process.stderr is None:
            return
        try:
            for line in process.stderr:
                if line.strip():
                    log.debug("mcp[%s] %s", self.name, line.rstrip())
        except Exception:
            pass

    # -- protocol --------------------------------------------------------
    def _handshake(self) -> None:
        self._request(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": CLIENT_INFO,
            },
            timeout=STARTUP_TIMEOUT,
        )
        # A NOTIFICATION -- no id, and no reply is coming. Waiting for one
        # would hang the handshake until the timeout on every start.
        self._notify("notifications/initialized")

    def _discover(self) -> list[MCPTool]:
        result = self._request("tools/list", {}, timeout=STARTUP_TIMEOUT)
        tools = []
        for entry in result.get("tools", []):
            name = entry.get("name")
            if not name:
                continue
            tools.append(MCPTool(
                server=self.name, name=name,
                description=entry.get("description", "") or f"{name} via {self.name}",
                # `inputSchema` is MCP's spelling; the Messages API wants
                # `input_schema`. Defaulted to a valid empty object schema
                # because the API rejects a tool with no schema at all.
                input_schema=entry.get("inputSchema")
                or {"type": "object", "properties": {}},
            ))
        return tools

    def call(self, tool: str, arguments: dict) -> str:
        """Invoke a tool and flatten its content blocks to text."""
        result = self._request(
            "tools/call", {"name": tool, "arguments": arguments or {}}
        )
        parts = []
        for block in result.get("content", []):
            if block.get("type") == "text":
                parts.append(block.get("text", ""))
            else:
                # Images and embedded resources are real MCP block types that
                # ZEUS has no voice for. Naming them beats dropping them
                # silently, which would read to the model as an empty result.
                parts.append(f"[{block.get('type', 'unknown')} omitted]")
        text = "\n".join(p for p in parts if p)
        if result.get("isError"):
            raise MCPError(text or f"{tool} failed")
        return text or "(the tool returned no output)"

    # -- transport -------------------------------------------------------
    def _write(self, payload: dict) -> None:
        process = self._process
        if process is None or process.stdin is None:
            raise MCPError(f"{self.name} is not running")
        try:
            process.stdin.write(json.dumps(payload) + "\n")
            process.stdin.flush()
        except (BrokenPipeError, ValueError, OSError) as problem:
            raise MCPError(f"{self.name} closed its input: {problem}") from problem

    def _notify(self, method: str, params: dict | None = None) -> None:
        with self._lock:
            self._write({"jsonrpc": "2.0", "method": method,
                         "params": params or {}})

    def _request(self, method: str, params: dict,
                 timeout: float | None = None) -> dict:
        budget = timeout or self._timeout
        with self._lock:
            self._next_id += 1
            request_id = self._next_id
            self._write({"jsonrpc": "2.0", "id": request_id,
                         "method": method, "params": params})
            message = self._read_until(request_id, budget)
        if "error" in message:
            error = message["error"]
            raise MCPError(
                f"{self.name}.{method}: {error.get('message', error)}"
            )
        return message.get("result", {})

    def _read_until(self, request_id: int, budget: float) -> dict:
        """Read replies until the one we asked for arrives, or time runs out.

        Skipping rather than failing on a non-matching message is required,
        not defensive: a server may legitimately emit notifications (progress,
        logging) between the request and its answer, and a client that treated
        the first line as the reply would mis-read every one of them.
        """
        process = self._process
        if process is None or process.stdout is None:
            raise MCPError(f"{self.name} is not running")
        deadline = time.monotonic() + budget
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.stop()
                raise MCPError(
                    f"{self.name} did not answer within {budget:.0f}s; "
                    f"the server was stopped"
                )
            line = self._readline_with_deadline(process, remaining)
            if line is None:
                self.stop()
                raise MCPError(f"{self.name} exited without answering")
            line = line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except ValueError:
                # Not protocol. Some servers print a banner to stdout despite
                # the spec; log it and keep reading rather than dying on it.
                log.debug("mcp[%s] non-JSON on stdout: %s", self.name, line[:200])
                continue
            if message.get("id") == request_id:
                return message

    @staticmethod
    def _readline_with_deadline(process, remaining: float) -> str | None:
        """A readline that cannot outlive the budget.

        `process.stdout.readline()` blocks forever on a server that has
        stopped talking but not exited -- the failure this whole file is
        shaped around. The read happens on a throwaway thread so the caller
        keeps its deadline; if it times out the caller kills the process,
        which is what unblocks the orphan.
        """
        holder: list[str] = []

        def read() -> None:
            try:
                holder.append(process.stdout.readline())
            except Exception:
                pass

        worker = threading.Thread(target=read, daemon=True)
        worker.start()
        worker.join(timeout=remaining)
        if worker.is_alive():
            return ""             # still blocked: let the caller re-check its deadline
        if not holder:
            return None           # the read raised: treat as gone
        return holder[0] or None  # readline() returns "" at EOF
