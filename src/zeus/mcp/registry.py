"""Turn configured MCP servers into tools the brain can call.

This is the seam where ZEUS stops being a check-in daemon and starts being
able to act: email, files, the browser, anything with an MCP server. The
brain already runs an Anthropic tool loop, so an MCP tool only has to arrive
in the same shape as `save_goal` -- a name, a JSON schema, and a callable.

THE DESTRUCTIVE GATE LIVES HERE, and this is the moment the spec named for
it. §12 says any tool marked destructive must be stated aloud and confirmed
by voice before it runs; that clause was amended on 2026-08-07 to build the
gate in Slice 2 "beside the first tool that needs it", because a permission
mechanism with zero callers is hard to design well. MCP is the first tool
that needs it -- `save_goal` cannot delete your mail.

FAILURE IS PER-SERVER, NOT GLOBAL. One server that will not start must not
cost ZEUS the others, or the ritual, or the daemon. A broken Gmail server
means no mail tools and a loud log line; it does not mean ZEUS stops asking
what your one thing is.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from zeus.mcp.client import MCPError, MCPServer, MCPTool

log = logging.getLogger(__name__)

# Tools whose names suggest they change or destroy something the user cares
# about. Matched on the tool NAME, which is the only thing every MCP server
# is guaranteed to give us -- descriptions are free text and often absent.
#
# WHY A DENYLIST OF VERBS AND NOT A SERVER ALLOWLIST: the model picks the
# tool, so the question is never "is this server trusted" but "is this
# particular call going to be hard to undo". Erring wide is the right
# direction -- a needless confirmation costs three seconds, a silent
# `delete_all_mail` costs the mail.
DESTRUCTIVE_PATTERN = re.compile(
    r"(?:^|_)(delete|remove|rm|drop|destroy|purge|erase|trash|wipe|"
    r"send|post|publish|create|write|update|edit|modify|move|rename|"
    r"install|uninstall|execute|run|kill|shutdown|restart|revoke|"
    r"transfer|pay|buy|order|cancel|archive|reply|forward)(?:$|_)",
    re.IGNORECASE,
)


def looks_destructive(tool_name: str) -> bool:
    """Would running this without asking be hard to take back?"""
    return bool(DESTRUCTIVE_PATTERN.search(tool_name))


@dataclass
class ServerConfig:
    name: str
    command: list[str]
    env: dict[str, str] = field(default_factory=dict)
    enabled: bool = True


class Confirmer:
    """Asks the user, out loud, before a destructive tool runs.

    Speaking the ACTION rather than the tool name is the point: "Send an
    email to sam@example.com?" is answerable, "Run gmail__send_message?" is
    not, and a question the user cannot evaluate is not consent.
    """

    AFFIRMATIVE = {"yes", "yeah", "yep", "yup", "sure", "ok", "okay",
                   "go ahead", "do it", "confirm", "confirmed", "please do"}

    def __init__(self, voice) -> None:
        self._voice = voice

    def confirm(self, tool: MCPTool, arguments: dict) -> bool:
        if self._voice is None:
            # No voice, no consent. Refusing is the only safe default: a
            # scheduled check-in or a headless run has nobody to ask, and
            # "nobody objected" is not the same as "someone agreed".
            log.warning("mcp: refusing %s — no voice channel to confirm with",
                        tool.qualified_name)
            return False
        self._voice.speak([self._question(tool, arguments)])
        answer = (self._voice.listen() or "").strip().lower().rstrip(".!")
        granted = answer in self.AFFIRMATIVE or any(
            answer.startswith(word + " ") for word in self.AFFIRMATIVE
        )
        log.info("mcp: %s %s (heard %r)", tool.qualified_name,
                 "confirmed" if granted else "declined", answer)
        if not granted:
            self._voice.speak(["Okay, I won't."])
        return granted

    @staticmethod
    def _question(tool: MCPTool, arguments: dict) -> str:
        action = tool.name.replace("_", " ")
        # At most two arguments, and short ones. The question has to be
        # listened to, and a spoken sentence carrying a 2,000-character email
        # body is not a question anybody can answer.
        details = []
        for key, value in list(arguments.items())[:2]:
            text = str(value)
            if len(text) > 60:
                text = text[:57] + "…"
            details.append(f"{key.replace('_', ' ')} {text}")
        suffix = f", with {', '.join(details)}" if details else ""
        return f"You want me to {action} using {tool.server}{suffix}. Shall I?"


class MCPRegistry:
    """Every running server, and the tool callables they expose."""

    def __init__(self, confirmer: Confirmer | None = None,
                 store=None) -> None:
        self._servers: dict[str, MCPServer] = {}
        self._tools: dict[str, MCPTool] = {}
        self._confirmer = confirmer
        # EVERY tool call is logged, MCP included. Spec §1 promises the
        # dashboard 'history to display on the day it launches', and a
        # tool surface where two calls are recorded and fourteen are not
        # makes that log actively misleading -- the destructive ones are
        # precisely the ones you would later want to look up.
        self._store = store

    def start(self, configs: list[ServerConfig]) -> None:
        for config in configs:
            if not config.enabled:
                log.info("mcp: %s is disabled; skipping", config.name)
                continue
            server = MCPServer(config.name, config.command, config.env)
            try:
                server.start()
            except MCPError:
                # Per-server, not global. See the module docstring.
                log.error("mcp: %s failed to start; continuing without it",
                          config.name, exc_info=True)
                server.stop()
                continue
            self._servers[config.name] = server
            for tool in server.tools:
                if tool.qualified_name in self._tools:
                    log.warning("mcp: duplicate tool %s; keeping the first",
                                tool.qualified_name)
                    continue
                self._tools[tool.qualified_name] = tool

    def stop(self) -> None:
        for server in self._servers.values():
            server.stop()
        self._servers.clear()
        self._tools.clear()

    @property
    def tools(self) -> list[MCPTool]:
        return list(self._tools.values())

    def definitions(self) -> list[dict[str, Any]]:
        """Anthropic tool definitions for everything discovered."""
        return [
            {
                "name": tool.qualified_name,
                "description": self._describe(tool),
                "input_schema": tool.input_schema,
            }
            for tool in self._tools.values()
        ]

    @staticmethod
    def _describe(tool: MCPTool) -> str:
        note = (
            " Requires spoken confirmation before it runs."
            if looks_destructive(tool.name) else ""
        )
        return f"{tool.description} (via the {tool.server} MCP server).{note}"

    def callables(self) -> dict[str, Callable[..., str]]:
        """Name -> callable, gated where §12 requires it."""
        return {
            name: self._wrap(tool) for name, tool in self._tools.items()
        }

    def beta_tools(self) -> list[Any]:
        """The same tools, as SDK objects the Tool Runner can execute.

        `beta_tool` normally infers a schema from a Python signature, which
        an MCP tool does not have -- its schema arrives as JSON from the
        server. Passing name/description/input_schema explicitly is the
        supported way to bypass inference, so an MCP tool ends up
        indistinguishable from `save_goal` at the point of use.
        """
        from anthropic import beta_tool

        built = []
        for name, tool in self._tools.items():
            built.append(beta_tool(
                self._wrap(tool),
                name=name,
                description=self._describe(tool),
                input_schema=tool.input_schema,
            ))
        return built

    def _log(self, tool: MCPTool, arguments: dict, result, ok: bool,
             started: float, error: str | None) -> None:
        """Mirror an MCP call into the action log, never failing the call.

        A dashboard write that raised would turn a successful tool call into
        a failed one, which is precisely backwards: the log exists to
        describe what happened, not to decide it.
        """
        if self._store is None:
            return
        try:
            self._store.log_action(
                tool.qualified_name, arguments,
                result if result is None else str(result)[:4000], ok,
                int((time.monotonic() - started) * 1000), error,
            )
        except Exception:
            log.debug("mcp: could not write the action log", exc_info=True)

    def _wrap(self, tool: MCPTool) -> Callable[..., str]:
        server = self._servers[tool.server]
        gated = looks_destructive(tool.name)

        def invoke(**arguments: Any) -> str:
            if gated:
                if self._confirmer is None:
                    log.warning("mcp: refusing %s — no confirmer is wired",
                                tool.qualified_name)
                    return (
                        f"{tool.qualified_name} needs spoken confirmation and "
                        f"none is available, so nothing was done."
                    )
                if not self._confirmer.confirm(tool, arguments):
                    # Returned to the MODEL, not raised: a refusal is a
                    # normal outcome the model must be able to talk about,
                    # not an error it should retry around.
                    return (
                        f"The user declined. {tool.qualified_name} was NOT "
                        f"run. Do not try it again this turn."
                    )
            started = time.monotonic()
            try:
                result = server.call(tool.name, arguments)
            except MCPError as problem:
                log.error("mcp: %s failed", tool.qualified_name, exc_info=True)
                self._log(tool, arguments, None, False,
                          started, str(problem))
                return f"{tool.qualified_name} failed: {problem}"
            self._log(tool, arguments, result, True, started, None)
            return result

        invoke.__name__ = tool.qualified_name
        invoke.__doc__ = tool.description
        return invoke


def load_server_configs(raw: dict[str, Any]) -> list[ServerConfig]:
    """Read the `[mcp.servers]` block of config.toml.

        [mcp.servers.filesystem]
        command = ["npx", "-y", "@modelcontextprotocol/server-filesystem",
                   "/Users/you/Documents"]

    A malformed entry is skipped with a loud line rather than raising: this
    is parsed at daemon startup, and a typo here must not become a KeepAlive
    respawn loop -- the failure mode _load_config_or_default already exists
    to prevent.
    """
    configs = []
    for name, entry in (raw or {}).items():
        if not isinstance(entry, dict):
            log.error("mcp: [mcp.servers.%s] is not a table; skipping", name)
            continue
        command = entry.get("command")
        if not isinstance(command, list) or not command or not all(
            isinstance(part, str) for part in command
        ):
            log.error("mcp: [mcp.servers.%s] needs command = [\"...\"]; "
                      "skipping", name)
            continue
        env = entry.get("env") or {}
        if not isinstance(env, dict):
            log.error("mcp: [mcp.servers.%s] env must be a table; ignoring it",
                      name)
            env = {}
        configs.append(ServerConfig(
            name=str(name), command=[str(p) for p in command],
            env={str(k): str(v) for k, v in env.items()},
            enabled=bool(entry.get("enabled", True)),
        ))
    return configs
