"""MCP client, registry, and the §12 destructive gate.

No test here reaches the network or spawns npx. The server double below is a
real subprocess speaking real JSON-RPC over real pipes — it is the protocol
that is under test, not a mock of it — but it is a few lines of Python, so
the suite stays offline and fast (§13).
"""
from __future__ import annotations

import json
import sys
import textwrap

import pytest

from zeus.mcp.client import MCPError, MCPServer
from zeus.mcp.registry import (
    Confirmer, MCPRegistry, ServerConfig, load_server_configs, looks_destructive,
)

# A conforming server: handshake, two tools, one of which is destructive.
FAKE_SERVER = textwrap.dedent("""
    import json, sys
    TOOLS = [
        {"name": "read_note", "description": "Read a note.",
         "inputSchema": {"type": "object",
                         "properties": {"path": {"type": "string"}}}},
        {"name": "delete_note", "description": "Delete a note.",
         "inputSchema": {"type": "object",
                         "properties": {"path": {"type": "string"}}}},
    ]
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        msg = json.loads(line)
        if "id" not in msg:                      # a notification: no reply
            continue
        method, out = msg["method"], {}
        if method == "initialize":
            out = {"protocolVersion": "2025-06-18", "capabilities": {},
                   "serverInfo": {"name": "fake", "version": "1"}}
        elif method == "tools/list":
            out = {"tools": TOOLS}
        elif method == "tools/call":
            name = msg["params"]["name"]
            args = msg["params"].get("arguments", {})
            if name == "boom":
                out = {"content": [{"type": "text", "text": "it broke"}],
                       "isError": True}
            else:
                out = {"content": [{"type": "text",
                                    "text": f"{name} ran with {sorted(args)}"}]}
        print(json.dumps({"jsonrpc": "2.0", "id": msg["id"], "result": out}),
              flush=True)
""")


@pytest.fixture
def server(tmp_path):
    path = tmp_path / "fake_server.py"
    path.write_text(FAKE_SERVER)
    instance = MCPServer("fake", [sys.executable, str(path)])
    instance.start()
    yield instance
    instance.stop()


# ---- protocol ---------------------------------------------------------
def test_the_handshake_discovers_the_servers_tools(server):
    assert [t.name for t in server.tools] == ["read_note", "delete_note"]
    assert server.tools[0].qualified_name == "fake__read_note"


def test_tool_names_are_namespaced_with_characters_the_api_accepts(server):
    """The Messages API restricts tool names to [a-zA-Z0-9_-], so a dotted
    name is rejected outright — and two servers may both offer `search`."""
    import re

    for tool in server.tools:
        assert re.fullmatch(r"[a-zA-Z0-9_-]{1,128}", tool.qualified_name)


def test_calling_a_tool_returns_its_text(server):
    assert server.call("read_note", {"path": "a.txt"}) == \
        "read_note ran with ['path']"


def test_a_server_error_is_raised_not_returned_as_success(server):
    with pytest.raises(MCPError, match="it broke"):
        server.call("boom", {})


def test_a_server_that_never_answers_is_killed_rather_than_hung(tmp_path):
    """The lesson the voice loop already paid for four times.

    An MCP server is an arbitrary third-party process called from the middle
    of a conversation. One that stops answering without exiting would park
    the ritual thread forever — heartbeat included — and the process stays
    ALIVE, so launchd's KeepAlive never restarts it.
    """
    path = tmp_path / "silent.py"
    path.write_text("import sys, time\nfor _ in sys.stdin: time.sleep(60)\n")
    instance = MCPServer("silent", [sys.executable, str(path)], timeout=1.0)
    with pytest.raises(MCPError, match="did not answer within"):
        instance.start()
    instance.stop()


def test_a_server_that_dies_mid_request_is_reported_not_awaited(tmp_path):
    path = tmp_path / "quitter.py"
    path.write_text("import sys\nsys.stdin.readline()\n")
    instance = MCPServer("quitter", [sys.executable, str(path)], timeout=5.0)
    with pytest.raises(MCPError):
        instance.start()
    instance.stop()


def test_a_server_that_cannot_be_launched_fails_loudly(tmp_path):
    instance = MCPServer("missing", ["/nonexistent/definitely-not-here"])
    with pytest.raises(MCPError, match="could not start"):
        instance.start()


# ---- the destructive gate (spec §12) ----------------------------------
@pytest.mark.parametrize("name", [
    "delete_note", "send_message", "write_file", "move_file", "create_event",
    "run_command", "purge_inbox", "update_row", "reply_all", "archive_thread",
])
def test_verbs_that_are_hard_to_take_back_are_gated(name):
    assert looks_destructive(name), f"{name} should require confirmation"


@pytest.mark.parametrize("name", [
    "read_note", "list_directory", "search_files", "get_file_info",
    "directory_tree", "read_multiple_files",
])
def test_reads_are_not_gated(name):
    """A gate on every tool is a gate nobody listens to."""
    assert not looks_destructive(name)


class _Voice:
    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.spoken: list[str] = []

    def speak(self, sentences) -> None:
        self.spoken.extend(sentences)

    def listen(self) -> str:
        return self.answer


def _registry(server, voice=None, store=None):
    registry = MCPRegistry(
        confirmer=Confirmer(voice) if voice is not None else None, store=store
    )
    registry._servers[server.name] = server
    registry._tools = {t.qualified_name: t for t in server.tools}
    return registry


def test_a_destructive_tool_asks_before_it_runs(server):
    voice = _Voice("yes")
    calls = _registry(server, voice).callables()
    result = calls["fake__delete_note"](path="a.txt")
    assert voice.spoken, "the user was never asked"
    assert "Shall I?" in voice.spoken[0]
    assert "delete note" in voice.spoken[0], "asked about the tool, not the action"
    assert "delete_note ran" in result


def test_a_refusal_stops_the_call_and_tells_the_model_so(server):
    """Returned to the model, not raised: a refusal is a normal outcome the
    model must be able to talk about, not an error to retry around."""
    voice = _Voice("no, don't")
    result = _registry(server, voice).callables()["fake__delete_note"](path="a")
    assert "declined" in result and "NOT" in result
    assert "Do not try it again" in result
    assert voice.spoken[-1] == "Okay, I won't."


def test_a_read_is_not_interrupted_by_a_question(server):
    voice = _Voice("yes")
    _registry(server, voice).callables()["fake__read_note"](path="a.txt")
    assert voice.spoken == [], "a read should never stop to ask"


def test_no_voice_means_no_consent(server):
    """A scheduled check-in has nobody to ask, and "nobody objected" is not
    the same as "someone agreed"."""
    result = _registry(server, None).callables()["fake__delete_note"](path="a")
    assert "needs spoken confirmation" in result
    assert "nothing was done" in result


def test_confirmation_is_refused_when_the_voice_channel_is_absent(server):
    result = _registry(server, _Voice("")).callables()["fake__delete_note"](path="a")
    assert "declined" in result


# ---- registry ---------------------------------------------------------
def test_every_mcp_call_reaches_the_action_log(server, tmp_path):
    """§1 promises the dashboard 'history to display on the day it launches'.

    A tool surface where two calls are recorded and fourteen are not makes
    that log actively misleading — and the destructive ones are precisely
    the ones you would want to look up afterwards.
    """
    from zeus.clock import FakeClock
    from zeus.memory.store import Store
    from datetime import datetime, timezone

    store = Store(tmp_path / "zeus.db",
                  FakeClock(datetime(2026, 8, 8, tzinfo=timezone.utc)))
    _registry(server, _Voice("yes"), store).callables()["fake__read_note"](path="a")
    logged = store.recent_actions()
    assert [a.tool for a in logged] == ["fake__read_note"]
    assert logged[0].ok is True
    store.close()


def test_a_server_that_will_not_start_costs_only_its_own_tools(tmp_path):
    """One broken server must not cost ZEUS the others, or the ritual."""
    good = tmp_path / "good.py"
    good.write_text(FAKE_SERVER)
    registry = MCPRegistry()
    registry.start([
        ServerConfig("broken", ["/nonexistent/nope"]),
        ServerConfig("good", [sys.executable, str(good)]),
    ])
    assert [t.server for t in registry.tools] == ["good", "good"]
    registry.stop()


def test_the_tool_definitions_carry_the_servers_own_schema(server):
    definition = _registry(server).definitions()[0]
    assert definition["name"] == "fake__read_note"
    assert definition["input_schema"]["properties"]["path"]["type"] == "string"


def test_a_gated_tool_says_so_in_its_description(server):
    definitions = {d["name"]: d for d in _registry(server).definitions()}
    assert "spoken confirmation" in definitions["fake__delete_note"]["description"]
    assert "spoken confirmation" not in definitions["fake__read_note"]["description"]


# ---- config -----------------------------------------------------------
def test_a_malformed_server_entry_is_skipped_not_raised():
    """Parsed at daemon startup under KeepAlive:true, so a typo here must not
    become a respawn loop — the failure _load_config_or_default exists for."""
    configs = load_server_configs({
        "ok": {"command": ["npx", "server"]},
        "no_command": {"env": {"A": "B"}},
        "not_a_table": "npx server",
        "bad_command": {"command": "npx server"},
        "off": {"command": ["x"], "enabled": False},
    })
    assert [c.name for c in configs] == ["ok", "off"]
    assert configs[1].enabled is False


def test_env_is_carried_through_to_the_server():
    config = load_server_configs(
        {"gmail": {"command": ["npx", "s"], "env": {"TOKEN": "abc"}}}
    )[0]
    assert config.env == {"TOKEN": "abc"}
