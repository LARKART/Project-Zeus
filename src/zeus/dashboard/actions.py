"""The dashboard's only write path: adding and removing MCP servers.

THIS IS THE MOST DANGEROUS FILE IN ZEUS, and it is worth saying why in one
place. An MCP server entry is a COMMAND LINE. A request that adds one is a
request to run an arbitrary program on this Mac, with your user's
privileges, at the next daemon start. Everywhere else the dashboard is
read-only by construction (`mode=ro`); here it is not, so the defences are
explicit rather than structural.

THE ATTACK THAT MAKES THIS NECESSARY. `127.0.0.1` is not a security boundary
against the browser. Any page on the internet can make your browser POST a
cross-origin form to http://127.0.0.1:8787 -- the attacker cannot READ the
response, but the side effect still happens, and here the side effect is
"install a command that will be executed". A drive-by page could hand
someone a shell on your machine while you read the news.

So a write must clear all four of:

  1. a same-origin marker -- Sec-Fetch-Site, or a matching Origin header;
  2. an unguessable token, minted per process and rendered into the page,
     compared in constant time;
  3. Content-Type application/x-www-form-urlencoded that ARRIVED via fetch
     rather than a form submission, which the token requirement enforces
     anyway since a cross-site form cannot read the token;
  4. the connection actually coming from loopback.

Any one of them missing is a refusal, and refusals say why -- this is a tool
for one person on their own machine, and a silent 403 while debugging your
own dashboard is its own kind of failure.
"""
from __future__ import annotations

import hmac
import logging
import secrets
import shlex
from dataclasses import dataclass
from pathlib import Path

from zeus.mcp import store as mcp_store

log = logging.getLogger(__name__)

# Minted once per dashboard process. Not persisted: a restart invalidating
# every open tab is the correct trade for a token that never sits on disk.
CSRF_TOKEN = secrets.token_urlsafe(32)

# Server names are used as a namespace prefix on every tool
# (`files__read_file`), and the Messages API restricts tool names to
# [a-zA-Z0-9_-]. Enforced here so a bad name fails at the form rather than
# silently producing tools the API rejects at conversation time.
NAME_RULE = "letters, digits, hyphen and underscore only, 1-32 characters"
MAX_NAME = 32


@dataclass(frozen=True)
class Result:
    ok: bool
    message: str
    status: int = 200


def valid_name(name: str) -> bool:
    if not name or len(name) > MAX_NAME:
        return False
    return all(c.isalnum() or c in "-_" for c in name) and "__" not in name


def check_origin(headers, host: str, port: int) -> str | None:
    """None if the request is same-origin, else why it was refused.

    Sec-Fetch-Site is the reliable signal and every current browser sends
    it; Origin is the fallback for clients that do not. A request carrying
    NEITHER is allowed only because curl and the tests send neither -- the
    CSRF token is what actually stops a cross-site write, and a browser
    cannot omit these headers to dodge this check.
    """
    site = headers.get("Sec-Fetch-Site")
    if site is not None and site not in ("same-origin", "none"):
        return f"cross-site request refused (Sec-Fetch-Site: {site})"
    origin = headers.get("Origin")
    if origin:
        allowed = {f"http://{host}:{port}", f"http://localhost:{port}"}
        if origin not in allowed:
            return f"cross-origin request refused (Origin: {origin})"
    return None


def check_token(supplied: str | None) -> str | None:
    """None if the token is right, else why it was refused.

    compare_digest, not ==: string comparison short-circuits on the first
    differing byte, which leaks the token's prefix through timing. The window
    is small on loopback and the fix is one function call.
    """
    if not supplied:
        return "missing CSRF token"
    if not hmac.compare_digest(supplied, CSRF_TOKEN):
        return "bad CSRF token"
    return None


def parse_command(raw: str) -> tuple[list[str] | None, str]:
    """Split a command line the way a shell would, without running one.

    shlex, never shell=True. The string arrives from a web form, and handing
    it to a shell would turn a quoting mistake into command injection on top
    of the execution this feature already grants deliberately.
    """
    raw = (raw or "").strip()
    if not raw:
        return None, "the command is required"
    try:
        parts = shlex.split(raw)
    except ValueError as problem:
        return None, f"could not parse the command ({problem})"
    if not parts:
        return None, "the command is required"
    return parts, ""


def parse_env(raw: str) -> tuple[dict[str, str] | None, str]:
    """KEY=VALUE per line. Empty is fine; malformed is not."""
    env: dict[str, str] = {}
    for number, line in enumerate((raw or "").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            return None, f"line {number} of the environment is not KEY=VALUE"
        key, _, value = line.partition("=")
        key = key.strip()
        if not key or " " in key:
            return None, f"line {number} has an invalid variable name"
        env[key] = value.strip()
    return env, ""


def add_server(root: Path, form: dict[str, str]) -> Result:
    name = (form.get("name") or "").strip()
    if not valid_name(name):
        return Result(False, f"Invalid server name — {NAME_RULE}.", 400)

    command, problem = parse_command(form.get("command", ""))
    if command is None:
        return Result(False, problem.capitalize() + ".", 400)

    env, problem = parse_env(form.get("env", ""))
    if env is None:
        return Result(False, problem.capitalize() + ".", 400)

    try:
        mcp_store.add(root, name, command, env)
    except OSError as failure:
        log.error("mcp: could not save %s", name, exc_info=True)
        return Result(False, f"Could not write the server list: {failure}", 500)

    log.warning("dashboard: MCP server %r added, command %r", name, command)
    return Result(
        True,
        f"Saved {name}. Restart ZEUS to load it — servers are started once, "
        f"at daemon startup, so tool discovery does not happen mid-conversation.",
    )


def remove_server(root: Path, form: dict[str, str]) -> Result:
    name = (form.get("name") or "").strip()
    if not valid_name(name):
        return Result(False, "Invalid server name.", 400)
    try:
        removed = mcp_store.remove(root, name)
    except OSError as failure:
        return Result(False, f"Could not write the server list: {failure}", 500)
    if not removed:
        # Servers declared in config.toml are deliberately not removable from
        # here: the dashboard owns mcp.json and nothing else, and silently
        # failing would look like the button was broken.
        return Result(False, f"{name} is not in mcp.json — a server declared "
                             f"in config.toml has to be removed there.", 404)
    log.warning("dashboard: MCP server %r removed", name)
    return Result(True, f"Removed {name}. Restart ZEUS to apply it.")


def toggle_server(root: Path, form: dict[str, str]) -> Result:
    name = (form.get("name") or "").strip()
    enabled = (form.get("enabled") or "").lower() in ("1", "true", "yes", "on")
    if not valid_name(name):
        return Result(False, "Invalid server name.", 400)
    try:
        changed = mcp_store.set_enabled(root, name, enabled)
    except OSError as failure:
        return Result(False, f"Could not write the server list: {failure}", 500)
    if not changed:
        return Result(False, f"{name} is not in mcp.json.", 404)
    state = "enabled" if enabled else "disabled"
    return Result(True, f"{name} {state}. Restart ZEUS to apply it.")


HANDLERS = {
    "/api/mcp/add": add_server,
    "/api/mcp/remove": remove_server,
    "/api/mcp/toggle": toggle_server,
}
