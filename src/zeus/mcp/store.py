"""Where MCP servers added from the dashboard are kept.

A SEPARATE FILE FROM config.toml, on purpose. The dashboard has to write
this, and config.toml is a file the user hand-edits and comments. Python
ships a TOML reader and no writer, so saving would mean either serialising
TOML by hand or round-tripping through a parser that discards every comment
and every bit of formatting the user put there. Neither is worth it to store
a handful of commands.

So config.toml stays yours, mcp.json is ZEUS's, and both are read at
startup. A server defined in config.toml wins on a name clash -- the file
you edited by hand outranks the one a web page wrote.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

FILENAME = "mcp.json"


def path_for(root: Path) -> Path:
    return root / FILENAME


def load(root: Path) -> dict[str, Any]:
    """Every server the dashboard has saved. Never raises.

    Read at daemon startup under KeepAlive:true, so a corrupt file must
    degrade to "no dashboard servers" and a loud line rather than becoming a
    respawn loop -- the same rule _load_config_or_default already follows.
    """
    path = path_for(root)
    try:
        if not path.is_file():
            return {}
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        log.error("mcp: could not read %s; ignoring it", path, exc_info=True)
        return {}
    servers = data.get("servers")
    if not isinstance(servers, dict):
        log.error("mcp: %s has no 'servers' table; ignoring it", path)
        return {}
    return servers


def save(root: Path, servers: dict[str, Any]) -> None:
    """Replace the file atomically.

    Written to a temporary file in the SAME directory and renamed, because
    rename is atomic within a filesystem while a plain truncate-and-write is
    not: a crash midway through leaves a half-written JSON file that the next
    startup cannot read, and the user's server list is gone. The daemon may
    be reading this file at the moment the dashboard writes it.
    """
    root.mkdir(parents=True, exist_ok=True)
    path = path_for(root)
    handle, temporary = tempfile.mkstemp(dir=str(root), prefix=".mcp-", suffix=".json")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as file:
            json.dump({"servers": servers}, file, indent=2, sort_keys=True)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        # 0600: an MCP entry can carry an API token in `env`, and this file
        # is written by a web page. It gets the same treatment as ~/.zeus/env.
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def add(root: Path, name: str, command: list[str], env: dict[str, str],
        enabled: bool = True) -> None:
    servers = load(root)
    servers[name] = {"command": command, "env": env, "enabled": enabled}
    save(root, servers)


def remove(root: Path, name: str) -> bool:
    servers = load(root)
    if name not in servers:
        return False
    del servers[name]
    save(root, servers)
    return True


def set_enabled(root: Path, name: str, enabled: bool) -> bool:
    servers = load(root)
    if name not in servers:
        return False
    servers[name]["enabled"] = enabled
    save(root, servers)
    return True


def merged(config_servers: dict[str, Any], root: Path) -> dict[str, Any]:
    """config.toml's servers, plus the dashboard's, config.toml winning.

    The hand-edited file outranks the one a web page wrote: if you took the
    trouble to declare a server in config.toml, a stale dashboard entry of
    the same name must not quietly replace it.
    """
    combined = dict(load(root))
    for name, entry in (config_servers or {}).items():
        if name in combined:
            log.info("mcp: %s is defined in both config.toml and %s; "
                     "config.toml wins", name, FILENAME)
        combined[name] = entry
    return combined
