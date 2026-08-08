"""The dashboard's HTTP server. Loopback only, read-only, no framework.

BINDS 127.0.0.1, NEVER 0.0.0.0 — spec §12. The difference is one string and
it is the whole of the dashboard's security model: this page shows every
goal, every transcript and every tool call, with no authentication, because
it is unreachable from anywhere but this machine. Binding the wildcard
address would publish all of it to the local network, and on a café or
office wifi that is a stranger reading your journal. The address is a
module constant and a test pins it.

Stdlib http.server rather than a framework, because the project takes no
new third-party dependency. The dashboard is a handful of read-only GETs;
the complexity a framework would remove is not present.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo

from zeus.dashboard.data import Snapshot, read_snapshot
from zeus.dashboard.render import (
    live_session, poller_hash, render_json, render_page,
)

log = logging.getLogger(__name__)

# Loopback, always. See the module docstring before changing this.
BIND_HOST = "127.0.0.1"
DEFAULT_PORT = 8787


class DashboardServer(ThreadingHTTPServer):
    """A ThreadingHTTPServer carrying the snapshot source its handler needs.

    Threading rather than serial: a browser opens several connections for
    one page load, and a serial server makes them queue behind each other
    for no reason. Every request is a fresh read-only database connection,
    so there is no shared mutable state between them to protect.
    """

    daemon_threads = True

    def __init__(self, address, snapshot_source: Callable[[], Snapshot]) -> None:
        super().__init__(address, _Handler)
        self.snapshot_source = snapshot_source


class _Handler(BaseHTTPRequestHandler):
    server_version = "zeus-dashboard"
    # Silence BaseHTTPRequestHandler's default stderr access log; the daemon
    # owns stderr and a page load should not scribble on it.
    def log_message(self, fmt: str, *args) -> None:  # noqa: A003
        log.debug("dashboard %s", fmt % args)

    def _respond(self, status: int, body: str, content_type: str) -> None:
        payload = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        # This page renders local, untrusted-in-practice text (transcribed
        # speech, model-authored tool arguments). Everything is escaped at
        # render time; these headers are the second layer, so one missed
        # escape is not immediately an executing script.
        #
        # script-src NAMES ONE HASH, and 'unsafe-inline' is deliberately
        # absent. The wake-word pop-up needs a poller, but allowing inline
        # script generally would mean a `<script>` smuggled through a
        # transcript would run too. With a hash, only the exact bytes of
        # render._POLLER may execute; anything else -- including a one-
        # character edit to the poller itself -- is refused by the browser.
        # connect-src 'self' is what lets it reach /api/session and nothing
        # else, so a missed escape cannot exfiltrate the page.
        self.send_header(
            "Content-Security-Policy",
            f"default-src 'none'; style-src 'unsafe-inline'; "
            f"script-src {poller_hash()}; connect-src 'self'; "
            f"base-uri 'none'; form-action 'none'; frame-ancestors 'none'",
        )
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(payload)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's contract
        route = self.path.split("?", 1)[0].rstrip("/") or "/"
        if route == "/favicon.ico":
            self._respond(404, "", "text/plain")
            return
        try:
            snapshot = self.server.snapshot_source()
        except Exception:
            # A dashboard that 500s tells the user nothing about the daemon
            # it exists to report on. Say what broke, in the browser.
            log.exception("dashboard snapshot failed")
            self._respond(
                500,
                "ZEUS dashboard could not read its data. See the daemon log.",
                "text/plain",
            )
            return
        if route == "/api/session":
            # The pop-up's whole feed. Kept separate from /api/snapshot and
            # deliberately tiny: it is fetched every two seconds for as long
            # as the page is open, and shipping the entire action log and
            # every transcript at that cadence would be absurd.
            import json as _json
            self._respond(200, _json.dumps(live_session(snapshot)),
                          "application/json")
            return
        if route == "/api/snapshot":
            self._respond(200, render_json(snapshot), "application/json")
            return
        if route == "/":
            self._respond(200, render_page(snapshot), "text/html")
            return
        self._respond(404, "Not found", "text/plain")

    def do_HEAD(self) -> None:  # noqa: N802
        self.do_GET()


def build_server(
    db_path: Path, journal_dir: Path, tz: ZoneInfo,
    port: int = DEFAULT_PORT, settings: dict | None = None,
    now: Callable[[], datetime] | None = None,
) -> DashboardServer:
    """Bind the dashboard to loopback and return it, not yet serving.

    Returned unstarted so a caller can read `server_address` for the port
    actually assigned — passing port=0 asks the OS for a free one, which is
    how the tests avoid fighting over 8787 (and each other).
    """
    clock = now or (lambda: datetime.now(timezone.utc))

    def snapshot() -> Snapshot:
        return read_snapshot(db_path, journal_dir, tz, clock(), settings)

    return DashboardServer((BIND_HOST, port), snapshot)


def serve(server: DashboardServer) -> None:
    host, port = server.server_address[:2]
    log.info("dashboard listening on http://%s:%s", host, port)
    print(f"ZEUS dashboard → http://{host}:{port}  (Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()
