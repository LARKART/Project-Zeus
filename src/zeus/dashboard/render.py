"""The dashboard's HTML.

DESIGN. Near-black with a green bias, mint accent used sparingly and with a
glow, tinted icon badges on the summary cards, and the day's history as a
grid of cards rather than a wall of table rows. The palette is committed
rather than derived -- ZEUS is a thing that listens in a dark room, and the
page looks like it.

NO EXTERNAL ASSETS. No CDN stylesheet, no web font, no remote image. The
page must render identically with the network unplugged, because half of
what it exists to diagnose is a machine whose network is broken. That also
rules out a webfont: this is a Mac-only tool, so the system stack resolves
to SF Pro and SF Mono every time rather than gambling on a silent fallback.

EXACTLY ONE SCRIPT, PINNED BY HASH. The wake-word popup has to appear
without the user touching anything, which a static page cannot do. Rather
than open the Content-Security-Policy to inline script generally, the
poller below is hashed and named in `script-src 'sha256-...'` -- so that
one script may run and nothing else, not even another inline block. With
JavaScript off the Session panel still renders server-side; only the
unprompted pop-up is lost.

EVERYTHING IS ESCAPED. Every value here came from somewhere the user or the
model controls -- goal text and transcripts are literally transcribed
speech, and `actions.args_json` is whatever the model passed. `esc()` is
applied at the point of interpolation, so there is one rule to check rather
than one per writer.
"""
from __future__ import annotations

import base64
import hashlib
import json
from datetime import datetime, timedelta
from html import escape
from typing import Any
from zoneinfo import ZoneInfo

from zeus.dashboard.data import Snapshot

# (id, label, group). Single source for the nav, the panels, and the
# generated active-state CSS, so a tab cannot come to point at the wrong
# panel -- a mistake that would look like a styling bug, not a broken link.
PANELS = [
    ("today", "Today", "RITUAL"),
    ("goals", "Goals", "RITUAL"),
    ("checkins", "Check-ins", "RITUAL"),
    ("session", "Session", "ACTIVITY"),
    ("log", "Action log", "ACTIVITY"),
    ("journal", "Journal", "ACTIVITY"),
    ("system", "System", "SYSTEM"),
]

_GOAL_STATUS = {
    "done": ("Done", "ok"),
    "partial": ("Partial", "warn"),
    "missed": ("Missed", "bad"),
    "carried": ("Carried", "info"),
    "pending": ("Pending", "muted"),
    "none": ("—", "empty"),
}
_CHECKIN_OUTCOME = {
    "answered": ("Answered", "ok"),
    "deferred": ("Deferred", "warn"),
    "no_answer": ("No answer", "warn"),
    "skipped": ("Skipped", "bad"),
}


def esc(value: Any) -> str:
    if value is None:
        return ""
    return escape(str(value), quote=True)


def _ago(delta: timedelta | None) -> str:
    if delta is None:
        return "never"
    seconds = int(delta.total_seconds())
    if seconds < 0:
        return "in the future"
    if seconds < 60:
        return f"{seconds}s ago"
    if seconds < 3600:
        return f"{seconds // 60}m ago"
    if seconds < 86400:
        return f"{seconds // 3600}h {(seconds % 3600) // 60}m ago"
    return f"{seconds // 86400}d ago"


def _local(value: datetime, snapshot: Snapshot, fmt: str = "%Y-%m-%d %H:%M") -> str:
    """Render a stored UTC timestamp in the user's zone.

    The database is UTC everywhere by §6.3. A page showing those values raw
    would report the 21:00 evening check-in as having fired at 04:00 the
    next day -- the exact confusion the UTC/local seam has already produced
    six defects from.
    """
    try:
        zone = ZoneInfo(snapshot.timezone_name)
    except Exception:
        return esc(value.isoformat(timespec="seconds"))
    return esc(value.astimezone(zone).strftime(fmt))


def _at(raw: Any, snapshot: Snapshot, fmt: str = "%Y-%m-%d %H:%M") -> str:
    if not raw:
        return "—"
    if isinstance(raw, datetime):
        return _local(raw, snapshot, fmt)
    try:
        parsed = datetime.fromisoformat(str(raw))
    except (ValueError, TypeError):
        return esc(raw)
    if parsed.tzinfo is None:
        from datetime import timezone as _tz
        parsed = parsed.replace(tzinfo=_tz.utc)
    return _local(parsed, snapshot, fmt)


def _pill(label: str, kind: str) -> str:
    return f'<span class="pill {esc(kind)}"><i></i>{esc(label)}</span>'


def _goal_pill(status: str | None) -> str:
    label, kind = _GOAL_STATUS.get(status or "none", (status or "—", "muted"))
    return _pill(label, kind)


def _outcome_pill(outcome: str | None) -> str:
    label, kind = _CHECKIN_OUTCOME.get(outcome or "", (outcome or "—", "muted"))
    return _pill(label, kind)


def _empty(message: str) -> str:
    return f'<p class="empty">{esc(message)}</p>'


def _table(headers: list[str], rows: list[list[str]], empty: str) -> str:
    if not rows:
        return _empty(empty)
    head = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    return (
        f'<div class="scroll"><table><thead><tr>{head}</tr></thead>'
        f"<tbody>{body}</tbody></table></div>"
    )


def _panel(panel_id: str, body: str) -> str:
    return f'<section class="panel" id="{esc(panel_id)}">{body}</section>'


# ---- summary cards ----------------------------------------------------
# (glyph, kind) -- the glyph is a geometric mark, not an emoji: emoji render
# at a different weight on every OS and read as decoration where these are
# meant to read as state.
def _stat(glyph: str, kind: str, value: str, label: str) -> str:
    return f"""
<div class="stat {esc(kind)}">
  <span class="badge">{esc(glyph)}</span>
  <div><b>{esc(value)}</b><span>{esc(label)}</span></div>
</div>"""


def _stats(snapshot: Snapshot) -> str:
    failures = sum(1 for a in snapshot.actions if not a["ok"])
    open_checkins = sum(
        1 for c in snapshot.today_checkins
        if c["outcome"] in ("deferred", "no_answer")
    )
    return f"""<div class="stats">
{_stat("◆", "ok", str(snapshot.streak.current), "day streak")}
{_stat("●", "warn" if open_checkins else "ok", str(open_checkins), "check-ins open")}
{_stat("▲", "bad" if failures else "ok", str(failures), "failed tool calls")}
{_stat("■", "info", f"{snapshot.streak.kept_days}/{snapshot.streak.total_days}",
       "goals kept")}
</div>"""


# ---- panels -----------------------------------------------------------
def _today(snapshot: Snapshot) -> str:
    health = snapshot.health
    dot = {"alive": "ok", "stale": "bad", "never": "muted"}[health.status]
    goal = snapshot.today_goal
    if goal:
        goal_block = (
            f'<p class="goal">{esc(goal["text"])}</p>'
            f'<div class="row">{_goal_pill(goal["status"])}'
            + (f'<span class="muted">{esc(goal["notes"])}</span>'
               if goal.get("notes") else "")
            + "</div>"
        )
    else:
        goal_block = _empty("No goal set for today. The morning check-in asks for it.")

    rituals = "".join(
        f'<div class="ritual"><b>{esc(c["kind"].title())}</b>'
        f'{_outcome_pill(c["outcome"])}'
        f'<span class="muted">attempt {esc(c["attempts"])}'
        + (f' · retry {_at(c["retry_at"], snapshot, "%H:%M")}'
           if c.get("retry_at") else "")
        + (f' · fired {_at(c["fired_at"], snapshot, "%H:%M")}'
           if c.get("fired_at") else "")
        + "</span></div>"
        for c in snapshot.today_checkins
    ) or _empty("Neither check-in has run today.")

    cells = "".join(
        f'<span class="cell {esc(_GOAL_STATUS.get(d["status"], ("", "muted"))[1])}"'
        f' title="{esc(d["date"])} — {esc(d["text"] or "no goal")}"></span>'
        for d in snapshot.streak.recent
    )
    return _panel("today", f"""
{_stats(snapshot)}
<div class="grid-2">
  <div class="card feature">
    <span class="eyebrow">Today · {esc(snapshot.today)}</span>
    {goal_block}
    <div class="divider"></div>
    <span class="eyebrow">Check-ins</span>
    {rituals}
  </div>
  <div class="card">
    <span class="eyebrow">Daemon</span>
    <div class="health"><span class="dot {dot}"></span>
      <div><b>{esc(health.status.title())}</b>
      <span class="muted">{esc(health.detail)}</span>
      <span class="muted">heartbeat {esc(_ago(health.age))}</span></div>
    </div>
    <div class="divider"></div>
    <span class="eyebrow">Last {len(snapshot.streak.recent)} days</span>
    <div class="strip">{cells}</div>
    <div class="legend">{"".join(
        f'<span><i class="cell {kind}"></i>{esc(label)}</span>'
        for label, kind in _GOAL_STATUS.values())}</div>
  </div>
</div>""")


def _goals(snapshot: Snapshot) -> str:
    """Goals as a card grid, the shape the reference uses for its assets.

    A goal has the same anatomy an asset does -- a name, a category, a pair
    of dates, and a state -- and reads far better as a card than as a row
    whose notes column wraps to four lines.
    """
    if not snapshot.goals:
        body = _empty("No goals recorded yet.")
    else:
        cards = "".join(f"""
<article class="card goal-card {esc(g["status"])}">
  <h3>{esc(g["text"])}</h3>
  <span class="muted mono">{esc(g["date"])}</span>
  <div class="kv"><span>Set</span><b>{_at(g.get("set_at"), snapshot, "%b %d, %H:%M")}</b></div>
  <div class="kv"><span>Reviewed</span><b>{_at(g.get("reviewed_at"), snapshot, "%b %d, %H:%M")}</b></div>
  {f'<p class="notes">{esc(g["notes"])}</p>' if g.get("notes") else ""}
  <div class="card-foot">{_goal_pill(g["status"])}</div>
</article>""" for g in snapshot.goals)
        body = f'<div class="grid-3">{cards}</div>'
    return _panel("goals", body)


def _checkins(snapshot: Snapshot) -> str:
    rows = [
        [f'<span class="mono">{esc(c["local_date"])}</span>', esc(c["kind"].title()),
         _outcome_pill(c["outcome"]),
         f'<span class="mono">{esc(c["attempts"])}</span>',
         f'<span class="mono muted">{_at(c.get("fired_at"), snapshot)}</span>',
         f'<span class="mono muted">{_at(c.get("retry_at"), snapshot)}</span>',
         "yes" if c.get("notified") else "no"]
        for c in snapshot.checkins
    ]
    return _panel("checkins", _table(
        ["Date", "Kind", "Outcome", "Attempts", "Fired", "Retry due", "Notified"],
        rows, "No check-ins recorded yet."))


def _thread(conversation: dict, snapshot: Snapshot) -> str:
    return "".join(
        f'<div class="bubble {esc(m["role"])}">'
        f'<span class="who">{esc(m["role"])}</span>'
        f'<p>{esc(m["content"])}</p>'
        f'<span class="when">{_at(m["ts"], snapshot, "%H:%M")}</span></div>'
        for m in conversation["messages"]
    ) or _empty("No messages in this conversation.")


def _session(snapshot: Snapshot) -> str:
    """Conversations as chat threads, newest first, live one pinned open."""
    if not snapshot.conversations:
        body = _empty("No conversations yet. Say the wake word to start one.")
    else:
        blocks = []
        for index, conversation in enumerate(snapshot.conversations):
            live = conversation.get("ended_at") is None
            blocks.append(
                f'<details class="card thread"{" open" if live or index == 0 else ""}>'
                f'<summary>{_pill("live" if live else conversation["trigger"],
                                  "ok" if live else "info")}'
                f'<b class="mono">#{esc(conversation["id"])}</b>'
                f'<span class="muted">{_at(conversation["started_at"], snapshot)}'
                f' · {len(conversation["messages"])} messages</span></summary>'
                f'<div class="chat">{_thread(conversation, snapshot)}</div></details>'
            )
        body = "".join(blocks)
    return _panel("session", body)


def _log(snapshot: Snapshot) -> str:
    if not snapshot.actions:
        body = _empty("No tool calls recorded yet.")
    else:
        entries = []
        for action in snapshot.actions:
            args = json.dumps(action.get("args"), ensure_ascii=False, default=str)
            result = json.dumps(action.get("result"), ensure_ascii=False, default=str)
            failed = not action["ok"]
            label, value = (
                ("Error", action.get("error") or result) if failed else ("Result", result)
            )
            entries.append(f"""
<article class="card entry {'failed' if failed else ''}">
  <div class="entry-head">
    <span class="dot {'bad' if failed else 'ok'}"></span>
    <b class="mono">{esc(action["tool"])}</b>
    <span class="muted">{_at(action["ts"], snapshot)}</span>
    <span class="muted push">{esc(action["duration_ms"])} ms</span>
  </div>
  <div class="io"><span class="gutter">Input</span><code>{esc(args)}</code></div>
  <div class="io"><span class="gutter {'bad' if failed else ''}">{label}</span>
    <code>{esc(value)}</code></div>
</article>""")
        body = "".join(entries)
    return _panel("log", body)


def _journal(snapshot: Snapshot) -> str:
    if not snapshot.journal:
        body = _empty("No journal entries yet.")
    else:
        body = "".join(
            f'<details class="card" open><summary>'
            f'<b class="mono">{esc(entry["date"])}.md</b></summary>'
            f'<pre>{esc(entry["body"])}</pre></details>'
            for entry in snapshot.journal
        )
    return _panel("journal", body)


def _system(snapshot: Snapshot) -> str:
    jobs = _table(
        ["Job", "Cron", "Last run", "Next run", ""],
        [[f'<span class="mono">{esc(j["name"])}</span>',
          f'<span class="mono">{esc(j["schedule"])}</span>',
          f'<span class="mono muted">{_at(j.get("last_run_at"), snapshot)}</span>',
          f'<span class="mono muted">{_at(j.get("next_run_at"), snapshot)}</span>',
          _pill("Enabled" if j.get("enabled") else "Disabled",
                "ok" if j.get("enabled") else "muted")]
         for j in snapshot.jobs],
        "No jobs registered — the daemon registers them at startup.")
    settings = _table(
        ["Setting", "Value"],
        [[esc(k), f'<span class="mono">{esc(v)}</span>']
         for k, v in sorted(snapshot.settings.items())],
        "No settings to show.")
    facts = _table(
        ["Key", "Value", "Source", "Learned"],
        [[f'<span class="mono">{esc(f["key"])}</span>', esc(f["value"]),
          esc(f["source"]),
          f'<span class="mono muted">{_at(f.get("learned_at"), snapshot)}</span>']
         for f in snapshot.facts],
        "Nothing learned yet — pattern learning arrives in Slice 4.")
    return _panel("system",
                  f"<h4>Scheduled jobs</h4>{jobs}<h4>Settings</h4>{settings}"
                  f"<h4>Facts</h4>{facts}")


# ---- the wake-word pop-up ---------------------------------------------
def live_session(snapshot: Snapshot) -> dict:
    """The conversation currently in progress, if there is one.

    `ended_at IS NULL` is the whole test: CheckIn and _handle_activation both
    close their conversation in a `finally`, so an open row means one is
    genuinely running right now rather than merely recent.
    """
    for conversation in snapshot.conversations:
        if conversation.get("ended_at") is None:
            return {
                "active": True,
                "id": conversation["id"],
                "trigger": conversation["trigger"],
                "messages": [
                    {"role": m["role"], "content": m["content"]}
                    for m in conversation["messages"]
                ],
            }
    return {"active": False, "id": None, "trigger": None, "messages": []}


def _popup(snapshot: Snapshot) -> str:
    """Rendered server-side too, so it is right on first paint.

    The poller only has to keep it right afterwards -- which also means the
    panel degrades to "correct but static" rather than "absent" when
    JavaScript is off.
    """
    session = live_session(snapshot)
    bubbles = "".join(
        f'<div class="bubble {esc(m["role"])}">'
        f'<span class="who">{esc(m["role"])}</span>'
        f'<p>{esc(m["content"])}</p></div>'
        for m in session["messages"]
    )
    return f"""
<aside id="wake" class="popup{' on' if session['active'] else ''}" aria-live="polite">
  <div class="popup-head">
    <span class="mic"><i></i></span>
    <div><b>ZEUS is listening</b>
      <span class="muted" id="wake-trigger">{esc(session["trigger"] or "wake word")}</span>
    </div>
    <a class="close" href="#" aria-label="dismiss">×</a>
  </div>
  <div class="chat" id="wake-chat">{bubbles}</div>
</aside>"""


# The ONLY script on the page, pinned by hash in the CSP. Deliberately tiny:
# poll, diff, paint. It touches nothing but the pop-up.
_POLLER = """
(function () {
  var box = document.getElementById('wake');
  var chat = document.getElementById('wake-chat');
  var trigger = document.getElementById('wake-trigger');
  var last = '';
  function draw(s) {
    var key = JSON.stringify(s);
    if (key === last) return;
    last = key;
    box.classList.toggle('on', !!s.active);
    if (!s.active) return;
    trigger.textContent = s.trigger || 'wake word';
    chat.textContent = '';
    s.messages.forEach(function (m) {
      var b = document.createElement('div');
      b.className = 'bubble ' + m.role;
      var who = document.createElement('span');
      who.className = 'who';
      who.textContent = m.role;
      var p = document.createElement('p');
      p.textContent = m.content;
      b.appendChild(who); b.appendChild(p); chat.appendChild(b);
    });
    chat.scrollTop = chat.scrollHeight;
  }
  function poll() {
    fetch('/api/session', { cache: 'no-store' })
      .then(function (r) { return r.json(); })
      .then(draw)
      .catch(function () {});
  }
  setInterval(poll, 2000);
  poll();
  document.querySelector('#wake .close').addEventListener('click', function (e) {
    e.preventDefault();
    box.classList.remove('on');
  });
})();
"""


def poller_hash() -> str:
    """The CSP source expression that allows exactly this script and no other."""
    digest = hashlib.sha256(_POLLER.encode("utf-8")).digest()
    return f"'sha256-{base64.b64encode(digest).decode('ascii')}'"


# ---- page -------------------------------------------------------------
def _nav() -> str:
    return '<nav class="segments">' + "".join(
        f'<a class="seg seg-{pid}" href="#{pid}">{esc(label)}</a>'
        for pid, label, _ in PANELS
    ) + "</nav>"


def render_page(snapshot: Snapshot) -> str:
    banner = ""
    if snapshot.error:
        banner = (f'<div class="banner"><b>Cannot read the database.</b> '
                  f"{esc(snapshot.error)}</div>")
    panels = "".join([
        _today(snapshot), _goals(snapshot), _checkins(snapshot), _session(snapshot),
        _log(snapshot), _journal(snapshot), _system(snapshot),
    ])
    dot = {"alive": "ok", "stale": "bad", "never": "muted"}[snapshot.health.status]
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ZEUS</title>
<style>{_CSS}{_ACTIVE_STATE_CSS}</style>
</head>
<body>
<div class="aurora"></div>
<div class="wrap">
  <header class="masthead">
    <div class="mark"><span>Z</span></div>
    <div class="wordmark">
      <h1>ZEUS <em>Assistant</em></h1>
      <p>LISTEN · ASK · REMEMBER</p>
    </div>
    <div class="actions">
      <span class="btn ghost"><span class="dot {dot}"></span>{esc(snapshot.health.status)}</span>
      <span class="btn ghost">{esc(snapshot.timezone_name)}</span>
      <span class="btn solid">{esc(snapshot.today)}</span>
    </div>
  </header>
  {_nav()}
  <main class="content">{banner}{panels}</main>
  <footer class="foot">
    Local to this machine · bound to 127.0.0.1 · nothing here leaves your Mac
  </footer>
</div>
{_popup(snapshot)}
<script>{_POLLER}</script>
</body>
</html>"""


def render_json(snapshot: Snapshot) -> str:
    def encode(value: Any) -> Any:
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, timedelta):
            return value.total_seconds()
        if hasattr(value, "__dict__"):
            return value.__dict__
        return str(value)

    return json.dumps(snapshot.__dict__, default=encode, indent=2)


# Generated, not hand-typed: seven near-identical selector pairs written by
# hand is seven chances to point a segment at the wrong panel.
_ACTIVE_STATE_CSS = "".join(
    f"body:has(#{pid}:target) .seg-{pid}{{color:#04140c;background:var(--accent);"
    f"box-shadow:0 0 18px var(--glow);}}"
    for pid, _, _ in PANELS
)

_CSS = """
:root {
  --bg:#070b09; --surface:#0d1310; --card:#101713; --line:#1d2a23;
  --ink:#e6efe9; --muted:#8a9b92; --dim:#5f7168;
  --accent:#4ade80; --accent-2:#22c55e; --glow:rgba(74,222,128,.28);
  --ok:#4ade80; --ok-bg:rgba(74,222,128,.12);
  --warn:#fbbf24; --warn-bg:rgba(251,191,36,.12);
  --bad:#f87171; --bad-bg:rgba(248,113,113,.12);
  --info:#60a5fa; --info-bg:rgba(96,165,250,.12);
  --empty:rgba(255,255,255,.06);
  --mono:ui-monospace,"SF Mono",SFMono-Regular,Menlo,Consolas,monospace;
  --sans:-apple-system,BlinkMacSystemFont,"SF Pro Text","Segoe UI",Roboto,sans-serif;
  --r:14px;
}
/* A committed dark design, but a utility opened at midday should not sear.
   Light is a full token set, never a partial override. */
@media (prefers-color-scheme: light) {
  :root:not([data-theme="dark"]) {
    --bg:#f3f6f4; --surface:#ffffff; --card:#ffffff; --line:#dde5e0;
    --ink:#0f1a14; --muted:#5c6b63; --dim:#84968d;
    --accent:#15803d; --accent-2:#166534; --glow:rgba(21,128,61,.18);
    --ok:#15803d; --ok-bg:rgba(21,128,61,.1);
    --warn:#a16207; --warn-bg:rgba(161,98,7,.1);
    --bad:#b91c1c; --bad-bg:rgba(185,28,28,.1);
    --info:#1d4ed8; --info-bg:rgba(29,78,216,.1);
    --empty:rgba(15,26,20,.07);
  }
}
:root[data-theme="light"] {
  --bg:#f3f6f4; --surface:#ffffff; --card:#ffffff; --line:#dde5e0;
  --ink:#0f1a14; --muted:#5c6b63; --dim:#84968d;
  --accent:#15803d; --accent-2:#166534; --glow:rgba(21,128,61,.18);
  --ok:#15803d; --ok-bg:rgba(21,128,61,.1);
  --warn:#a16207; --warn-bg:rgba(161,98,7,.1);
  --bad:#b91c1c; --bad-bg:rgba(185,28,28,.1);
  --info:#1d4ed8; --info-bg:rgba(29,78,216,.1);
  --empty:rgba(15,26,20,.07);
}
* { box-sizing:border-box; }
body {
  margin:0; background:var(--bg); color:var(--ink); font-family:var(--sans);
  font-size:14px; line-height:1.55; -webkit-font-smoothing:antialiased;
  min-height:100vh;
}
.aurora {
  position:fixed; inset:0; pointer-events:none; z-index:0;
  background:
    radial-gradient(60rem 30rem at 12% -10%, var(--glow), transparent 60%),
    radial-gradient(50rem 26rem at 92% 4%, rgba(34,197,94,.16), transparent 62%);
}
.wrap { position:relative; z-index:1; max-width:1180px; margin:0 auto;
  padding:26px 22px 40px; }
.mono, code, pre, table, .pill, .eyebrow { font-family:var(--mono); }
a { color:inherit; text-decoration:none; }
a:focus-visible, summary:focus-visible {
  outline:2px solid var(--accent); outline-offset:3px; border-radius:8px;
}
/* masthead */
.masthead { display:flex; align-items:center; gap:15px; margin-bottom:22px;
  flex-wrap:wrap; }
.mark {
  width:46px; height:46px; border-radius:13px; flex:none; display:grid;
  place-items:center; background:linear-gradient(150deg,var(--accent),var(--accent-2));
  box-shadow:0 0 26px var(--glow); color:#04140c; font-weight:700; font-size:22px;
  font-family:var(--mono);
}
.wordmark h1 { margin:0; font-size:26px; font-weight:650; letter-spacing:-.01em; }
.wordmark h1 em { font-style:normal; color:var(--accent); }
.wordmark p { margin:2px 0 0; color:var(--dim); font-size:10.5px;
  letter-spacing:.22em; font-family:var(--mono); }
.actions { margin-left:auto; display:flex; gap:9px; flex-wrap:wrap; }
.btn {
  display:inline-flex; align-items:center; gap:7px; padding:8px 15px;
  border-radius:11px; font-size:12.5px; border:1px solid var(--line);
  background:var(--surface); color:var(--muted); white-space:nowrap;
  text-transform:capitalize;
}
.btn.solid {
  background:linear-gradient(150deg,var(--accent),var(--accent-2)); color:#04140c;
  border-color:transparent; box-shadow:0 0 20px var(--glow); font-weight:600;
  font-family:var(--mono);
}
/* segmented nav */
.segments {
  display:flex; gap:5px; padding:5px; margin-bottom:20px; overflow-x:auto;
  background:var(--surface); border:1px solid var(--line); border-radius:var(--r);
}
.seg { padding:8px 15px; border-radius:10px; color:var(--muted); font-size:12.5px;
  white-space:nowrap; }
.seg:hover { color:var(--ink); background:var(--empty); }
/* panels switch on :target -- with no hash yet, Today opens */
.panel { display:none; }
.panel:target { display:block; }
body:not(:has(.panel:target)) #today { display:block; }
/* stat cards */
.stats { display:grid; grid-template-columns:repeat(4,1fr); gap:13px;
  margin-bottom:20px; }
@media (max-width:860px) { .stats { grid-template-columns:repeat(2,1fr); } }
.stat {
  display:flex; align-items:center; gap:13px; padding:15px 17px;
  background:var(--card); border:1px solid var(--line); border-radius:var(--r);
}
.stat > div { display:flex; flex-direction:column; }
.stat b { font-size:26px; font-family:var(--mono); font-variant-numeric:tabular-nums;
  line-height:1.15; }
.stat span:last-child { color:var(--muted); font-size:11.5px; }
.badge { width:36px; height:36px; border-radius:11px; display:grid; place-items:center;
  font-size:13px; flex:none; }
.stat.ok .badge { color:var(--ok); background:var(--ok-bg);
  box-shadow:inset 0 0 0 1px var(--ok-bg); }
.stat.warn .badge { color:var(--warn); background:var(--warn-bg); }
.stat.bad .badge { color:var(--bad); background:var(--bad-bg); }
.stat.info .badge { color:var(--info); background:var(--info-bg); }
.stat.bad { border-color:rgba(248,113,113,.35); }
.stat.warn { border-color:rgba(251,191,36,.3); }
/* cards */
.card {
  background:var(--card); border:1px solid var(--line); border-radius:var(--r);
  padding:18px 20px; margin-bottom:13px;
}
.grid-2 { display:grid; grid-template-columns:1.55fr 1fr; gap:13px; }
@media (max-width:820px) { .grid-2 { grid-template-columns:1fr; } }
.grid-3 { display:grid; grid-template-columns:repeat(3,1fr); gap:13px; }
@media (max-width:980px) { .grid-3 { grid-template-columns:repeat(2,1fr); } }
@media (max-width:660px) { .grid-3 { grid-template-columns:1fr; } }
.grid-2 > .card, .grid-3 > .card { margin-bottom:0; }
.eyebrow { display:block; color:var(--dim); font-size:10.5px; letter-spacing:.16em;
  text-transform:uppercase; margin-bottom:10px; }
.goal { margin:0 0 12px; font-size:23px; line-height:1.3; font-weight:640;
  text-wrap:balance; }
.row { display:flex; align-items:center; gap:10px; flex-wrap:wrap; }
.muted { color:var(--muted); font-size:12.5px; }
.divider { height:1px; background:var(--line); margin:16px 0; }
.ritual { display:flex; align-items:center; gap:10px; padding:5px 0; flex-wrap:wrap; }
.ritual b { min-width:66px; font-size:13px; }
.health { display:flex; gap:11px; align-items:flex-start; }
.health div { display:flex; flex-direction:column; }
.dot { width:8px; height:8px; border-radius:50%; flex:none; margin-top:7px; }
.dot.ok { background:var(--ok); box-shadow:0 0 9px var(--ok); }
.dot.warn { background:var(--warn); } .dot.bad { background:var(--bad); }
.dot.muted { background:var(--dim); }
.btn .dot, .entry-head .dot { margin-top:0; }
.strip { display:flex; gap:3px; flex-wrap:wrap; margin-bottom:11px; }
.cell { width:13px; height:13px; border-radius:4px; background:var(--empty); flex:none; }
.cell.ok { background:var(--ok); } .cell.warn { background:var(--warn); }
.cell.bad { background:var(--bad); } .cell.info { background:var(--info); }
.cell.muted { background:var(--dim); opacity:.45; }
.legend { display:flex; gap:12px; flex-wrap:wrap; color:var(--muted); font-size:11px; }
.legend span { display:flex; align-items:center; gap:5px; }
.legend i { width:9px; height:9px; border-radius:3px; display:inline-block; }
/* goal cards */
.goal-card { display:flex; flex-direction:column; gap:5px; }
.goal-card h3 { margin:0; font-size:15px; font-weight:620; line-height:1.35;
  text-wrap:balance; }
.kv { display:flex; justify-content:space-between; gap:12px; font-size:12px;
  color:var(--muted); }
.kv b { color:var(--ink); font-weight:550; font-family:var(--mono); font-size:11.5px; }
.notes { margin:6px 0 0; color:var(--muted); font-size:12.5px; }
.card-foot { margin-top:auto; padding-top:12px; }
.goal-card.done { border-color:rgba(74,222,128,.28); }
.goal-card.missed { border-color:rgba(248,113,113,.3); }
/* pills */
.pill { display:inline-flex; align-items:center; gap:6px; padding:3px 10px;
  border-radius:999px; font-size:11px; white-space:nowrap; }
.pill i { width:5px; height:5px; border-radius:50%; background:currentColor; flex:none; }
.pill.ok { color:var(--ok); background:var(--ok-bg); }
.pill.warn { color:var(--warn); background:var(--warn-bg); }
.pill.bad { color:var(--bad); background:var(--bad-bg); }
.pill.info { color:var(--info); background:var(--info-bg); }
.pill.muted, .pill.empty { color:var(--muted); background:var(--empty); }
/* tables */
.scroll { overflow-x:auto; background:var(--card); border:1px solid var(--line);
  border-radius:var(--r); }
table { width:100%; border-collapse:collapse; font-size:12.5px;
  font-variant-numeric:tabular-nums; }
th, td { text-align:left; padding:11px 16px; border-bottom:1px solid var(--line);
  vertical-align:middle; }
th { color:var(--dim); font-size:10.5px; letter-spacing:.11em; white-space:nowrap;
  text-transform:uppercase; }
tbody tr:last-child td { border-bottom:0; }
tbody tr:hover { background:var(--empty); }
h4 { margin:22px 0 9px; font-size:10.5px; color:var(--dim); font-family:var(--mono);
  letter-spacing:.16em; text-transform:uppercase; }
h4:first-of-type { margin-top:0; }
/* action log */
.entry.failed { border-color:rgba(248,113,113,.35); }
.entry-head { display:flex; align-items:center; gap:10px; margin-bottom:11px;
  flex-wrap:wrap; }
.entry-head b { font-size:13px; }
.push { margin-left:auto; }
.io { display:flex; gap:12px; align-items:flex-start; padding:3px 0; }
.gutter { flex:none; width:52px; color:var(--dim); font-family:var(--mono);
  font-size:10.5px; letter-spacing:.08em; padding-top:2px; }
.gutter.bad { color:var(--bad); }
.io code { color:var(--muted); font-size:11.5px; overflow-wrap:anywhere; }
/* chat */
.chat { display:flex; flex-direction:column; gap:10px; padding-top:12px; }
.bubble { max-width:80%; padding:9px 13px; border-radius:14px; background:var(--empty); }
.bubble .who { display:block; font-size:9.5px; letter-spacing:.14em;
  text-transform:uppercase; color:var(--dim); font-family:var(--mono); }
.bubble p { margin:2px 0 0; white-space:pre-wrap; font-size:13px; }
.bubble .when { font-size:10.5px; color:var(--dim); font-family:var(--mono); }
.bubble.assistant { align-self:flex-start; border-bottom-left-radius:5px;
  background:var(--ok-bg); }
.bubble.user { align-self:flex-end; border-bottom-right-radius:5px; }
summary { cursor:pointer; display:flex; align-items:center; gap:10px; list-style:none;
  font-size:12.5px; }
summary::-webkit-details-marker { display:none; }
pre { margin:10px 0 0; padding:14px 16px; background:var(--bg);
  border:1px solid var(--line); border-radius:10px; overflow-x:auto;
  white-space:pre-wrap; font-size:11.5px; color:var(--muted); }
.empty { color:var(--muted); font-size:13px; margin:4px 0; }
.banner { padding:14px 18px; border-radius:var(--r); margin-bottom:16px;
  color:var(--bad); background:var(--bad-bg); border:1px solid var(--bad); }
.foot { margin-top:26px; color:var(--dim); font-size:11.5px; text-align:center; }
/* wake-word pop-up */
.popup {
  position:fixed; right:20px; bottom:20px; width:min(370px,calc(100vw - 40px));
  max-height:min(520px,70vh); display:none; flex-direction:column; z-index:20;
  background:var(--surface); border:1px solid var(--accent); border-radius:18px;
  box-shadow:0 0 0 1px var(--glow),0 18px 50px rgba(0,0,0,.5),0 0 40px var(--glow);
  padding:15px 17px; overflow:hidden;
}
.popup.on { display:flex; }
.popup-head { display:flex; align-items:center; gap:11px; }
.popup-head > div { display:flex; flex-direction:column; }
.popup-head b { font-size:13.5px; }
.mic { width:32px; height:32px; border-radius:50%; flex:none; display:grid;
  place-items:center; background:var(--ok-bg); }
.mic i { width:9px; height:9px; border-radius:50%; background:var(--accent);
  box-shadow:0 0 0 0 var(--glow); animation:pulse 1.8s ease-out infinite; }
@keyframes pulse {
  0% { box-shadow:0 0 0 0 var(--glow); }
  70% { box-shadow:0 0 0 13px rgba(74,222,128,0); }
  100% { box-shadow:0 0 0 0 rgba(74,222,128,0); }
}
.close { margin-left:auto; color:var(--dim); font-size:19px; line-height:1;
  padding:0 4px; }
.close:hover { color:var(--ink); }
.popup .chat { overflow-y:auto; }
@media (prefers-reduced-motion: reduce) {
  * { transition:none !important; animation:none !important; }
}
"""
