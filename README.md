# ZEUS

A voice assistant that runs on your Mac, wakes when you speak to it, and asks
what your one goal is each morning and whether it happened each evening.

## Requirements

macOS 12+ on Intel or Apple Silicon, and [uv](https://docs.astral.sh/uv/).
No Homebrew required.

**Install Python 3.12**, exactly as below. The source itself needs only 3.11
(`requires-python = ">=3.11"`, and the suite passes on a real 3.11), but the
pinned dependency set does not resolve there — `numpy==2.5.1` requires 3.12 —
and the audio wheels stop at 3.12 at the other end. `uv pip install` names the
offending pin if you try anything else.

## Install

```bash
uv python install 3.12
uv venv --python 3.12
uv pip install -e ".[dev]"
export ANTHROPIC_API_KEY=sk-ant-...
```

## First run — in this order

Run the self-test **from Terminal first**. macOS attributes microphone
permission to the process that asks, and a background LaunchAgent that has
never been granted access will open the microphone successfully and hear
nothing but silence.

```bash
.venv/bin/zeus doctor      # environment report
.venv/bin/zeus selftest    # grants microphone access, verifies audio in and out
```

Before loading the LaunchAgent, write your API key to `~/.zeus/env`, not
just your shell. `export ANTHROPIC_API_KEY=...` is enough for `doctor` and
`selftest` above because those run in your shell, which already has it — but
launchd inherits **no** shell environment, so the installed LaunchAgent
starts with no key and every conversation silently fails unless the key
also lives in this file:

```bash
mkdir -p ~/.zeus
printf 'ANTHROPIC_API_KEY=sk-ant-...\n' > ~/.zeus/env
chmod 600 ~/.zeus/env
```

`~/.zeus/env` is read only by ZEUS at startup, to populate the environment
for that process — it is never written by ZEUS, never read into
`config.toml`, and holds nothing but this one line.

```bash
.venv/bin/zeus install-agent
launchctl load ~/Library/LaunchAgents/com.zeus.daemon.plist
```

## Usage

Say **"hey jarvis"** to start a conversation.

**Not "Hi Zeus" — not yet.** openWakeWord ships six pretrained models
(`alexa`, `hey_mycroft`, `hey_jarvis`, `hey_rhasspy`, `timer`, `weather`)
and none of them is a Zeus. A wake word is a trained model, not a config
string, so pointing `[wake] model` at `"hi_zeus"` would only fail to load.
Getting the real phrase means training a custom model with openWakeWord's
own tooling and then setting:

```toml
[wake]
model = "/Users/you/.zeus/models/hi_zeus.onnx"
```

The plumbing already takes an arbitrary model path — it is the model that
does not exist. Until it does, `hey_jarvis` is the closest thing that
actually works, and pretending otherwise would just mean a daemon that
never wakes.

ZEUS asks for your goal at 11:00 and reviews it at 21:00. If you are away,
on a call, or in a Focus mode, it defers or notifies quietly instead of
talking at you.

## Connecting apps (MCP)

ZEUS can call tools on any [MCP](https://modelcontextprotocol.io) server —
your files, your mail, a browser, anything with a server. List them in
`~/.zeus/config.toml`:

```toml
[mcp.servers.files]
command = ["npx", "-y", "@modelcontextprotocol/server-filesystem",
           "/Users/you/Documents"]

[mcp.servers.gmail]
command = ["npx", "-y", "@your/gmail-mcp-server"]
env = { GMAIL_TOKEN = "..." }
```

Tools are discovered at startup and namespaced by server (`files__read_file`),
so two servers may both offer `search`. They arrive at the brain in exactly
the same shape as the built-in `save_goal`, so the model chooses between all
of them together.

**Anything hard to take back is confirmed out loud first.** ZEUS states the
action and waits for a spoken yes:

> "You want me to write file using files, with path notes.txt. Shall I?"

Say no and it is not run — the model is told so, and told not to retry.
Gating is by verb (`delete`, `send`, `write`, `move`, `run`, `pay`…), so
reads and searches are never interrupted. If there is no voice channel to
ask through — a scheduled check-in, a headless run — a destructive tool is
**refused**, because "nobody objected" is not "someone agreed".

Every MCP call is written to the action log alongside the built-in tools,
so the dashboard shows all of it.

One broken server costs only its own tools: ZEUS logs it and carries on.

## Dashboard

```bash
.venv/bin/zeus dashboard          # → http://127.0.0.1:8787
```

Everything ZEUS has recorded, on one page: today's goal and check-ins,
daemon health, your streak, goal history, every check-in attempt, every tool
call with its arguments and result, full transcripts, the journal, and the
scheduled jobs and settings. Say the wake word while it is open and a live
session pops up in the corner.

It needs no microphone, no API key, and no running daemon — so it also works
as the answer to "is ZEUS actually alive?".

Four things it will not do:

- **It cannot write.** The database is opened read-only, so a page load can
  never take a write lock the voice loop is waiting on.
- **It binds `127.0.0.1` only**, never the wildcard address. The page shows
  every transcript with no authentication; that is safe exactly and only
  because nothing off this machine can reach it.
- **It runs no third-party script.** One inline poller, pinned in the
  Content-Security-Policy by hash, is the whole of its JavaScript.
- **It shows no secret.** The page never reads your API key.

**Do not deploy it.** Publishing this page publishes your journal. The
`demo/` directory holds a frozen copy built from invented data
(`python tools/build_demo.py`) for showing the project off; that is the only
version meant to leave the machine.

## Your data

Everything lives in `~/.zeus/` — delete that directory and ZEUS knows
nothing. Audio is never written to disk; only transcripts are stored. Your
transcribed text goes to the Anthropic API; your audio does not, because
transcription runs locally.

**Retention is not yet enforced.** `transcript_retention_days` exists in
`~/.zeus/config.toml` (default 90) but nothing currently reads it — no purge
job runs, so transcripts are kept indefinitely in Slice 1. Automatic
deletion after that window is planned for Slice 2. Until then, the only way
to remove old transcripts is to delete them yourself from `~/.zeus/`.

## Tests

```bash
.venv/bin/pytest
```

No test requires a microphone, speakers, or a network connection.
