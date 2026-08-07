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

Say **"hey jarvis"** to start a conversation. (openWakeWord ships no "zeus"
model; see `[wake] model` in `~/.zeus/config.toml` to swap in a custom one.)

ZEUS asks for your goal at 11:00 and reviews it at 21:00. If you are away,
on a call, or in a Focus mode, it defers or notifies quietly instead of
talking at you.

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
