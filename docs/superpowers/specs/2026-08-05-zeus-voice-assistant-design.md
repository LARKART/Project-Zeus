# ZEUS — Design Spec (Slice 1: Spine + Daily Ritual)

**Date:** 2026-08-05
**Status:** Approved for planning
**Scope:** Slice 1 of 4. Later slices are sketched in §14 but are **not** specified here.

---

## 1. Overview

ZEUS is a voice-driven personal assistant that runs continuously on the user's Mac. It
wakes on a spoken keyword, holds short spoken conversations, and proactively checks in
twice a day — at 11:00 to ask what the day's one goal is, and at 21:00 to ask whether it
happened.

The full product vision spans roughly ten subsystems (voice I/O, scheduling, memory and
pattern learning, device telemetry, web access, music control, file operations, shell and
Shortcuts automation, calendar, email, plus a dashboard and a network watchman). That is
too large for one specification. This document covers **Slice 1: the spine**, proved
end-to-end by shipping exactly one user-visible feature — the daily check-in ritual.

Everything the later slices need in order to be *possible* is built here (tool dispatch,
the action log, a generic recurring scheduler, swappable provider interfaces). Nothing
they need in order to be *complete* is built here.

### 1.1 Slice 1 deliverable

A daemon that:

1. Runs at login and stays running.
2. Wakes on a spoken keyword and holds a short conversation.
3. At 11:00 asks for the day's goal — after checking whether it is a reasonable moment to
   speak — and records the answer.
4. At 21:00 recalls that goal, asks how it went, and records the outcome.
5. Logs every tool call it makes, so the Slice 2 dashboard has history to display on the
   day it launches.

### 1.2 Explicit non-goals for Slice 1

Device health, file operations, music control, shell/Shortcuts automation, calendar,
email, web browsing, the dashboard, the menu bar app, the network watchman, and pattern
learning are **all out of scope**. Slice 1 builds the surfaces they plug into.

---

## 2. Decisions and rationale

Recorded so that later slices do not relitigate them.

| # | Decision | Rationale |
|---|---|---|
| D1 | Slice the project; build the spine first, proved via the daily ritual | A ten-subsystem spec is not implementable. The ritual is the smallest thing that exercises mic → STT → brain → TTS → scheduler → memory end to end and is independently useful on day one. |
| D2 | Provider interfaces are cloud-ready; Slice 1 defaults are local | The user selected "cloud-first, pluggable" but holds only an Anthropic key, which covers the brain and neither STT nor TTS. Interfaces are written for cloud; defaults are what runs today. |
| D3 | Wake word, not hotkey | User preference: hands-free operation. |
| D4 | openWakeWord, not Picovoice | User preference: no third-party account. Consequence in R2 below. |
| D5 | Context gate before speaking | User preference: never interrupt badly. Slice 1 uses local signals only; calendar-awareness arrives in Slice 3. |
| D6 | SQLite + human-readable markdown journal | Structured data makes Slice 4 pattern queries tractable; the journal keeps history auditable and greppable without tooling. |
| D7 | Brief accountability partner, not coach or logger | User preference. Two-to-three exchanges keeps a twice-daily ritual survivable past week two. |
| D8 | Anthropic SDK **Tool Runner**, not the Claude Agent SDK | The Agent SDK is Claude Code as a library — coding-shaped built-in tools and a permission model not built for "don't send that email without asking." ZEUS needs purpose-built tools with per-tool gating. |
| D9 | Single always-on daemon | The wake-word listener holds an open input stream; a second process wanting the microphone requires a lock and hand-off protocol. `KeepAlive` plus startup catch-up buys the resilience that a split design would provide. |
| D10 | Action log and generic job scheduler land in Slice 1 | A dashboard can only display history recorded from the beginning, and the watchman needs its own cadence. Retrofitting either is expensive; adding them now is cheap. |
| D11 | Watchman monitors *exposure*, not "is my IP tracked" | Whether a remote server logs an IP is invisible from the client — no signal returns. The buildable equivalent is monitoring exposure surface and local security posture. |

---

## 3. Verified environment

Every row below was **executed and confirmed** on the target machine on 2026-08-05, not
assumed. The design depends on these; re-verify if the machine changes.

| Property | Value |
|---|---|
| OS | macOS 12.7.6 (Monterey), build 21H1320 |
| Architecture | x86_64 (Intel) |
| System Python | 3.14.6 — **unusable**, no wheels for the audio stack |
| Package manager | `uv` 0.11.7 — provisions standalone CPython, no Homebrew needed |
| Target Python | 3.12.13 (via `uv python install 3.12`) |
| Homebrew | **Absent** |
| `ffmpeg` / `sox` | **Absent** — `av` wheels bundle FFmpeg, so neither is required |
| Microphone | Built-in, 2 input channels @ 44100 Hz |
| TTS | `/usr/bin/say` — produced a 179 KB AIFF in test |
| Playback | `/usr/bin/afplay` |
| Idle detection | `ioreg -c IOHIDSystem` → `HIDIdleTime` (nanoseconds) — works, zero dependencies |
| Focus/DND | `~/Library/DoNotDisturb/DB/` exists; `Assertions.json` absent while no Focus active |
| Automation hooks | `/usr/bin/osascript`, `/usr/bin/shortcuts` |

### 3.1 Dependency resolution (verified via `uv pip install --dry-run` on CPython 3.12)

| Package | Version | Role |
|---|---|---|
| `sounddevice` | 0.5.5 | Audio capture; bundles PortAudio |
| `openwakeword` | 0.6.0 | Wake-word detection |
| `onnxruntime` | 1.19.2 | openWakeWord inference backend |
| `faster-whisper` | 1.2.1 | Speech-to-text |
| `ctranslate2` | 4.8.1 | faster-whisper inference engine |
| `av` | 18.0.0 | Audio decoding; bundles FFmpeg |
| `tokenizers` | 0.23.1 | faster-whisper tokenizer |
| `numpy` | 2.5.1 | Frame handling |
| `anthropic` | latest | Claude brain, Tool Runner |
| `pyobjc-framework-quartz` | 12.2.1 | Screen-lock detection |

`pvporcupine` 4.0.3 also resolves and is retained as a documented fallback (R2).

---

## 4. Architecture

### 4.1 Process model

One process: `zeusd`. Managed by a macOS **LaunchAgent** at
`~/Library/LaunchAgents/com.zeus.daemon.plist` with `RunAtLoad` and `KeepAlive`. It owns
the microphone, wake-word detection, the scheduler, and the conversation loop.

The LaunchAgent invokes the venv interpreter by **absolute path**. Nothing depends on
shell initialization, `PATH`, or the user's profile.

### 4.2 Runtime isolation

`uv` creates a project-local virtual environment on CPython 3.12.13. The system Python
3.14 is never used — no wheels exist for the audio stack on that version, and this is the
single largest environmental risk the design removes.

### 4.3 Downtime recovery

`KeepAlive` restarts the daemon after a crash. Restarting is not sufficient on its own: a
laptop closed at 11:00 would silently skip a day. So on **every** startup ZEUS runs a
catch-up pass — it reads the `heartbeat` row, computes which scheduled jobs came due since
that timestamp, and fires anything missed (subject to §9.2 staleness rules).

### 4.4 Data flow

```
   ┌──────────────────────────────────────────────┐
   │  CoreAudio input (16 kHz mono int16)         │
   └────────────────────┬─────────────────────────┘
                        │  single stream, opened once
                  audio.MicStream
                   │           │
        ring buffer (3 s)   live frame queue
                   │           │
                   │           ▼
                   │    audio.Activator ──► wake event
                   │           │
                   └──────────►│ capture reaches BACKWARDS into
                               │ the ring buffer so the utterance
                               │ start is not clipped
                               ▼
                        audio.Endpointer  (energy + silence timeout)
                               │
                               ▼  utterance bytes
                        stt.Transcriber ──► text
                               │
                               ▼
                      brain.Conversation  (Tool Runner, claude-opus-5)
                          │         │
                          │         └──► memory.Store + memory.Journal
                          ▼
                        tts.Speaker ──► say | afplay ──► speakers
```

---

## 5. Components and interfaces

Each unit has one purpose, a defined interface, and is testable without hardware.

```
zeusd                       daemon: wiring, supervision, lifecycle
│
├── audio.MicStream         sole owner of the input stream
├── audio.Activator    ◄──  protocol
│     ├── WakeWordActivator (openWakeWord)      default
│     └── HotkeyActivator                       fallback
├── audio.Endpointer        energy + silence timeout → utterance boundary
│
├── stt.Transcriber    ◄──  protocol
│     ├── LocalWhisper      (faster-whisper, base.en, int8)   default
│     └── CloudSTT          (requires key)                    later
│
├── tts.Speaker        ◄──  protocol
│     ├── MacSay           (`say` → `afplay`)                 default
│     └── CloudTTS          (requires key)                    later
│
├── brain.Conversation      Tool Runner loop
├── brain.prompts           system prompt + check-in scripts
│
├── memory.Store            SQLite
├── memory.Journal          markdown, one file per day
│
├── context.Presence        screen lock · idle · Focus · call apps → verdict
└── schedule.Scheduler      generic recurring jobs + catch-up
```

### 5.1 Interface contracts

```python
class Activator(Protocol):
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def events(self) -> Iterator[ActivationEvent]: ...   # blocks until activation

class Transcriber(Protocol):
    def transcribe(self, pcm: bytes, sample_rate: int) -> str: ...   # "" if unintelligible

class Speaker(Protocol):
    def say(self, text: str) -> None: ...   # blocks until playback completes
    def stop(self) -> None: ...
```

Two implementations of each exist from day one. That is what makes D2 real rather than
aspirational: adding an OpenAI or ElevenLabs key later is a `config.toml` edit, not a
refactor.

### 5.2 `MicStream` — the one subtle component

The wake detector needs a *continuous* 16 kHz feed. Utterance capture needs the *same*
feed, on demand. Opening two streams is not an option, so the stream is opened once and
fans out.

`MicStream` additionally maintains a **3-second ring buffer**. When the wake word fires,
capture reaches *backwards* into that buffer before reading forward. Without this, the
first words after the keyword are lost while the detector is still deciding — "Zeus,
what's my battery" arrives as "battery". This is the standard wake-word clipping bug and
it is designed out rather than patched later.

Format throughout: **16 kHz, mono, int16**. CoreAudio resamples from the device's native
44.1 kHz. openWakeWord consumes 1280-sample (80 ms) chunks; Whisper consumes the assembled
utterance.

---

## 6. Data model

```
~/.zeus/
├── config.toml          times, voice, providers, retention, thresholds
├── zeus.db              SQLite (WAL mode)
├── journal/
│   └── 2026-08-05.md    human-readable daily log
├── models/              whisper + wake-word weights
└── logs/
    └── zeusd.log
```

### 6.1 WAL mode is required, not optional

The daemon writes continuously; the Slice 2 dashboard reads concurrently. Under SQLite's
default rollback journal those block each other, and a dashboard page load would stall the
voice loop. WAL permits concurrent readers alongside one writer. Combined with a
`busy_timeout`, this is what keeps the two from interfering.

### 6.2 Schema

| Table | Purpose | Columns |
|---|---|---|
| `goals` | the ritual's substance | `id`, `date`, `text`, `status`, `set_at`, `reviewed_at`, `notes` |
| `checkins` | did the ritual fire, and what happened | `id`, `kind`, `scheduled_for`, `fired_at`, `outcome`, `attempts` |
| `actions` | every tool call — the dashboard's spine | `id`, `ts`, `conversation_id`, `tool`, `args_json`, `result_json`, `ok`, `duration_ms`, `error` |
| `conversations` | session envelope | `id`, `started_at`, `ended_at`, `trigger` |
| `messages` | transcript | `id`, `conversation_id`, `role`, `content`, `ts` |
| `facts` | substrate for Slice 4 pattern learning | `id`, `key`, `value`, `learned_at`, `source` |
| `jobs` | generic recurring scheduler | `name`, `schedule`, `last_run_at`, `next_run_at`, `enabled` |
| `heartbeat` | single row; powers catch-up | `ts` |

Enumerations:

- `goals.status` ∈ `pending` · `done` · `partial` · `missed` · `carried`
- `checkins.kind` ∈ `morning` · `evening`
- `checkins.outcome` ∈ `answered` · `no_answer` · `deferred` · `skipped`
- `conversations.trigger` ∈ `wake` · `schedule`

### 6.3 Two correctness rules

**Timestamps are stored in UTC; schedules are expressed in local wall-clock time.**
"11:00" must still mean 11:00 to the user after a daylight-saving transition. The scheduler
resolves local time through `zoneinfo` against the system timezone; the database stores
ISO-8601 UTC everywhere. Mixing these up produces a bug that only appears twice a year.

**Raw audio is never written to disk.** Frames are transcribed and discarded. Transcripts
persist under a retention window set in `config.toml`. Transcripts are the most sensitive
data ZEUS holds, so their lifetime is an explicit dial rather than a default the user has
to discover.

---

## 7. Conversation design

### 7.1 Model configuration

| Setting | Value | Reason |
|---|---|---|
| Model | `claude-opus-5` | Default per current guidance; handles Slice 2–3 tool work without a later migration |
| Thinking | adaptive (**on**) | Disabling thinking on Opus 5 has a documented failure mode where a tool call is emitted as visible text and silently never runs |
| Effort | `low` (check-ins), `medium` (ad-hoc) | Opus 5 performs strongly at low effort; this is the primary latency and cost lever |
| Caching | `cache_control` on system prompt + tool definitions | Largest cost lever. Comfortably exceeds Opus 5's 512-token cache minimum |
| Streaming | on | TTS begins at the **first complete sentence** rather than after the full reply — materially changes perceived responsiveness |

Expected cost at this volume: a few dollars per month.

### 7.2 Check-in scripts

**Morning (11:00).** Ask for the day's single goal. If the answer is vague, ask **once**
for something concrete, then accept whatever comes back. Confirm and end. Hard ceiling of
three exchanges.

**Evening (21:00).** Recall the morning's goal, ask whether it happened, record the
outcome without judgment, offer to carry it forward **once**, and end. Hard ceiling of
three exchanges.

The ceilings are a hard requirement, not a stylistic preference. An accountability ritual
that runs long is abandoned.

### 7.3 Half-duplex — a deliberate v1 limitation

The wake detector is **muted while ZEUS is speaking**. An open microphone hears `say`
through the speakers and re-triggers on ZEUS's own voice. The cost is that the user cannot
interrupt mid-sentence. Fixing that requires acoustic echo cancellation, which is a project
in its own right and is deferred.

---

## 8. Context gate

Evaluated at the instant a check-in fires. Slice 1 uses **local signals only** — no
integration is required.

| Signal | Source |
|---|---|
| Screen locked | `Quartz.CGSessionCopyCurrentDictionary()` → `CGSSessionScreenIsLocked` |
| Idle time | `ioreg -c IOHIDSystem` → `HIDIdleTime` (ns) |
| Focus / DND active | presence of `~/Library/DoNotDisturb/DB/Assertions.json` with assertion records |
| Call in progress | configurable process-name list (`zoom.us`, `Microsoft Teams`, `FaceTime`, …) |

Verdict:

```
screen locked  OR  idle > 15 min      →  DEFER    see §9.3
Focus/DND on   OR  call app running   →  NOTIFY   silent notification; speaks on click or wake word
otherwise                             →  SPEAK    chime · 2 s pause · prompt · mic open 30 s
```

The call-app check is a **heuristic**, and is documented as such. Browser-based meetings
are undetectable this way. Genuine meeting-awareness arrives in Slice 3 via the calendar,
feeding this same decision point — the gate is designed to take that extra signal without
restructuring.

---

## 9. Scheduling

### 9.1 Generic jobs

The scheduler runs named jobs from the `jobs` table. `jobs.schedule` holds a **five-field
cron expression** (`minute hour day-of-month month day-of-week`), evaluated against local
wall-clock time per §6.3. Cron is chosen over a bespoke format because the Slice 4 watchman
needs interval cadences (`*/30 * * * *`) that a "daily at HH:MM" format cannot express.

Slice 1 registers exactly two jobs:

| `name` | `schedule` | Handler |
|---|---|---|
| `checkin_morning` | `0 11 * * *` | Morning check-in (§7.2) |
| `checkin_evening` | `0 21 * * *` | Evening check-in (§7.2) |

`config.toml`'s `morning` / `evening` values are the user-facing form; the daemon
translates them into these cron expressions on startup and reconciles the `jobs` table. The
mechanism is general so later slices register their own jobs without the scheduler being
rewritten.

### 9.2 Catch-up

On startup, for each enabled job, compute occurrences between `heartbeat.ts` and now:

- **Morning check-in missed, still the same day** → fire now, subject to the context gate.
- **Morning check-in missed, day has rolled over** → do not fire; record `outcome=skipped`.
- **Evening check-in missed** → record `outcome=skipped`; do not ask about yesterday today.

The rule: never replay a check-in whose moment has genuinely passed. A goal question at
15:00 is useful; at 09:00 the next day it is noise.

### 9.3 Timeouts and retries

There are **two distinct retry paths**, with different causes, cadences, and limits. They
are specified here and nowhere else; §8 defers to this section.

| Path | Cause | Interval | Attempts | On exhaustion |
|---|---|---|---|---|
| **DEFER** | Context gate returned `DEFER` — user is away or the screen is locked | `defer_retry_after` (20 min) | `max_defer_retries` (3) | Morning → fold into the evening check-in. Evening → record `outcome=skipped`. |
| **NO_ANSWER** | ZEUS spoke, but the listen window elapsed in silence | `no_answer_retry_after` (30 min) | `max_no_answer_retries` (1) | Morning → fold into the evening check-in. Evening → record `outcome=skipped`. |

"Fold into the evening check-in" means the evening script opens by asking for the goal
that was never captured, then proceeds normally. It does **not** mean firing a second
prompt at 21:00.

Both paths increment `checkins.attempts`. A check-in still awaiting a retry carries
`outcome=deferred`; the value is rewritten to `answered`, `no_answer`, or `skipped` when
the sequence terminates.

The listen window itself is `listen_timeout` (30 s) of silence after the prompt finishes
playing.

---

## 10. Failure handling

The governing rule: **fail loudly, never pretend.** A voice assistant that silently stops
listening is worse than one that says it is broken.

| Failure | Behavior |
|---|---|
| Microphone denied by TCC, or stream returns silence | Startup self-test detects it → error log + macOS notification → **degrade to notification-only mode**; check-ins still fire, without voice |
| Whisper model absent | Download with progress on first run; if offline, refuse to start with an explicit message |
| Empty or unintelligible transcript | "I didn't catch that" **once**, then end the turn cleanly |
| Anthropic API error or rate limit | SDK retries; on hard failure speak a canned line and mark the check-in `deferred`, **not** `missed` |
| `stop_reason == "refusal"` | Checked **before** reading `content`; unchecked, the code indexes an empty array and crashes |
| Tool call raises | Return `is_error: true` so the model can adapt; log to `actions` with `ok=0` |
| Daemon crash | `KeepAlive` restarts; catch-up handles what was missed |
| DB locked / disk full | WAL + `busy_timeout`; degrade to journal-only writes |

### 10.1 Startup self-test

Before entering the main loop the daemon captures one second of audio and asserts non-zero
RMS. This exists specifically to catch the TCC failure mode described in R1, where the
stream opens successfully and returns pure silence forever. Without the test, ZEUS appears
to be running perfectly while being deaf.

---

## 11. Configuration

`~/.zeus/config.toml`, read at startup:

```toml
[schedule]
timezone                 = "system"   # resolved via zoneinfo
morning                  = "11:00"
evening                  = "21:00"
defer_retry_after        = "20m"      # §9.3 DEFER path
max_defer_retries        = 3
no_answer_retry_after    = "30m"      # §9.3 NO_ANSWER path
max_no_answer_retries    = 1

[audio]
sample_rate   = 16000
ring_seconds  = 3
silence_timeout = "1.5s"
listen_timeout  = "30s"

[wake]
provider   = "openwakeword"
model      = "hey_jarvis"    # replace with a custom zeus.onnx when trained

[stt]
provider   = "local_whisper"
model      = "base.en"
compute    = "int8"

[tts]
provider   = "mac_say"
voice      = "Alex"

[brain]
model            = "claude-opus-5"
effort_checkin   = "low"
effort_adhoc     = "medium"

[context]
idle_threshold = "15m"
call_apps      = ["zoom.us", "Microsoft Teams", "FaceTime"]

[privacy]
transcript_retention_days = 90
```

---

## 12. Security and privacy

- **All data is local.** `~/.zeus/` is the complete footprint; deleting it removes
  everything ZEUS knows.
- **Audio is never persisted.** Only transcripts, subject to `transcript_retention_days`.
- **What leaves the machine in Slice 1:** transcribed text and ZEUS's replies, to the
  Anthropic API. Audio does not, because STT runs locally. This is a stronger privacy
  position than the "cloud-first" choice implied, and is a consequence of D2.
- **The API key is read from the environment**, never written into config or source.
- **Destructive tools are gated.** Any tool marked destructive causes ZEUS to state the
  action aloud and require explicit spoken confirmation before executing.
  **Amended 2026-08-07 — the gate moves to Slice 2.** This clause originally
  said the mechanism was built in Slice 1, ahead of its first caller, so that
  Slices 2 and 3 would not each invent one. The final whole-branch review
  found it absent from the code *and* from the implementation plan: no
  `destructive` marker, no confirmation path, nothing. Slice 1 ships no
  destructive tool, so nothing is exploitable — but the spec was claiming a
  mechanism that did not exist, which is worse than an acknowledged gap.
  Ruled: build the gate in Slice 2, beside the first tool that needs it. A
  permission mechanism with zero callers is hard to design well and easy to
  get subtly wrong; the original reasoning (don't invent it three times) is
  still honoured as long as Slice 2 builds it once, before its first
  destructive tool ships.
- **The dashboard binds to `127.0.0.1` only** (Slice 2) and is never exposed to the local
  network.

---

## 13. Testing

The protocol boundaries in §5 exist largely so that **no automated test requires a
microphone, speakers, or a network call.**

Fakes: `FakeActivator`, `FakeTranscriber` (scripted strings), `FakeSpeaker` (records what
was said), `FakeClock`, and a stubbed brain.

| Test | Asserts |
|---|---|
| Golden path — morning | Clock hits 11:00 → prompt spoken → transcript fed → `goals` row written, journal line appended |
| Golden path — evening | Prior goal recalled → outcome recorded with correct `status` |
| Context gate | Table-driven across every combination of locked / idle / Focus / call-app → expected verdict |
| Catch-up | Stale heartbeat → correct set of missed jobs identified and §9.2 rules applied |
| Ring buffer | Utterance assembled from a wake event includes pre-roll audio (no clipping) |
| Retry paths (§9.3) | DEFER retries 3× at 20 min; NO_ANSWER retries 1× at 30 min; `checkins.attempts` increments; exhaustion folds a morning check-in into the evening one and marks an evening one `skipped` |
| Failure paths | Empty transcript, API error, refusal stop reason, locked DB each produce the §10 behavior |
| Timezone | A check-in across a DST boundary still fires at local 11:00 |

**`zeus selftest`** is a manual command — capture one second, transcribe it, speak a line —
deliberately excluded from the automated suite because it is the one thing that genuinely
requires hardware.

---

## 14. Roadmap beyond Slice 1

Sketched for context. Each gets its own spec.

| Slice | Contents |
|---|---|
| **2 — Local tools + web dashboard** | Device health, file read/edit, music control, shell and Shortcuts automation; the `127.0.0.1` dashboard showing goals, streaks, the action log, and tool health |
| **3 — Cloud integrations** | Calendar, email, web search and browsing via MCP; calendar-awareness feeds the §8 context gate |
| **4 — Watchman + menu bar + patterns** | Exposure and posture monitoring, outbound traffic audit, LAN device scan, breach lookup; menu bar app; pattern learning over accumulated history |

---

## 15. Open risks

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| **R1** | **Microphone permission (TCC).** macOS attributes microphone access to the *responsible process*, and a LaunchAgent-spawned Python binary is an awkward subject. The failure is silent: the stream opens and returns pure silence indefinitely. | **High** — most likely first-run failure | First launch performed manually from Terminal to trigger a real prompt; §10.1 startup self-test converts a silent failure into a loud one |
| **R2** | **No "Zeus" wake-word model.** openWakeWord ships no such model; out of the box ZEUS wakes to `hey_jarvis`. | Medium — cosmetic but contrary to intent | Wake model is a config path to an ONNX file. A custom `zeus.onnx` can be trained free in-browser (~1 hour) and dropped in with no code change. `pvporcupine` remains a fallback if openWakeWord accuracy disappoints. |
| **R3** | **Focus/DND detection is inferred.** `Assertions.json` was confirmed *absent* while no Focus was active; the converse — that it appears with assertion records while Focus **is** active — was **not** verified, because that requires toggling Focus interactively. | Medium | First implementation task is to verify both states by toggling Focus manually. If the inference is wrong, the gate degrades to idle + lock + call-app signals only, which still function. |
| **R4** | **Local Whisper latency on Intel CPU.** `base.en` at int8 adds seconds per utterance versus a cloud API. | Medium — a UX annoyance, not a defect | Accepted consequence of D2. `Transcriber` is a protocol; adding a cloud key later is a config edit. Model size is configurable if `base.en` proves too slow or too inaccurate. |
| **R5** | **Wake-word false triggers.** openWakeWord is less accurate than commercial alternatives, and pre-trained models are more prone to spurious activation. | Medium | Detection threshold is configurable. `HotkeyActivator` ships alongside as a deterministic path for debugging and as a permanent alternative. |
| **R6** | **No barge-in.** §7.3 — the user cannot interrupt ZEUS mid-sentence. | Low | Accepted for v1. Check-in replies are short by design. Revisit only if it proves irritating in daily use. |
| **R7** | **Watchman alert noise** (Slice 4). The selected scope includes outbound-traffic auditing and LAN scanning, both of which produce false alarms on a normal home network. | Low — deferred | Flagged during design and accepted by the user. Slice 4 must include a baselining period and tunable thresholds before alerts are enabled. |
