"""ZEUS command line. See spec §13 — `selftest` is the only hardware path."""
from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import sys
from pathlib import Path

from zeus.config import Config, load_config

log = logging.getLogger(__name__)

LABEL = "com.zeus.daemon"
PLIST_PATH = Path.home() / "Library/LaunchAgents" / f"{LABEL}.plist"


def launch_agent_plist(python_path: Path, log_path: Path, env_path: Path) -> str:
    """Generate the LaunchAgent plist.

    The interpreter is referenced by absolute path so nothing depends on
    shell initialisation or PATH (spec §4.1).

    ENV_PATH, NOT THE KEY ITSELF. launchd does not read .env, and a
    LaunchAgent inherits none of your shell environment — so without this
    the daemon starts with no ANTHROPIC_API_KEY and every brain call fails
    at runtime, quietly. The obvious fix is an EnvironmentVariables entry
    holding the key, but the spec says the key lives in the environment
    only and is "never written to config or source", and a plist in
    ~/Library/LaunchAgents is config. So the plist carries only a PATH to
    the key file; cmd_run loads it (see _load_env_file). launchctl setenv
    was the other candidate and was rejected: it does not survive a reboot,
    so ZEUS would come back deaf to its own API after every restart.
    """
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key><string>{LABEL}</string>
    <key>ProgramArguments</key>
    <array>
        <string>{python_path}</string>
        <string>-m</string>
        <string>zeus.cli</string>
        <string>run</string>
    </array>
    <key>EnvironmentVariables</key>
    <dict>
        <key>ZEUS_ENV_FILE</key><string>{env_path}</string>
    </dict>
    <key>RunAtLoad</key><true/>
    <key>KeepAlive</key><true/>
    <key>StandardOutPath</key><string>{log_path}</string>
    <key>StandardErrorPath</key><string>{log_path}</string>
</dict>
</plist>
"""


def _load_env_file(path: Path) -> int:
    """Load KEY=VALUE lines into os.environ. Returns how many were set.

    Ten lines of stdlib instead of python-dotenv: Global Constraints forbid
    new dependencies. Deliberately minimal — no interpolation, no `export`
    prefix, no multi-line values. It exists for one variable.

    Existing environment wins, so running the daemon by hand from a shell
    that already exported ANTHROPIC_API_KEY behaves the same as launchd
    loading it from the file.
    """
    if not path.exists():
        return 0
    loaded = 0
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key and key not in os.environ:
            os.environ[key] = value.strip().strip("'\"")
            loaded += 1
    return loaded


def _probe_say() -> bool:
    return Path("/usr/bin/say").exists()


def _probe_afplay() -> bool:
    return Path("/usr/bin/afplay").exists()


def _probe_api_key() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def _probe_transcriber() -> bool:
    """Is the speech-to-text backend actually importable?

    LocalWhisper.transcribe() catches Exception and returns "", so a missing
    faster_whisper or a corrupt model is indistinguishable from a quiet room
    — ZEUS just never hears anything and says so to nobody. This is the
    cheap half of the check; cmd_selftest does the real one by transcribing
    audio it actually captured.
    """
    return importlib.util.find_spec("faster_whisper") is not None


def cmd_doctor(config: Config) -> int:
    # >= (3, 11), not == (3, 12): Global Constraints say Python 3.11+, so
    # exact equality would report a FAILURE on 3.11 or 3.13 while ZEUS runs
    # perfectly well on both. A doctor that lies about health is worse than
    # no doctor.
    checks = [
        ("python", f"{sys.version_info.major}.{sys.version_info.minor}",
         sys.version_info[:2] >= (3, 11)),
        ("say", "/usr/bin/say", _probe_say()),
        ("afplay", "/usr/bin/afplay", _probe_afplay()),
        ("ANTHROPIC_API_KEY", "environment", _probe_api_key()),
        ("transcriber", "faster_whisper", _probe_transcriber()),
        ("zeus root", str(config.root), config.root.exists()),
        ("database", str(config.db_path), config.db_path.exists()),
        ("LaunchAgent", str(PLIST_PATH), PLIST_PATH.exists()),
    ]
    print("ZEUS environment report\n")
    healthy = True
    for name, detail, ok in checks:
        print(f"  {'OK  ' if ok else 'FAIL'}  {name:<20} {detail}")
        # A missing database or LaunchAgent is expected before first run.
        if not ok and name in {
            "python", "say", "afplay", "ANTHROPIC_API_KEY", "transcriber",
        }:
            healthy = False
    print()
    if not healthy:
        print("Not ready. Fix the FAIL lines above.")
    return 0 if healthy else 1


def cmd_selftest(config: Config) -> int:
    """Capture, TRANSCRIBE, and speak. Requires real hardware — never in CI.

    The transcription step is the point, not a bonus. audio_self_test only
    proves frames are arriving with non-zero energy; it says nothing about
    whether those frames become words. LocalWhisper.transcribe() swallows
    every exception and returns "", so a missing model file, an
    uninstalled faster_whisper, or a corrupt download all present as ZEUS
    silently never understanding anything. Printing what came back is the
    only way a user can tell "you said nothing" from "I am broken".
    """
    from zeus.audio.endpointer import Endpointer, capture_utterance
    from zeus.audio.mic import FRAME_SAMPLES, MicStream
    from zeus.daemon import audio_self_test
    from zeus.stt import build_transcriber
    from zeus.tts import build_speaker

    print("Capturing one second of audio — say something now...")
    mic = MicStream(config.audio)
    mic.start()
    try:
        if not audio_self_test(mic):
            print(
                "FAIL: the microphone produced no audio.\n"
                "  Grant microphone access to this interpreter in\n"
                "  System Preferences > Security & Privacy > Privacy > Microphone,\n"
                "  then run 'zeus selftest' again from Terminal."
            )
            return 1
        print("OK: microphone is producing audio.")

        print("Now say a short sentence, then stop...")
        frames_per_second = config.audio.sample_rate / FRAME_SAMPLES
        audio = capture_utterance(
            mic.frames(), Endpointer(config.audio), pre_roll=b"",
            listen_timeout_frames=int(10 * frames_per_second),
        )
    finally:
        mic.stop()

    if not audio:
        print("FAIL: captured no utterance — the endpointer heard only silence.")
        return 1
    transcriber = build_transcriber(config.stt, config.models_dir)
    heard = transcriber.transcribe(audio, config.audio.sample_rate)
    if not heard:
        print(
            "FAIL: audio was captured but transcription returned nothing.\n"
            "  The model may be missing or corrupt. Check that faster_whisper\n"
            f"  is installed and that {config.models_dir} holds the model."
        )
        return 1
    print(f'OK: transcription works — I heard "{heard}"')

    speaker = build_speaker(config.tts)
    speaker.say("ZEUS self test complete. I can hear you and you can hear me.")
    print("OK: speech synthesis worked.")
    return 0


def cmd_install_agent(config: Config) -> int:
    python_path = Path(sys.executable).resolve()
    config.log_path.parent.mkdir(parents=True, exist_ok=True)
    PLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    PLIST_PATH.write_text(
        launch_agent_plist(python_path, config.log_path, config.env_path)
    )
    print(f"Wrote {PLIST_PATH}\n")
    print("Load it with:")
    print(f"  launchctl unload {PLIST_PATH} 2>/dev/null")
    print(f"  launchctl load {PLIST_PATH}\n")
    if not config.env_path.exists():
        print(
            f"BEFORE loading it, put your API key in {config.env_path}:\n"
            f"  printf 'ANTHROPIC_API_KEY=sk-ant-...\\n' > {config.env_path}\n"
            f"  chmod 600 {config.env_path}\n"
            "launchd inherits none of your shell environment, so without this\n"
            "file the daemon starts fine and then fails on every request.\n"
        )
    print(
        "IMPORTANT: run 'zeus selftest' from Terminal FIRST so macOS prompts\n"
        "for microphone access. A LaunchAgent that has never been granted\n"
        "access opens the stream successfully and hears only silence."
    )
    return 0


def cmd_run(config: Config) -> int:
    from zeus.daemon import build_daemon

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # Under launchd the environment is empty but ZEUS_ENV_FILE points at the
    # key file (see launch_agent_plist). Run from a shell and the file is
    # usually absent while the variable is already exported — both paths end
    # with ANTHROPIC_API_KEY in os.environ, which is the only place the
    # Anthropic client reads it from.
    env_file = Path(os.environ.get("ZEUS_ENV_FILE", config.env_path))
    _load_env_file(env_file)
    if not _probe_api_key():
        log.error(
            "ANTHROPIC_API_KEY is not set and %s did not supply it. "
            "Every conversation will fail. Run 'zeus doctor'.", env_file,
        )
    build_daemon(config).run_forever()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="zeus", description="ZEUS voice assistant")
    parser.add_argument("--root", type=Path, default=None, help="ZEUS data directory")
    sub = parser.add_subparsers(dest="command")
    for name, help_text in [
        ("run", "run the daemon in the foreground"),
        ("selftest", "check the microphone and speakers (requires hardware)"),
        ("doctor", "print an environment report"),
        ("install-agent", "write the LaunchAgent plist"),
    ]:
        subparser = sub.add_parser(name, help=help_text)
        # `--root` is declared on the TOP-level parser only, but argparse
        # subparsers consume every token after the subcommand name and hand
        # them to the chosen subparser's own parser — one that knows nothing
        # about `--root`. So `zeus doctor --root DIR` (root after the
        # subcommand, the form every doctor/selftest/install-agent test and
        # every real invocation uses) raised "unrecognized arguments" and
        # SystemExit(2), while `zeus --root DIR doctor` (root before) worked.
        # Mirroring the flag onto each subparser fixes the common case.
        # default=argparse.SUPPRESS (not None) matters: subparsers re-apply
        # their own defaults onto the shared namespace after parsing, so a
        # plain default=None here would silently clobber a --root already
        # captured by the top-level parser when it appears BEFORE the
        # subcommand instead.
        subparser.add_argument(
            "--root", type=Path, default=argparse.SUPPRESS, help="ZEUS data directory"
        )

    # argparse RAISES SystemExit(2) on an unknown subcommand rather than
    # returning — so without this, main() only sometimes returns an int and
    # `main(["frobnicate"]) == 2` is unreachable. Catching it here keeps the
    # declared `main(argv) -> int` contract true on every path, which is
    # what makes main() callable as a library function and testable without
    # pytest.raises.
    try:
        args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    except SystemExit as exit_request:
        return int(exit_request.code or 0)
    if not args.command:
        parser.print_usage()
        return 2

    config = load_config(root=args.root) if args.root else load_config()
    return {
        "run": cmd_run,
        "selftest": cmd_selftest,
        "doctor": cmd_doctor,
        "install-agent": cmd_install_agent,
    }[args.command](config)


if __name__ == "__main__":
    raise SystemExit(main())
