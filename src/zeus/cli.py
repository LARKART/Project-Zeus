"""ZEUS command line. See spec §13 — `selftest` is the only hardware path."""
from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import plistlib
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

    BUILT WITH plistlib, NOT AN f-STRING. Interpolating paths into XML by
    hand breaks on any path containing an XML metacharacter: a checkout
    under /Users/me/R&D/zeus produced "ExpatError: not well-formed" and a
    plist launchctl refuses to load, while cmd_install_agent still printed
    "Wrote ..." and the load instructions. A path containing < produced
    "mismatched tag".

    It is also a security property, not just robustness. With hand-built
    XML, a crafted path can CLOSE the surrounding element and inject its
    own — a review demonstrated an env_path that produced a *valid* plist
    carrying EnvironmentVariables -> ANTHROPIC_API_KEY: sk-ant-pwned. That
    turns "the plist never holds the key" from an invariant into a property
    of whichever inputs the tests happen to use. plistlib escapes by
    construction, so the guarantee holds for every input.
    """
    return plistlib.dumps(
        {
            "Label": LABEL,
            "ProgramArguments": [str(python_path), "-m", "zeus.cli", "run"],
            "EnvironmentVariables": {"ZEUS_ENV_FILE": str(env_path)},
            "RunAtLoad": True,
            "KeepAlive": True,
            "StandardOutPath": str(log_path),
            "StandardErrorPath": str(log_path),
        }
    ).decode()


def _load_env_file(path: Path) -> int:
    """Load KEY=VALUE lines into os.environ. Returns how many were set.

    Ten lines of stdlib instead of python-dotenv: Global Constraints forbid
    new dependencies. Deliberately minimal — no interpolation, no `export`
    prefix, no multi-line values. It exists for one variable.

    Existing environment wins, so running the daemon by hand from a shell
    that already exported ANTHROPIC_API_KEY behaves the same as launchd
    loading it from the file.
    """
    # THE GUARD GOES INSIDE THE TRY. is_file() is not the safe probe it looks
    # like: pathlib swallows only ENOENT/ENOTDIR/EBADF/ELOOP and RE-RAISES
    # EACCES, EPERM and ENAMETOOLONG. Verified — a file inside a mode-000
    # directory raises PermissionError straight out of is_file(). So an
    # untraversable parent, a TCC-protected folder (a LaunchAgent statting
    # ~/Documents gets EPERM), or a >1024-char path all escaped this function
    # from the very line written to make it safe.
    #
    # Worse now that cmd_doctor calls this too: the health oracle would crash
    # on exactly the condition it exists to diagnose, and cmd_run would
    # crash-loop under KeepAlive:true.
    #
    # is_file() over exists() is still right — a directory passes exists() and
    # then read_text() raises IsADirectoryError, and ZEUS_ENV_FILE="" resolves
    # to Path(".") and hits that. It just has to be guarded like everything
    # else.
    try:
        if not path.is_file():
            return 0
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        # A mode-000 file, a non-UTF-8 file, a dangling symlink. Degrade the
        # way cmd_run already degrades when the key is simply absent: log and
        # carry on. Raising here would propagate out of cmd_run, and under
        # launchd with KeepAlive:true that is a respawn loop.
        log.error("could not read %s; continuing without it", path, exc_info=True)
        return 0
    loaded = 0
    for line in text.splitlines():
        # A UTF-8 BOM at the top of the file would otherwise become part of
        # the first key's name if not stripped here.
        line = line.strip().lstrip("\ufeff")
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        # Skip keys with whitespace instead of setting them. `export FOO=bar`
        # would otherwise create a variable literally named "export FOO" and
        # report success while the real key stays unset — the worst shape,
        # since it looks like it worked. The docstring says `export` is
        # unsupported; this makes the code agree.
        value = value.strip().strip("'\"")
        # A NUL anywhere is fatal to os.environ: CPython rejects it with
        # ValueError, which is NOT an OSError and so slips past the guard
        # above — and this assignment sits outside that try in any case.
        #
        # It is reachable, because NUL is valid UTF-8 and read_text() passes
        # it straight through. A file saved as UTF-16LE — by `iconv -t
        # UTF-16LE`, an editor's "Unicode" option, or a truncated write —
        # decodes cleanly and then EVERY assignment raises, uncaught, out of
        # cmd_run and cmd_doctor. That is I2's failure mode again: a respawn
        # loop under KeepAlive:true, and a health oracle that dies on the
        # condition it exists to diagnose.
        #
        # Skip the bad line and keep the good ones, exactly as the
        # whitespace-key skip does — one corrupt line must not cost you a
        # valid ANTHROPIC_API_KEY on the next.
        if (not key or " " in key or "\t" in key
                or "\x00" in key or "\x00" in value):
            continue
        if key not in os.environ:
            os.environ[key] = value
            loaded += 1
    return loaded


def _probe_python() -> bool:
    """A function, not an inline expression, so a test can monkeypatch it.

    Inline, the check can only ever be tested against the interpreter
    running the tests — so `== (3, 12)` and `>= (3, 11)` are
    indistinguishable on a 3.12 venv, and the test that exists to pin the
    floor pins nothing.
    """
    return sys.version_info[:2] >= (3, 11)


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


def _probe_env_file(path: Path) -> bool:
    """is_file() re-raises EACCES/EPERM/ENAMETOOLONG — see _load_env_file."""
    try:
        return path.is_file()
    except OSError:
        return False


def _probe_sounddevice() -> bool:
    """Hard import inside MicStream.start(); without it ZEUS cannot hear."""
    return importlib.util.find_spec("sounddevice") is not None


def _probe_openwakeword() -> bool:
    """Hard import when the wake detector loads its model."""
    return importlib.util.find_spec("openwakeword") is not None


def cmd_doctor(config: Config) -> int:
    # Load the env file FIRST, exactly as cmd_run does. Without this, doctor
    # reads only os.environ: a key exported in ~/.zshrc makes doctor report
    # all-OK, and then the LaunchAgent — which inherits no shell environment —
    # starts with no key and fails every brain call, with one line in
    # zeusd.log as the only signal. Being blind to that is being blind to the
    # exact failure ZEUS_ENV_FILE exists to prevent.
    env_file = Path(os.environ.get("ZEUS_ENV_FILE", config.env_path))
    _load_env_file(env_file)
    checks = [
        ("python", f"{sys.version_info.major}.{sys.version_info.minor}",
         _probe_python()),
        ("say", "/usr/bin/say", _probe_say()),
        ("ANTHROPIC_API_KEY", "environment or env file", _probe_api_key()),
        ("transcriber", "faster_whisper", _probe_transcriber()),
        ("audio input", "sounddevice", _probe_sounddevice()),
        ("wake word", "openwakeword", _probe_openwakeword()),
        ("zeus root", str(config.root), config.root.exists()),
        ("database", str(config.db_path), config.db_path.exists()),
        # env_file, NOT config.env_path: ZEUS_ENV_FILE overrides it, so
        # reporting the default while having loaded the override is the same
        # lying-oracle bug this check was added to fix.
        ("env file", str(env_file), _probe_env_file(env_file)),
        ("afplay", "/usr/bin/afplay", _probe_afplay()),
        ("LaunchAgent", str(PLIST_PATH), PLIST_PATH.exists()),
    ]
    print("ZEUS environment report\n")
    healthy = True
    for name, detail, ok in checks:
        print(f"  {'OK  ' if ok else 'FAIL'}  {name:<20} {detail}")
        # FATAL = things that stop ZEUS working. afplay is NOT one: nothing
        # in src/ invokes it (TTS goes through /usr/bin/say), so it is
        # reported and ignored. sounddevice and openwakeword ARE fatal —
        # both are hard imports on the capture path, and neither was checked
        # at all before. A missing database, env file, or LaunchAgent is
        # expected before first run.
        if not ok and name in {
            "python", "say", "ANTHROPIC_API_KEY", "transcriber",
            "audio input", "wake word",
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
    # ASK. MacSay.say() swallows every exception and ignores the return code —
    # deliberately, so one failed sentence cannot abort a check-in — which
    # means nothing downstream can tell whether a sound was actually produced.
    # A misspelled or uninstalled voice exits 1 silently, and this function's
    # entire purpose is detecting exactly that. selftest is interactive by
    # design, so asking is the honest oracle.
    #
    # RuntimeError as well as EOFError: input() raises EOFError at end of
    # input, but RuntimeError("lost sys.stdin") when fd 0 is closed outright,
    # as in `zeus selftest 0<&-`. Either way, treat it as "not heard" — a
    # self-test that cannot confirm must not claim success.
    try:
        heard_it = input("Did you hear that? [y/N] ").strip().lower()
    except (EOFError, RuntimeError):
        heard_it = ""
    if heard_it not in ("y", "yes"):
        print(
            "FAIL: speech synthesis did not produce audible output.\n"
            "  Check System Settings > Sound > Output, and that the voice\n"
            f"  {config.tts.voice!r} is installed (System Settings >\n"
            "  Accessibility > Spoken Content > System Voice > Manage Voices)."
        )
        return 1
    print("OK: speech synthesis worked.")
    return 0


def cmd_install_agent(config: Config) -> int:
    # sys.executable AS-IS. It is already absolute, and NOT .resolve()d:
    # .resolve() follows the venv symlink out of the venv, to the real
    # interpreter behind it. Measured here:
    #   sys.executable  .../.venv/bin/python                -> imports zeus
    #   .resolve()      ~/.local/share/uv/python/.../python3.12
    #                                                       -> ModuleNotFoundError
    # A venv works by the PATH the interpreter is invoked through, so the
    # resolved binary has none of the venv's site-packages. The plist would
    # name an interpreter that cannot run `-m zeus.cli`, and KeepAlive:true
    # turns that into an infinite respawn loop.
    python_path = Path(sys.executable)
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
