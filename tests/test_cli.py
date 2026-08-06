import os
import plistlib
import sys
from pathlib import Path

import pytest

from zeus.cli import launch_agent_plist, main
from zeus.config import Config


ENV = Path("/Users/x/.zeus/env")

# Every probe cmd_doctor consults. Tests that want a clean "all healthy"
# baseline mock all of these to True and then flip the one under test.
ALL_PROBES = (
    "_probe_python",
    "_probe_say",
    "_probe_afplay",
    "_probe_api_key",
    "_probe_transcriber",
    "_probe_sounddevice",
    "_probe_openwakeword",
)


def _mock_probes(monkeypatch, cli, **overrides):
    """Mock every doctor probe to True, except the names given in overrides."""
    for name in ALL_PROBES:
        monkeypatch.setattr(cli, name, overrides.get(name, lambda: True))


def test_plist_is_valid_and_uses_absolute_paths():
    xml = launch_agent_plist(
        Path("/opt/zeus/.venv/bin/python"), Path("/tmp/z.log"), ENV
    )
    parsed = plistlib.loads(xml.encode())

    assert parsed["Label"] == "com.zeus.daemon"
    assert parsed["ProgramArguments"][0] == "/opt/zeus/.venv/bin/python"
    assert all(Path(a).is_absolute() for a in parsed["ProgramArguments"][:1])


def test_plist_enables_keepalive_and_runatload():
    parsed = plistlib.loads(
        launch_agent_plist(Path("/x/python"), Path("/tmp/z.log"), ENV).encode()
    )
    assert parsed["KeepAlive"] is True
    assert parsed["RunAtLoad"] is True


def test_plist_routes_both_streams_to_the_log():
    parsed = plistlib.loads(
        launch_agent_plist(Path("/x/python"), Path("/tmp/z.log"), ENV).encode()
    )
    assert parsed["StandardOutPath"] == "/tmp/z.log"
    assert parsed["StandardErrorPath"] == "/tmp/z.log"


def test_plist_invokes_the_module_not_a_shell():
    parsed = plistlib.loads(
        launch_agent_plist(Path("/x/python"), Path("/tmp/z.log"), ENV).encode()
    )
    assert parsed["ProgramArguments"][1:] == ["-m", "zeus.cli", "run"]


def test_plist_points_at_the_env_file_and_never_holds_the_key():
    """launchd inherits no shell environment, so the daemon needs SOME way
    to find the key — but the spec says the key is environment-only and
    never written to config, and this plist is config. The path is what the
    plist may carry; the secret is not."""
    xml = launch_agent_plist(Path("/x/python"), Path("/tmp/z.log"), ENV)
    parsed = plistlib.loads(xml.encode())
    assert parsed["EnvironmentVariables"]["ZEUS_ENV_FILE"] == str(ENV)
    assert "ANTHROPIC_API_KEY" not in xml
    assert "sk-ant" not in xml


def test_plist_escapes_xml_metacharacters_in_paths():
    """Round1 finding I1: hand-built XML broke on any path containing an XML
    metacharacter — a checkout under /Users/me/R&D/zeus produced
    "ExpatError: not well-formed", and cmd_install_agent still printed
    "Wrote ..." regardless. plistlib.dumps escapes by construction."""
    hostile = Path("/Users/me/R&D/<zeus>/.zeus/env")
    xml = launch_agent_plist(Path("/x/python"), Path("/tmp/z.log"), hostile)
    parsed = plistlib.loads(xml.encode())
    assert parsed["EnvironmentVariables"]["ZEUS_ENV_FILE"] == str(hostile)


def test_plist_survives_an_xml_injection_attempt_in_the_env_path():
    """Round1 finding I1: with the old f-string implementation, this exact
    payload closes the surrounding <dict> and injects a brand new
    EnvironmentVariables key — producing a VALID plist carrying
    ANTHROPIC_API_KEY: sk-ant-pwned. Verified independently against the old
    f-string template before writing this test. With plistlib.dumps the
    payload can only ever end up as a literal string value, so "the plist
    never holds the key" is an invariant rather than a property of the
    inputs the tests happen to use."""
    payload = (
        "x</string></dict><key>EnvironmentVariables</key>"
        "<dict><key>ANTHROPIC_API_KEY</key><string>sk-ant-pwned"
    )
    xml = launch_agent_plist(Path("/x/python"), Path("/tmp/z.log"), Path(payload))
    parsed = plistlib.loads(xml.encode())
    assert parsed["EnvironmentVariables"] == {"ZEUS_ENV_FILE": payload}


def test_env_file_is_loaded_into_the_environment(monkeypatch, tmp_path):
    from zeus.cli import _load_env_file

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    env = tmp_path / "env"
    env.write_text(
        "# a comment\n"
        "\n"
        "ANTHROPIC_API_KEY=sk-ant-test-value\n"
        'QUOTED="quoted-value"\n'
    )
    assert _load_env_file(env) == 2
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-test-value"
    assert os.environ["QUOTED"] == "quoted-value"


def test_env_file_does_not_override_the_real_environment(monkeypatch, tmp_path):
    """A key exported in the shell wins over the file, so running the daemon
    by hand behaves the same as launchd loading it."""
    from zeus.cli import _load_env_file

    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-the-shell")
    env = tmp_path / "env"
    env.write_text("ANTHROPIC_API_KEY=from-the-file\n")
    assert _load_env_file(env) == 0
    assert os.environ["ANTHROPIC_API_KEY"] == "from-the-shell"


def test_a_missing_env_file_is_not_an_error(tmp_path):
    from zeus.cli import _load_env_file

    assert _load_env_file(tmp_path / "nope") == 0


def test_env_file_loader_ignores_a_directory(tmp_path):
    """Round1 finding I2: exists() is True for a directory too, and the old
    code fell through to read_text(), raising IsADirectoryError out of
    cmd_run — a respawn loop under launchd's KeepAlive:true."""
    from zeus.cli import _load_env_file

    directory = tmp_path / "somedir"
    directory.mkdir()
    assert _load_env_file(directory) == 0


def test_env_file_loader_treats_an_empty_path_as_a_directory(tmp_path, monkeypatch):
    """ZEUS_ENV_FILE="" resolves to Path("") == Path("."), the process's cwd
    — a directory, not a file."""
    from zeus.cli import _load_env_file

    monkeypatch.chdir(tmp_path)
    assert _load_env_file(Path("")) == 0


@pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0,
    reason="root bypasses file permissions, so this can't be reproduced",
)
def test_env_file_loader_handles_unreadable_files_without_raising(tmp_path):
    """Round1 finding I2: a mode-000 file raised PermissionError out of
    cmd_run, uncaught."""
    from zeus.cli import _load_env_file

    env = tmp_path / "env"
    env.write_text("ANTHROPIC_API_KEY=sk-ant-test\n")
    env.chmod(0o000)
    try:
        assert _load_env_file(env) == 0
    finally:
        env.chmod(0o600)  # so tmp_path teardown can remove it


def test_env_file_loader_handles_non_utf8_content_without_raising(tmp_path):
    """Round1 finding I2: non-UTF-8 bytes raised UnicodeDecodeError out of
    cmd_run, uncaught."""
    from zeus.cli import _load_env_file

    env = tmp_path / "env"
    env.write_bytes(b"ANTHROPIC_API_KEY=\xff\xfe\n")
    assert _load_env_file(env) == 0


def test_env_file_loader_skips_keys_containing_whitespace(monkeypatch, tmp_path):
    """Round1 finding M1: `export ANTHROPIC_API_KEY=sk-...` used to set a
    variable literally named "export ANTHROPIC_API_KEY" and report
    loaded=1 while the real key stayed unset — the worst shape, since it
    looks like it worked."""
    from zeus.cli import _load_env_file

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    env = tmp_path / "env"
    env.write_text("export ANTHROPIC_API_KEY=sk-ant-test\n")
    assert _load_env_file(env) == 0
    assert "ANTHROPIC_API_KEY" not in os.environ
    assert "export ANTHROPIC_API_KEY" not in os.environ


def test_env_file_loader_strips_a_utf8_bom(monkeypatch, tmp_path):
    """Round1 finding M1: a UTF-8 BOM at the top of the file used to become
    part of the first key's name ("\ufeffANTHROPIC_API_KEY"), so the file's
    very first variable silently never set the real key."""
    from zeus.cli import _load_env_file

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    env = tmp_path / "env"
    env.write_bytes("\ufeffANTHROPIC_API_KEY=sk-ant-test\n".encode("utf-8"))
    assert _load_env_file(env) == 1
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-test"


def test_main_with_no_arguments_prints_usage(capsys):
    assert main([]) == 2
    assert "usage" in capsys.readouterr().out.lower()


def test_main_rejects_an_unknown_command(capsys):
    """RETURNS 2 — it does not raise. argparse calls parser.error() on an
    invalid subparser choice, which raises SystemExit(2); main() catches it
    so its declared `main(argv) -> int` contract holds on every path."""
    assert main(["frobnicate"]) == 2


def test_root_before_the_subcommand_survives_the_subparsers_default(
    monkeypatch, tmp_path
):
    """Round1 finding I5: default=argparse.SUPPRESS on each subparser's
    --root, NOT None, is load-bearing. A subparser re-applies its own
    default onto the shared namespace after parsing, so default=None would
    silently clobber a --root the top-level parser already captured when it
    appears BEFORE the subcommand — main(["--root", DIR, "doctor"]) would
    then quietly fall back to load_config()'s ~/.zeus default. Every other
    test in this file uses the after-subcommand ordering, so none of them
    would catch that regression."""
    import zeus.cli as cli

    seen_roots = []

    def fake_load_config(path=None, root=None):
        seen_roots.append(root)
        return Config(root=root if root is not None else tmp_path / "wrong-fallback")

    monkeypatch.setattr(cli, "load_config", fake_load_config)
    _mock_probes(monkeypatch, cli)

    assert main(["--root", str(tmp_path), "doctor"]) == 0
    assert seen_roots == [tmp_path]


def test_probe_python_accepts_311_and_313_not_only_312(monkeypatch):
    """Round1 finding I5: `== (3, 12)` and `>= (3, 11)` are indistinguishable
    when only ever exercised on the 3.12 interpreter running the tests.
    Driving _probe_python() directly with a mocked sys.version_info pins the
    actual floor regardless of which interpreter runs this suite."""
    import zeus.cli as cli

    monkeypatch.setattr(cli.sys, "version_info", (3, 11, 0, "final", 0))
    assert cli._probe_python() is True
    monkeypatch.setattr(cli.sys, "version_info", (3, 13, 2, "final", 0))
    assert cli._probe_python() is True
    monkeypatch.setattr(cli.sys, "version_info", (3, 10, 9, "final", 0))
    assert cli._probe_python() is False


def test_doctor_reports_and_returns_a_status(monkeypatch, tmp_path, capsys):
    import zeus.cli as cli

    _mock_probes(monkeypatch, cli)

    code = main(["doctor", "--root", str(tmp_path)])
    output = capsys.readouterr().out
    assert code == 0
    assert "say" in output and "ANTHROPIC_API_KEY" in output


def test_doctor_accepts_any_supported_python(monkeypatch, tmp_path, capsys):
    """Global Constraints say 3.11+, so doctor must not report a FAILURE on
    3.11 or 3.13 merely because it was developed on 3.12."""
    import zeus.cli as cli

    _mock_probes(monkeypatch, cli)
    assert sys.version_info[:2] >= (3, 11)
    assert main(["doctor", "--root", str(tmp_path)]) == 0


def test_doctor_fails_when_the_transcriber_is_missing(monkeypatch, tmp_path, capsys):
    """A broken transcriber is silent at runtime — LocalWhisper.transcribe()
    returns "" for every failure — so doctor is where it must surface."""
    import zeus.cli as cli

    _mock_probes(monkeypatch, cli, _probe_transcriber=lambda: False)

    assert main(["doctor", "--root", str(tmp_path)]) == 1
    assert "transcriber" in capsys.readouterr().out


def test_doctor_fails_when_the_api_key_is_missing(monkeypatch, tmp_path, capsys):
    import zeus.cli as cli

    _mock_probes(monkeypatch, cli, _probe_api_key=lambda: False)

    assert main(["doctor", "--root", str(tmp_path)]) == 1


def test_doctor_fails_when_sounddevice_is_missing(monkeypatch, tmp_path, capsys):
    """sounddevice is a hard import inside MicStream.start(); without it
    ZEUS cannot hear at all, so it belongs in the fatal set."""
    import zeus.cli as cli

    _mock_probes(monkeypatch, cli, _probe_sounddevice=lambda: False)

    assert main(["doctor", "--root", str(tmp_path)]) == 1
    assert "audio input" in capsys.readouterr().out


def test_doctor_fails_when_openwakeword_is_missing(monkeypatch, tmp_path, capsys):
    """openwakeword is a hard import when the wake detector loads its model;
    without it ZEUS never wakes, so it belongs in the fatal set."""
    import zeus.cli as cli

    _mock_probes(monkeypatch, cli, _probe_openwakeword=lambda: False)

    assert main(["doctor", "--root", str(tmp_path)]) == 1
    assert "wake word" in capsys.readouterr().out


def test_doctor_does_not_fail_when_afplay_is_missing(monkeypatch, tmp_path, capsys):
    """Round1 finding I4 (related): afplay is demoted to informational.
    Nothing in src/ invokes it — TTS goes through /usr/bin/say — so its
    absence alone must not flip doctor unhealthy."""
    import zeus.cli as cli

    _mock_probes(monkeypatch, cli, _probe_afplay=lambda: False)

    assert main(["doctor", "--root", str(tmp_path)]) == 0
    assert "afplay" in capsys.readouterr().out


def test_doctor_reads_the_key_from_the_env_file_not_just_os_environ(
    monkeypatch, tmp_path
):
    """Round1 finding I4: doctor used to read only os.environ. A key
    exported in ~/.zshrc (present in os.environ when you run `zeus doctor`
    by hand, absent when launchd starts the daemon with no shell) made
    doctor report all-OK right before install-agent shipped a LaunchAgent
    that would fail every brain call. Doctor must see what cmd_run will
    actually see: the env file at config.env_path."""
    import zeus.cli as cli

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    for probe in ("_probe_say", "_probe_afplay", "_probe_transcriber",
                  "_probe_python", "_probe_sounddevice", "_probe_openwakeword"):
        monkeypatch.setattr(cli, probe, lambda: True)

    root = tmp_path
    config = Config(root=root)
    config.env_path.parent.mkdir(parents=True, exist_ok=True)
    config.env_path.write_text("ANTHROPIC_API_KEY=sk-ant-from-file\n")

    try:
        assert main(["doctor", "--root", str(root)]) == 0
        assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-from-file"
    finally:
        os.environ.pop("ANTHROPIC_API_KEY", None)


def test_install_agent_names_the_venv_interpreter_not_the_resolved_symlink(
    monkeypatch, tmp_path
):
    """Round1 finding C1 (Critical): Path(sys.executable).resolve() follows
    the venv symlink OUT of the venv to the real interpreter behind it,
    which cannot `import zeus`. Verified independently: sys.executable
    imports zeus; its .resolve() raises ModuleNotFoundError. With
    KeepAlive:true launchd would respawn that failing job forever."""
    import zeus.cli as cli

    plist_path = tmp_path / "LaunchAgents" / "com.zeus.daemon.plist"
    monkeypatch.setattr(cli, "PLIST_PATH", plist_path)
    config = Config(root=tmp_path / "root")

    assert cli.cmd_install_agent(config) == 0

    parsed = plistlib.loads(plist_path.read_bytes())
    assert parsed["ProgramArguments"][0] == sys.executable


def test_install_agent_creates_the_log_directory(monkeypatch, tmp_path):
    """Round1 finding C1: launchd does not create StandardOutPath's parent
    directory itself. Without config.log_path.parent.mkdir(), the job fails
    to spawn the moment it tries to open the log."""
    import zeus.cli as cli

    plist_path = tmp_path / "LaunchAgents" / "com.zeus.daemon.plist"
    monkeypatch.setattr(cli, "PLIST_PATH", plist_path)
    config = Config(root=tmp_path / "root")

    assert not config.log_path.parent.exists()
    cli.cmd_install_agent(config)
    assert config.log_path.parent.exists()
