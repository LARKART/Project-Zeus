import os
import plistlib
import sys
from pathlib import Path

from zeus.cli import launch_agent_plist, main


ENV = Path("/Users/x/.zeus/env")


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


def test_main_with_no_arguments_prints_usage(capsys):
    assert main([]) == 2
    assert "usage" in capsys.readouterr().out.lower()


def test_main_rejects_an_unknown_command(capsys):
    """RETURNS 2 — it does not raise. argparse calls parser.error() on an
    invalid subparser choice, which raises SystemExit(2); main() catches it
    so its declared `main(argv) -> int` contract holds on every path."""
    assert main(["frobnicate"]) == 2


def test_doctor_reports_and_returns_a_status(monkeypatch, tmp_path, capsys):
    import zeus.cli as cli

    monkeypatch.setattr(cli, "_probe_say", lambda: True)
    monkeypatch.setattr(cli, "_probe_afplay", lambda: True)
    monkeypatch.setattr(cli, "_probe_api_key", lambda: True)
    monkeypatch.setattr(cli, "_probe_transcriber", lambda: True)

    code = main(["doctor", "--root", str(tmp_path)])
    output = capsys.readouterr().out
    assert code == 0
    assert "say" in output and "ANTHROPIC_API_KEY" in output


def test_doctor_accepts_any_supported_python(monkeypatch, tmp_path, capsys):
    """Global Constraints say 3.11+, so doctor must not report a FAILURE on
    3.11 or 3.13 merely because it was developed on 3.12."""
    import zeus.cli as cli

    for probe in ("_probe_say", "_probe_afplay", "_probe_api_key",
                  "_probe_transcriber"):
        monkeypatch.setattr(cli, probe, lambda: True)
    assert sys.version_info[:2] >= (3, 11)
    assert main(["doctor", "--root", str(tmp_path)]) == 0


def test_doctor_fails_when_the_transcriber_is_missing(monkeypatch, tmp_path, capsys):
    """A broken transcriber is silent at runtime — LocalWhisper.transcribe()
    returns "" for every failure — so doctor is where it must surface."""
    import zeus.cli as cli

    monkeypatch.setattr(cli, "_probe_say", lambda: True)
    monkeypatch.setattr(cli, "_probe_afplay", lambda: True)
    monkeypatch.setattr(cli, "_probe_api_key", lambda: True)
    monkeypatch.setattr(cli, "_probe_transcriber", lambda: False)

    assert main(["doctor", "--root", str(tmp_path)]) == 1
    assert "transcriber" in capsys.readouterr().out


def test_doctor_fails_when_the_api_key_is_missing(monkeypatch, tmp_path, capsys):
    import zeus.cli as cli

    monkeypatch.setattr(cli, "_probe_say", lambda: True)
    monkeypatch.setattr(cli, "_probe_afplay", lambda: True)
    monkeypatch.setattr(cli, "_probe_transcriber", lambda: True)
    monkeypatch.setattr(cli, "_probe_api_key", lambda: False)

    assert main(["doctor", "--root", str(tmp_path)]) == 1
