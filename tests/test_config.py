from datetime import timedelta
from pathlib import Path

import pytest

from zeus.config import load_config, parse_duration


@pytest.mark.parametrize(
    "text,expected",
    [
        ("30s", timedelta(seconds=30)),
        ("1.5s", timedelta(seconds=1.5)),
        ("20m", timedelta(minutes=20)),
        ("2h", timedelta(hours=2)),
    ],
)
def test_parse_duration(text, expected):
    assert parse_duration(text) == expected


def test_parse_duration_rejects_garbage():
    with pytest.raises(ValueError):
        parse_duration("soon")


def test_defaults_when_file_absent(tmp_path):
    cfg = load_config(path=tmp_path / "nope.toml", root=tmp_path)
    assert cfg.schedule.morning == "11:00"
    assert cfg.schedule.evening == "21:00"
    assert cfg.schedule.defer_retry_after == timedelta(minutes=20)
    assert cfg.schedule.max_defer_retries == 3
    assert cfg.schedule.no_answer_retry_after == timedelta(minutes=30)
    assert cfg.schedule.max_no_answer_retries == 1
    assert cfg.brain.model == "claude-opus-5"
    assert cfg.audio.sample_rate == 16000
    assert cfg.root == tmp_path


def test_file_overrides_defaults(tmp_path):
    (tmp_path / "config.toml").write_text(
        '[schedule]\nmorning = "09:30"\n[tts]\nvoice = "Samantha"\n'
    )
    cfg = load_config(path=tmp_path / "config.toml", root=tmp_path)
    assert cfg.schedule.morning == "09:30"
    assert cfg.tts.voice == "Samantha"
    assert cfg.schedule.evening == "21:00"  # untouched default survives
