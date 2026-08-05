import numpy as np

from zeus.audio.endpointer import Endpointer, capture_utterance, rms
from zeus.audio.mic import FRAME_SAMPLES
from zeus.config import AudioConfig

SILENCE = np.zeros(FRAME_SAMPLES, dtype=np.int16).tobytes()
SPEECH = (np.ones(FRAME_SAMPLES, dtype=np.int16) * 8000).tobytes()


def test_rms_of_silence_is_zero():
    assert rms(SILENCE) == 0.0


def test_rms_of_loud_audio_is_high():
    assert rms(SPEECH) > 0.2


def test_rms_of_empty_frame_is_zero():
    assert rms(b"") == 0.0


def test_silence_before_speech_never_ends_the_utterance():
    endpointer = Endpointer(AudioConfig())
    for _ in range(100):
        assert endpointer.feed(SILENCE) is False
    assert endpointer.saw_speech is False


def test_speech_then_sustained_silence_ends_the_utterance():
    # silence_timeout defaults to 1.5 s == ~19 frames of 80 ms
    endpointer = Endpointer(AudioConfig())
    endpointer.feed(SPEECH)
    assert endpointer.saw_speech is True

    results = [endpointer.feed(SILENCE) for _ in range(19)]
    assert results[-1] is True
    assert True not in results[:-1]


def test_speech_resets_the_silence_run():
    endpointer = Endpointer(AudioConfig())
    endpointer.feed(SPEECH)
    for _ in range(10):
        endpointer.feed(SILENCE)
    endpointer.feed(SPEECH)          # resets the counter
    results = [endpointer.feed(SILENCE) for _ in range(10)]
    assert True not in results        # not enough silence yet


def test_reset_clears_state():
    endpointer = Endpointer(AudioConfig())
    endpointer.feed(SPEECH)
    endpointer.reset()
    assert endpointer.saw_speech is False


def test_capture_prepends_pre_roll():
    endpointer = Endpointer(AudioConfig())
    frames = iter([SPEECH] + [SILENCE] * 19)
    audio = capture_utterance(frames, endpointer, pre_roll=SPEECH, listen_timeout_frames=100)
    assert audio.startswith(SPEECH + SPEECH)


def test_capture_returns_empty_when_only_silence_is_heard():
    endpointer = Endpointer(AudioConfig())
    frames = iter([SILENCE] * 50)
    audio = capture_utterance(frames, endpointer, pre_roll=b"", listen_timeout_frames=50)
    assert audio == b""


def test_capture_stops_at_the_listen_timeout():
    endpointer = Endpointer(AudioConfig())
    frames = iter([SPEECH] * 1000)
    audio = capture_utterance(frames, endpointer, pre_roll=b"", listen_timeout_frames=10)
    assert len(audio) == 10 * FRAME_SAMPLES * 2


def test_capture_returns_empty_even_when_pre_roll_holds_audio():
    endpointer = Endpointer(AudioConfig())
    audio = capture_utterance(
        [SILENCE, SILENCE, SILENCE],
        endpointer,
        pre_roll=SPEECH * 2,          # room noise captured before the trigger
        listen_timeout_frames=50,
    )
    assert audio == b""               # NOT the pre-roll bytes


def test_threshold_comparison_is_inclusive():
    frame = (np.ones(FRAME_SAMPLES, dtype=np.int16) * 655).tobytes()
    exact = rms(frame)

    at_threshold = Endpointer(AudioConfig(), threshold=exact)
    at_threshold.feed(frame)
    assert at_threshold.saw_speech is True     # >= : a frame AT the threshold is speech

    just_above = Endpointer(AudioConfig(), threshold=exact + 1e-9)
    just_above.feed(frame)
    assert just_above.saw_speech is False      # proves the assertion above is not vacuous


def test_reset_clears_the_silence_run_not_just_saw_speech():
    endpointer = Endpointer(AudioConfig())
    needed = endpointer._silence_frames_needed

    endpointer.feed(SPEECH)
    for _ in range(needed - 1):
        assert endpointer.feed(SILENCE) is False   # one frame short of firing
    endpointer.reset()
    assert endpointer._silence_run == 0  # pin the state directly: the very next call
    # below is feed(SPEECH), which itself zeroes _silence_run and would mask a
    # broken reset() if we relied on behavioural assertions alone.

    # A second utterance must need the FULL silence run again, not one frame.
    endpointer.feed(SPEECH)
    for i in range(needed - 1):
        assert endpointer.feed(SILENCE) is False, f"ended early on silence frame {i + 1}"
    assert endpointer.feed(SILENCE) is True


def test_pre_roll_does_not_consume_the_listen_budget():
    endpointer = Endpointer(AudioConfig())
    pre = SPEECH * 5
    audio = capture_utterance(
        [SPEECH] * 1000, endpointer, pre_roll=pre, listen_timeout_frames=10
    )
    assert len(audio) == len(pre) + 10 * FRAME_SAMPLES * 2
