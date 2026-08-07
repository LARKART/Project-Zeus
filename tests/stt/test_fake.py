from zeus.stt.fake import FakeTranscriber


def test_returns_scripted_strings_in_order():
    stt = FakeTranscriber(["Finish the auth flow", "Yes, mostly"])
    assert stt.transcribe(b"\x00" * 100, 16000) == "Finish the auth flow"
    assert stt.transcribe(b"\x00" * 200, 16000) == "Yes, mostly"


def test_returns_empty_once_exhausted():
    stt = FakeTranscriber(["only one"])
    stt.transcribe(b"", 16000)
    assert stt.transcribe(b"", 16000) == ""


def test_records_input_sizes():
    stt = FakeTranscriber(["a", "b"])
    stt.transcribe(b"\x00" * 320, 16000)
    stt.transcribe(b"\x00" * 640, 16000)
    assert stt.calls == [320, 640]
