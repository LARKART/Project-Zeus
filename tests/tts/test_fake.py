from zeus.tts.fake import FakeSpeaker


def test_records_everything_said():
    speaker = FakeSpeaker()
    speaker.say("Morning.")
    speaker.say("What's the goal?")
    assert speaker.said == ["Morning.", "What's the goal?"]


def test_records_stop_calls():
    speaker = FakeSpeaker()
    speaker.stop()
    assert speaker.stopped == 1
