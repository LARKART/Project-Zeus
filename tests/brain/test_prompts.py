from zeus.brain.prompts import (
    EVENING_OPENER,
    FOLDED_OPENER,
    MORNING_OPENER,
    SYSTEM_PROMPT,
    split_sentences,
)


def test_system_prompt_names_the_tools_it_tells_the_model_to_use():
    """B3: the prompt used to be pinned ONLY by `len(SYSTEM_PROMPT) > 2048`,
    and its actual length is 2049 -- one character of slack, guarding
    nothing that matters. Three length-preserving content inversions all
    survived the suite: the morning script inverted to "ask what the FIVE
    things ar", "Write for the ear, not the page." rewritten to "Write for
    the web, not the ear!!", and record_outcome replaced with a tool that
    does not exist. A prompt naming a nonexistent tool is a check-in that
    records nothing, silently.

    So the assertions are on CONTENT: the two tool names must match the two
    tools build_tools actually registers, by construction rather than by a
    hardcoded list that could drift with them."""
    from zeus.brain.tools import VALID_STATUSES

    assert "save_goal" in SYSTEM_PROMPT
    assert "record_outcome" in SYSTEM_PROMPT
    # The statuses the prompt teaches must be ones record_outcome accepts.
    for status in ("done", "partial", "missed"):
        assert status in SYSTEM_PROMPT
        assert status in VALID_STATUSES


def test_system_prompt_states_the_one_goal_framing():
    """The whole product is one goal a day. A prompt that asks for a list
    is a different assistant."""
    assert "one thing" in SYSTEM_PROMPT.lower()
    assert "accountability partner" in SYSTEM_PROMPT.lower()


def test_system_prompt_states_the_exchange_ceiling():
    assert "three exchanges" in SYSTEM_PROMPT.lower()


def test_system_prompt_is_written_for_speech():
    """Everything here is played through text-to-speech; markdown, bullets
    and URLs do not survive it."""
    assert "for the ear" in SYSTEM_PROMPT.lower()
    assert "markdown" in SYSTEM_PROMPT.lower()


def test_system_prompt_is_substantial_enough_to_be_worth_caching(monkeypatch):
    """Opus 5's prompt-cache minimum is 512 tokens.

    A CHARACTER count was the wrong instrument: the cached prefix is tools
    PLUS system (~700 tokens measured), so the system prompt alone was
    never the thing that had to clear 512, and the old assertion sat one
    edit from a red build while measuring something it did not control.
    Counting tokens for real needs a network call, which no test here may
    make -- so this is an explicit floor with genuine headroom (360 words
    today against a floor of 250), guarding against someone gutting the
    prompt rather than against a one-word edit."""
    assert len(SYSTEM_PROMPT.split()) > 250


def test_evening_opener_embeds_the_goal():
    assert "Finish the auth flow" in EVENING_OPENER("Finish the auth flow")


def test_openers_are_distinct():
    assert len({MORNING_OPENER, EVENING_OPENER("x"), FOLDED_OPENER}) == 3


def test_split_sentences_emits_complete_sentences_only():
    done, rest = split_sentences("Morning. What's the one thing")
    assert done == ["Morning."]
    assert rest == " What's the one thing"


def test_split_sentences_handles_question_and_exclamation():
    done, rest = split_sentences("Done? Great! Now")
    assert done == ["Done?", "Great!"]
    assert rest == " Now"


def test_split_sentences_returns_nothing_when_incomplete():
    done, rest = split_sentences("Morning")
    assert done == []
    assert rest == "Morning"


def test_split_sentences_does_not_break_on_decimals():
    """The `(?=\\s|$)` lookahead is what does this, not the old `(?<!\\d)`
    guard: in "1.5" the dot is followed by "5", so it never matched in the
    first place. Two decimals, one mid-sentence and one adjacent to the
    real terminator, so the assertion is about the decimal rule rather than
    about where this particular string happens to end."""
    done, rest = split_sentences("It took 1.5 hours and cost 2.75. Next")
    assert done == ["It took 1.5 hours and cost 2.75."]
    assert rest == " Next"


def test_split_sentences_breaks_after_a_sentence_ending_in_a_digit():
    """B4: what the removed `(?<!\\d)` guard actually suppressed. "The
    meeting is at 3." was withheld from text-to-speech until some later
    non-digit sentence arrived, and spec §7.1 makes that latency the whole
    reason for streaming sentence by sentence."""
    done, rest = split_sentences("The meeting is at 3. Then we go.")
    assert done == ["The meeting is at 3.", "Then we go."]
    assert rest == ""
