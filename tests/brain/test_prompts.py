from zeus.brain.prompts import (
    EVENING_OPENER,
    FOLDED_OPENER,
    MORNING_OPENER,
    SYSTEM_PROMPT,
    split_sentences,
)


def test_system_prompt_is_long_enough_to_cache():
    """Opus 5's prompt-cache minimum is 512 tokens; ~4 chars/token."""
    assert len(SYSTEM_PROMPT) > 2048


def test_system_prompt_states_the_exchange_ceiling():
    assert "three exchanges" in SYSTEM_PROMPT.lower()


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
    done, rest = split_sentences("It took 1.5 hours. Next")
    assert done == ["It took 1.5 hours."]
    assert rest == " Next"
