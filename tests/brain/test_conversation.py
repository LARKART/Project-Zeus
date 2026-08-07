from datetime import datetime, timezone

import pytest
from zoneinfo import ZoneInfo

from zeus.brain.conversation import Conversation
from zeus.brain.fake import FakeConversation
from zeus.clock import FakeClock
from zeus.config import BrainConfig
from zeus.memory.journal import Journal
from zeus.memory.store import Store

START = datetime(2026, 8, 5, 10, 0, tzinfo=timezone.utc)


# ---- stubs mirroring the anthropic Tool Runner shape --------------------
class StubEvent:
    def __init__(self, text):
        self.type = "content_block_delta"
        self.delta = type("D", (), {"type": "text_delta", "text": text})()


class StubStream:
    def __init__(self, chunks, stop_reason="end_turn", content=None):
        self._chunks = chunks
        self._stop_reason = stop_reason
        self._content = content if content is not None else []

    def __iter__(self):
        return iter(StubEvent(c) for c in self._chunks)

    def get_final_message(self):
        return type("M", (), {
            "stop_reason": self._stop_reason,
            "content": self._content,
            "stop_details": None,
        })()


class RaisingStream:
    """A stream that blows up as soon as it's iterated.

    Kept separate from StubStream (rather than adding a should-raise flag
    to it) so the normal stub stays simple. Exists to exercise the
    except-path history bookkeeping when an earlier round in the same turn
    already completed successfully.
    """

    def __iter__(self):
        raise RuntimeError("stream broke mid-turn")

    def get_final_message(self):  # pragma: no cover - never reached
        raise AssertionError("get_final_message() called on a stream that never iterated")


class StubClient:
    def __init__(self, streams):
        """streams[i] is turn i's response.

        A bare StubStream means that turn is a single round. A LIST of
        StubStreams means the Tool Runner yielded several rounds within that
        one turn — model call, tool execution, model call again — which is
        what the real runner does whenever a tool is used. An earlier draft
        returned the whole list on every call, conflating "later turns" with
        "more rounds in this turn", so a two-turn test had both streams
        consumed inside the first send().
        """
        self._streams = streams
        self.calls = []
        beta = type("B", (), {})()
        beta.messages = type("M", (), {"tool_runner": self._tool_runner})()
        self.beta = beta

    def _tool_runner(self, **kwargs):
        self.calls.append(kwargs)
        turn = self._streams[len(self.calls) - 1]
        return iter(turn if isinstance(turn, list) else [turn])


@pytest.fixture
def wiring(tmp_path):
    clock = FakeClock(START)
    store = Store(tmp_path / "zeus.db", clock)
    journal = Journal(tmp_path / "journal", clock, ZoneInfo("Africa/Lagos"))
    return store, journal, store.start_conversation("schedule")


def _conversation(client, wiring):
    store, journal, conv = wiring
    return Conversation(
        client=client, config=BrainConfig(), store=store, journal=journal,
        conversation_id=conv, system="SYSTEM", tools=[],
    )


def _assert_history_is_well_formed(messages):
    """Strict role alternation, and no unresolved tool_use anywhere.

    Written generically (no hardcoded role list) so it keeps meaning if the
    turn shape changes.
    """
    roles = [m["role"] for m in messages]
    assert all(roles[i] != roles[i + 1] for i in range(len(roles) - 1)), roles

    def _has_tool_use(message):
        content = message["content"]
        return isinstance(content, list) and any(
            isinstance(block, dict) and block.get("type") == "tool_use"
            for block in content
        )

    assert not any(_has_tool_use(m) for m in messages), messages


def test_send_yields_complete_sentences(wiring):
    client = StubClient([StubStream(["Morning.", " What's the ", "one thing?"])])
    assert list(_conversation(client, wiring).send("[morning]")) == [
        "Morning.", "What's the one thing?",
    ]


def test_trailing_fragment_is_flushed_at_the_end(wiring):
    client = StubClient([StubStream(["No punctuation here"])])
    assert list(_conversation(client, wiring).send("hi")) == ["No punctuation here"]


def test_messages_are_persisted(wiring):
    store, _, conv = wiring
    client = StubClient([StubStream(["Morning."])])
    list(_conversation(client, wiring).send("[morning]"))

    roles = [m.role for m in store.messages(conv)]
    assert roles == ["user", "assistant"]
    assert store.messages(conv)[1].content == "Morning."


def test_request_uses_the_required_model_settings(wiring):
    client = StubClient([StubStream(["ok."])])
    list(_conversation(client, wiring).send("hi"))
    kwargs = client.calls[0]

    assert kwargs["model"] == "claude-opus-5"
    assert kwargs["thinking"] == {"type": "adaptive"}     # never "disabled"
    assert kwargs["output_config"]["effort"] == "low"
    assert kwargs["stream"] is True
    assert kwargs["system"][0]["cache_control"] == {"type": "ephemeral"}


def test_refusal_is_handled_before_reading_content(wiring):
    client = StubClient([StubStream([], stop_reason="refusal")])
    spoken = list(_conversation(client, wiring).send("hi"))
    assert len(spoken) == 1
    assert "can't help" in spoken[0].lower()


def test_api_failure_yields_a_spoken_fallback(wiring):
    class Exploding(StubClient):
        def _tool_runner(self, **kwargs):
            raise RuntimeError("connection reset")

    spoken = list(_conversation(Exploding([]), wiring).send("hi"))
    assert len(spoken) == 1
    assert "reach" in spoken[0].lower()


def test_history_accumulates_across_turns(wiring):
    client = StubClient([StubStream(["One."]), StubStream(["Two."])])
    conversation = _conversation(client, wiring)
    list(conversation.send("first"))
    list(conversation.send("second"))

    second_call = client.calls[1]["messages"]
    assert [m["role"] for m in second_call] == ["user", "assistant", "user"]


def test_a_single_turn_can_span_several_tool_runner_rounds(wiring):
    """The real runner yields one stream per round whenever a tool is used;
    send() must surface sentences from every round, not just the first."""
    client = StubClient([[StubStream(["Saving that."]), StubStream(["Done."])]])
    assert list(_conversation(client, wiring).send("[morning]")) == [
        "Saving that.", "Done.",
    ]


def test_a_tool_using_turn_leaves_history_alternating(wiring):
    """A turn where the model calls a tool is >1 round: a tool_use round,
    then a text round. Appending an assistant message per round (instead of
    once per turn) leaves two consecutive assistant messages in history with
    no tool_result between them, which the real API rejects with a 400 on
    the NEXT send() — degrading a successful tool call into "I can't reach
    my brain right now" right after it worked."""
    tool_use_round = StubStream(
        [], stop_reason="tool_use",
        content=[{
            "type": "tool_use", "id": "toolu_1",
            "name": "record_outcome", "input": {},
        }],
    )
    text_round = StubStream(["Got it."])
    client = StubClient([
        [tool_use_round, text_round],
        StubStream(["Anything else?"]),
    ])
    conversation = _conversation(client, wiring)
    list(conversation.send("first"))
    list(conversation.send("second"))

    _assert_history_is_well_formed(client.calls[1]["messages"])


def test_a_refused_first_round_still_leaves_history_alternating(wiring):
    """A turn that is refused on its very first round contributes NO
    final_content. Because send() always appends a user message up front,
    a turn that contributes no assistant message back leaves two adjacent
    user messages, which the API rejects just as surely as a dangling
    tool_use."""
    client = StubClient([
        StubStream([], stop_reason="refusal"),
        StubStream(["Sure."]),
    ])
    conversation = _conversation(client, wiring)
    list(conversation.send("first"))
    list(conversation.send("second"))

    _assert_history_is_well_formed(client.calls[1]["messages"])


def test_a_refusal_after_a_tool_round_drops_the_dangling_tool_use(wiring):
    """Role alternation alone can look fine here — the defect is in the
    CONTENT: without clearing final_content on the refusal path, the
    dangling tool_use from the completed first round rides along into the
    next send() wearing a plausible ['user', 'assistant', 'user'] shape."""
    tool_round = StubStream(
        [], stop_reason="tool_use",
        content=[{
            "type": "tool_use", "id": "toolu_2",
            "name": "record_outcome", "input": {},
        }],
    )
    refusal_round = StubStream([], stop_reason="refusal")
    client = StubClient([
        [tool_round, refusal_round],
        StubStream(["Sure."]),
    ])
    conversation = _conversation(client, wiring)
    list(conversation.send("first"))
    list(conversation.send("second"))

    _assert_history_is_well_formed(client.calls[1]["messages"])


def test_an_exception_after_a_tool_round_drops_the_dangling_tool_use(wiring):
    """Same mechanism as the refusal case, via the except path instead: an
    earlier round's tool_use must not survive an exception in a later
    round of the same turn."""
    tool_round = StubStream(
        [], stop_reason="tool_use",
        content=[{
            "type": "tool_use", "id": "toolu_3",
            "name": "record_outcome", "input": {},
        }],
    )
    client = StubClient([
        [tool_round, RaisingStream()],
        StubStream(["Sure."]),
    ])
    conversation = _conversation(client, wiring)
    list(conversation.send("first"))
    list(conversation.send("second"))

    _assert_history_is_well_formed(client.calls[1]["messages"])


# ---- B2: the tool loop is bounded ---------------------------------------


def _tool_round(tool_id):
    """A round that asks for a tool and never produces text."""
    return StubStream(
        [], stop_reason="tool_use",
        content=[{
            "type": "tool_use", "id": tool_id,
            "name": "record_outcome", "input": {"status": "done"},
        }],
    )


def test_the_tool_runner_is_given_an_iteration_bound(wiring):
    """Without max_iterations the SDK's _should_stop() short-circuits on
    `self._max_iterations is not None`, so request -> tool -> request is
    bounded only by the model deciding to stop. record_outcome returning
    "There is no goal recorded for today" is a SUCCESSFUL result the model
    may simply retry: the check-in never returns, speak() never finishes,
    the scheduler thread blocks, and the API bill runs on unattended."""
    from zeus.brain.conversation import MAX_TOOL_ROUNDS

    client = StubClient([StubStream(["ok."])])
    list(_conversation(client, wiring).send("hi"))

    assert client.calls[0]["max_iterations"] == MAX_TOOL_ROUNDS
    assert MAX_TOOL_ROUNDS > 2, (
        "a normal tool-using turn is two rounds (tool_use, then text); a "
        "bound at or below that would truncate healthy conversations"
    )


def test_the_installed_sdk_accepts_max_iterations():
    """The kwarg is only useful if the real client takes it -- a stub that
    records whatever it is handed cannot tell a correct parameter name from
    a typo. Reads the signature; no network, no key in use."""
    import inspect

    import anthropic

    client = anthropic.Anthropic(api_key="test-key-never-sent")
    parameters = inspect.signature(client.beta.messages.tool_runner).parameters
    assert "max_iterations" in parameters


def test_running_out_of_tool_rounds_is_spoken_and_logged(wiring, caplog):
    """The runner stops SILENTLY when it hits max_iterations -- StopIteration,
    no exception, no warning. Left alone that is a check-in that just goes
    quiet, which §10 rules is worse than one that says it is broken."""
    import logging

    from zeus.brain.conversation import STUCK_LINE

    client = StubClient([[_tool_round(f"toolu_{i}") for i in range(6)]])
    with caplog.at_level(logging.ERROR, logger="zeus.brain.conversation"):
        spoken = list(_conversation(client, wiring).send("hi"))

    assert spoken == [STUCK_LINE]
    assert any("tool rounds" in r.message for r in caplog.records)


def test_running_out_of_tool_rounds_leaves_history_valid(wiring):
    """The trap the bound introduces if it is added carelessly: the last
    round the runner produced holds a tool_use whose tool_result will never
    be generated, and replaying it on the next send() is a 400."""
    client = StubClient([
        [_tool_round(f"toolu_{i}") for i in range(6)],
        StubStream(["Sure."]),
    ])
    conversation = _conversation(client, wiring)
    list(conversation.send("first"))
    list(conversation.send("second"))

    _assert_history_is_well_formed(client.calls[1]["messages"])


# ---- B1: a truncated reply must not sound complete ----------------------


def test_a_truncated_reply_says_so_instead_of_sounding_finished(wiring, caplog):
    """MAX_TOKENS is the ceiling for adaptive thinking PLUS text on Opus 5,
    so a long turn can exhaust it. Only `refusal` was branched on, so every
    other stop reason fell through and the buffered fragment was spoken as
    a finished sentence with nothing logged at WARNING or above."""
    import logging

    from zeus.brain.conversation import TRUNCATED_LINE

    client = StubClient([
        StubStream(["You said you'd finish the auth flow, and you got about "
                    "three quarters of the way thro"], stop_reason="max_tokens")
    ])
    with caplog.at_level(logging.WARNING, logger="zeus.brain.conversation"):
        spoken = list(_conversation(client, wiring).send("hi"))

    assert spoken[-1] == TRUNCATED_LINE, (
        "the truncated fragment was spoken as if the answer were complete"
    )
    assert "three quarters of the way thro" in spoken[0], (
        "what the model DID say should still be heard"
    )
    assert any(record.levelno >= logging.WARNING for record in caplog.records)


def test_a_truncated_tool_call_is_neither_run_nor_replayed(wiring):
    """The quieter half. The SDK accumulates tool input with
    partial_mode=True, so a cut-off {"status":"done","notes":"took about
    three ho would call record_outcome(status="done") with notes silently
    dropped and ok=True in the action log. The runner only executes those
    tools when the iterator is advanced again -- so the fix is to stop
    advancing it, which also keeps the dangling tool_use out of history."""
    truncated_tool_round = StubStream(
        [], stop_reason="max_tokens",
        content=[{
            "type": "tool_use", "id": "toolu_cut",
            "name": "record_outcome", "input": {"status": "done"},
        }],
    )
    client = StubClient([
        [truncated_tool_round, StubStream(["never reached."])],
        StubStream(["Sure."]),
    ])
    conversation = _conversation(client, wiring)
    spoken = list(conversation.send("first"))
    list(conversation.send("second"))

    assert "never reached." not in spoken, (
        "the runner was advanced past the truncated tool call"
    )
    _assert_history_is_well_formed(client.calls[1]["messages"])


# ---- B5, B6: round bookkeeping ------------------------------------------


def test_sentences_are_not_welded_across_rounds(wiring):
    """B5: the buffer was never reset between rounds, so a round ending
    without punctuation and the next round's opening text were emitted as
    one sentence -- two separate model messages welded into one line of
    speech."""
    client = StubClient([[StubStream(["Let me save that"]),
                          StubStream(["Done."])]])
    assert list(_conversation(client, wiring).send("[morning]")) == [
        "Let me save that", "Done.",
    ]


def test_an_empty_content_list_is_not_appended_to_history(wiring):
    """B6: the guard was `final_content is not None`, so an empty list
    passed it and appended {"role": "assistant", "content": []}, which the
    API rejects. Empty means the model contributed nothing, which is
    exactly when the spoken fallback should take over."""
    client = StubClient([
        StubStream(["Morning."], content=[]),
        StubStream(["Next."]),
    ])
    conversation = _conversation(client, wiring)
    list(conversation.send("first"))
    list(conversation.send("second"))

    history = client.calls[1]["messages"]
    assert all(message["content"] for message in history), history
    assert history[1] == {"role": "assistant", "content": "Morning."}


def test_fake_conversation_replays_a_script():
    fake = FakeConversation({"[morning]": ["Morning.", "What's the goal?"]})
    assert list(fake.send("[morning]")) == ["Morning.", "What's the goal?"]
    assert fake.sent == ["[morning]"]


def test_fake_conversation_falls_back_for_unscripted_input():
    fake = FakeConversation({})
    assert list(fake.send("anything")) == ["Got it."]


def test_fake_conversation_invokes_real_tools(wiring):
    """This is what lets the end-to-end test prove the real tool path."""
    from zeus.brain.tools import build_tool_callables

    store, journal, conv = wiring
    tools = build_tool_callables(store, journal, conv, "2026-08-05")
    fake = FakeConversation(
        tools=tools,
        tool_calls=[[("save_goal", {"text": "Finish the auth flow"})]],
    )

    list(fake.send("[morning]"))

    assert store.get_goal("2026-08-05").text == "Finish the auth flow"
    assert store.recent_actions()[0].tool == "save_goal"
    assert fake.invoked == [("save_goal", {"text": "Finish the auth flow"})]


def test_fake_conversation_invokes_tools_per_turn(wiring):
    from zeus.brain.tools import build_tool_callables

    store, journal, conv = wiring
    tools = build_tool_callables(store, journal, conv, "2026-08-05")
    fake = FakeConversation(
        tools=tools,
        tool_calls=[
            [("save_goal", {"text": "Ship it"})],
            [],
            [("record_outcome", {"status": "done"})],
        ],
    )

    list(fake.send("turn one"))
    list(fake.send("turn two"))
    list(fake.send("turn three"))

    assert [name for name, _ in fake.invoked] == ["save_goal", "record_outcome"]
    assert store.get_goal("2026-08-05").status == "done"


def test_fake_conversation_rejects_an_unknown_tool():
    fake = FakeConversation(tools={}, tool_calls=[[("nope", {})]])
    with pytest.raises(KeyError, match="nope"):
        list(fake.send("x"))
