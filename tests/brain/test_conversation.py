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

    submitted = client.calls[1]["messages"]
    roles = [m["role"] for m in submitted]
    assert all(roles[i] != roles[i + 1] for i in range(len(roles) - 1)), roles

    def _has_tool_use(message):
        content = message["content"]
        return isinstance(content, list) and any(
            isinstance(block, dict) and block.get("type") == "tool_use"
            for block in content
        )

    assert not any(_has_tool_use(m) for m in submitted)


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
