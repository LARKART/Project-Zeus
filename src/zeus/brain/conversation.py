"""Claude conversation over the Anthropic Tool Runner. See spec §7.1.

Streams the reply and yields complete sentences so text-to-speech can begin
before the model has finished — the difference between a responsive voice
interface and a laggy one.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Iterator

from zeus.brain.prompts import split_sentences
from zeus.config import BrainConfig
from zeus.memory.journal import Journal
from zeus.memory.store import Store

log = logging.getLogger(__name__)

MAX_TOKENS = 1024

# A BOUND ON THE request -> tool -> request LOOP. Without one the SDK's
# _should_stop() short-circuits on `self._max_iterations is not None`, so the
# loop ends only when the model decides to stop — and a successful-but-useless
# tool result is something it may simply retry. record_outcome returning
# "There is no goal recorded for today" is exactly that shape: the call
# succeeded, nothing changed, and trying again is a reasonable-looking move.
# The check-in would then never return, speak() would never finish, and the
# scheduler thread would block — a hang plus unbounded API spend on a daemon
# nobody is watching.
#
# 6, not 3: this counts ROUNDS WITHIN ONE TURN, not the three exchanges spec
# §7 allows (each exchange is its own send() with its own runner). A normal
# turn is one round (text) or two (tool_use, then text). Six leaves room to
# call both Slice 1 tools in sequence and recover from one failure, while
# capping the pathological case at six API calls instead of infinity.
MAX_TOOL_ROUNDS = 6

REFUSAL_LINE = "Sorry, I can't help with that one."
ERROR_LINE = "I can't reach my brain right now. I'll try again later."
# Spec §10's governing rule is "fail loudly, never pretend". A reply cut off
# at MAX_TOKENS must not be spoken as though it were complete.
TRUNCATED_LINE = "Sorry, I ran out of room there. That answer was cut off."
STUCK_LINE = "I got stuck going in circles there. Let's leave it for now."


class Conversation:
    def __init__(
        self,
        client: Any,
        config: BrainConfig,
        store: Store,
        journal: Journal,
        conversation_id: int,
        system: str,
        tools: list[Callable],
        effort: str | None = None,
    ) -> None:
        self._client = client
        self._config = config
        self._store = store
        self._journal = journal
        self._conversation_id = conversation_id
        self._system = system
        self._tools = tools
        self._effort = effort or config.effort_checkin
        self._messages: list[dict[str, Any]] = []

    def send(self, text: str) -> Iterator[str]:
        """Send a turn and yield the reply one complete sentence at a time."""
        self._messages.append({"role": "user", "content": text})
        self._store.add_message(self._conversation_id, "user", text)

        buffer = ""
        spoken: list[str] = []
        final_content = None
        stop_reason = None
        try:
            runner = self._client.beta.messages.tool_runner(
                model=self._config.model,
                max_tokens=MAX_TOKENS,
                max_iterations=MAX_TOOL_ROUNDS,
                # Adaptive thinking stays on. Disabling it on Opus 5 risks
                # tool calls being emitted as visible text that never run.
                thinking={"type": "adaptive"},
                output_config={"effort": self._effort},
                system=[{
                    "type": "text",
                    "text": self._system,
                    "cache_control": {"type": "ephemeral"},
                }],
                tools=self._tools,
                messages=list(self._messages),
                stream=True,
            )

            for stream in runner:
                for event in stream:
                    if getattr(event, "type", None) != "content_block_delta":
                        continue
                    delta = getattr(event, "delta", None)
                    if getattr(delta, "type", None) != "text_delta":
                        continue
                    buffer += delta.text
                    sentences, buffer = split_sentences(buffer)
                    for sentence in sentences:
                        spoken.append(sentence)
                        yield sentence

                final = stream.get_final_message()
                # Check stop_reason BEFORE touching content: on a refusal
                # `content` is empty and indexing it would crash (spec §10).
                stop_reason = getattr(final, "stop_reason", None)
                if stop_reason == "refusal":
                    log.warning("model refused the request")
                    spoken.append(REFUSAL_LINE)
                    yield REFUSAL_LINE
                    buffer = ""
                    # Discard any earlier round's content. If a tool round
                    # already completed, that content holds a tool_use block
                    # whose tool_result will never arrive, and replaying it
                    # next turn is a 400.
                    final_content = None
                    break
                if stop_reason == "max_tokens":
                    # THE REPLY WAS CUT OFF. Only `refusal` used to be
                    # branched on, so every other stop reason fell through
                    # and the buffered remainder was spoken as if it were a
                    # finished sentence, with nothing logged at WARNING or
                    # above. MAX_TOKENS is the ceiling for adaptive thinking
                    # PLUS text on Opus 5, so a long turn really can exhaust
                    # it.
                    #
                    # Breaking here also stops the SECOND, quieter
                    # consequence: the SDK accumulates tool input with
                    # partial_mode=True, so a truncated
                    # {"status":"done","notes":"took about three ho would
                    # call record_outcome(status="done") with notes silently
                    # dropped and ok=True in the action log. The runner only
                    # executes those tools when the iterator is advanced
                    # again, and it never is. (Truncation that drops a
                    # REQUIRED argument already failed cleanly.)
                    #
                    # What WAS said is still spoken — the user should hear
                    # the real content — and then the line that says it was
                    # cut off. final_content is dropped for the refusal
                    # path's reason: a truncated tool_use would never get a
                    # tool_result, and replaying it next turn is a 400.
                    log.warning(
                        "reply hit the %d-token ceiling and was cut off "
                        "mid-answer", MAX_TOKENS,
                    )
                    remainder = buffer.strip()
                    if remainder:
                        spoken.append(remainder)
                        yield remainder
                    buffer = ""
                    spoken.append(TRUNCATED_LINE)
                    yield TRUNCATED_LINE
                    final_content = None
                    break
                # Keep only the LAST round's content. A turn that uses a
                # tool is >1 round (tool_use, then text); appending per
                # round left two consecutive assistant messages with an
                # unresolved tool_use and no tool_result, which the API
                # rejects with a 400 on the NEXT send(). The runner already
                # handles tool_use/tool_result correctly inside the turn —
                # it copies our list rather than mutating it (anthropic
                # 0.120.2), so what we keep here is only what the next turn
                # replays, and one closing assistant message per turn keeps
                # role alternation valid by construction.
                final_content = final.content

                # FLUSH PER ROUND, not once at the end of the turn. The
                # buffer used to carry across rounds, so a round ending
                # "Let me save that" (no terminator) and a next round
                # opening "Done." were emitted as the single sentence
                # "Let me save thatDone." — two separate model messages
                # welded into one line of speech. A round boundary IS a
                # sentence boundary: whatever text a round ended with is
                # complete, because that round is over.
                remainder = buffer.strip()
                if remainder:
                    spoken.append(remainder)
                    yield remainder
                buffer = ""

            # RAN OUT OF ROUNDS. Reaching here with a tool_use stop reason
            # means the runner stopped because of max_iterations, not
            # because the model was done -- the tool results for that last
            # round will never be produced. The refusal and max_tokens
            # paths both break with their own stop reasons, so this
            # condition is unambiguous. Keeping that content would append
            # an assistant message holding a tool_use with no tool_result,
            # which the API rejects with a 400 on the NEXT send().
            if stop_reason == "tool_use":
                log.error(
                    "gave up after %d tool rounds in one turn — the model "
                    "kept calling tools without finishing", MAX_TOOL_ROUNDS,
                )
                spoken.append(STUCK_LINE)
                yield STUCK_LINE
                final_content = None

        except Exception:
            log.error("conversation failed", exc_info=True)
            yield ERROR_LINE
            spoken.append(ERROR_LINE)
            buffer = ""
            final_content = None  # same reason as the refusal path

        # Exactly one assistant message per turn, always. The model's own
        # closing content when the turn completed normally; otherwise
        # whatever ZEUS actually said out loud, so the transcript matches
        # what the user heard. Contributing nothing would leave two adjacent
        # user messages on the next send(), which the API rejects just as
        # surely as a dangling tool_use. Every path above flushes its own
        # trailing fragment before getting here, so `spoken` is already
        # complete when it is used as the fallback.
        #
        # TRUTHINESS, not `is not None`: an EMPTY content list passed the
        # old guard and appended {"role": "assistant", "content": []},
        # which the API rejects. Empty means the model contributed nothing,
        # which is exactly when the `spoken` fallback should take over.
        if final_content:
            self._messages.append(
                {"role": "assistant", "content": final_content}
            )
        elif spoken:
            self._messages.append(
                {"role": "assistant", "content": " ".join(spoken)}
            )

        if spoken:
            self._store.add_message(
                self._conversation_id, "assistant", " ".join(spoken)
            )
