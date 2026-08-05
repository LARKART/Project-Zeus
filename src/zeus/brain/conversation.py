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

REFUSAL_LINE = "Sorry, I can't help with that one."
ERROR_LINE = "I can't reach my brain right now. I'll try again later."


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
        try:
            runner = self._client.beta.messages.tool_runner(
                model=self._config.model,
                max_tokens=MAX_TOKENS,
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
                if getattr(final, "stop_reason", None) == "refusal":
                    log.warning("model refused the request")
                    spoken.append(REFUSAL_LINE)
                    yield REFUSAL_LINE
                    buffer = ""
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

        except Exception:
            log.error("conversation failed", exc_info=True)
            yield ERROR_LINE
            spoken.append(ERROR_LINE)
            buffer = ""

        if final_content is not None:
            self._messages.append(
                {"role": "assistant", "content": final_content}
            )

        remainder = buffer.strip()
        if remainder:
            spoken.append(remainder)
            yield remainder

        if spoken:
            self._store.add_message(
                self._conversation_id, "assistant", " ".join(spoken)
            )
