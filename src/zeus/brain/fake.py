"""Conversation test double. Replays a script; never touches the network.

Can also invoke real tool callables, so an end-to-end test exercises the
genuine brain -> tool -> action log -> database path instead of writing the
data it then asserts on.
"""
from __future__ import annotations

from typing import Callable, Iterator

ToolCall = tuple[str, dict]


class FakeConversation:
    def __init__(
        self,
        script: dict[str, list[str]] | None = None,
        tools: dict[str, Callable] | None = None,
        tool_calls: list[list[ToolCall]] | None = None,
    ) -> None:
        self._script = script or {}
        self._tools = tools or {}
        self._tool_calls = list(tool_calls or [])
        self.sent: list[str] = []
        self.invoked: list[ToolCall] = []

    def send(self, text: str) -> Iterator[str]:
        turn = len(self.sent)
        self.sent.append(text)

        planned = self._tool_calls[turn] if turn < len(self._tool_calls) else []
        for name, kwargs in planned:
            tool = self._tools.get(name)
            if tool is None:
                raise KeyError(
                    f"FakeConversation was asked to call {name!r} but was given "
                    f"only {sorted(self._tools)}"
                )
            tool(**kwargs)
            self.invoked.append((name, kwargs))

        for sentence in self._script.get(text, ["Got it."]):
            yield sentence
