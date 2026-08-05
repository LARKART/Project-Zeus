"""Tool definitions and the action-logging wrapper. See spec §6.2, §12.

Every tool call writes an `actions` row. That table is the spine of the
Slice 2 dashboard, which can only show history that was recorded from the
beginning — hence it exists in Slice 1 despite having nothing to display yet.
"""
from __future__ import annotations

import functools
import logging
import time
from typing import Callable

from zeus.memory.journal import Journal
from zeus.memory.store import Store

log = logging.getLogger(__name__)

VALID_STATUSES = {"done", "partial", "missed", "carried"}


def logged_tool(
    store: Store, conversation_id: int, name: str, fn: Callable
) -> Callable:
    """Wrap a tool so every call is timed and recorded.

    A raising tool returns an error *string* rather than propagating, so the
    model can adapt (spec §10) instead of the turn collapsing.
    """

    @functools.wraps(fn)
    def wrapper(**kwargs):
        started = time.monotonic()
        try:
            result = fn(**kwargs)
            elapsed = int((time.monotonic() - started) * 1000)
            store.log_action(name, kwargs, result, True, elapsed,
                             conversation_id=conversation_id)
            return result
        except Exception as exc:  # noqa: BLE001 — deliberately broad
            elapsed = int((time.monotonic() - started) * 1000)
            log.error("tool %s failed", name, exc_info=True)
            store.log_action(name, kwargs, None, False, elapsed,
                             error=str(exc), conversation_id=conversation_id)
            return f"The {name} tool failed: {exc}"

    return wrapper


def build_tool_callables(
    store: Store, journal: Journal, conversation_id: int, local_date: str
) -> dict[str, Callable]:
    """The action-logged tool bodies, keyed by tool name.

    Kept separate from `build_tools` so tests and FakeConversation can call
    the real bodies directly. Depending on the @beta_tool decorator to
    preserve __name__ or to stay callable is an SDK-internals assumption
    this codebase does not make.
    """

    def _save_goal(text: str) -> str:
        store.set_goal(local_date, text)
        journal.append(f"Goal set: {text}")
        return f"Saved today's goal: {text}"

    def _record_outcome(status: str, notes: str | None = None) -> str:
        if status not in VALID_STATUSES:
            return (
                f"Cannot record status {status!r}. "
                f"Valid values are: {', '.join(sorted(VALID_STATUSES))}."
            )
        goal = store.get_goal(local_date)
        if goal is None:
            return "There is no goal recorded for today, so nothing to update."
        store.update_goal(goal.id, status=status, notes=notes)
        journal.append(f"Outcome: {status}" + (f" — {notes}" if notes else ""))
        return f"Recorded today's goal as {status}."

    return {
        "save_goal": logged_tool(store, conversation_id, "save_goal", _save_goal),
        "record_outcome": logged_tool(
            store, conversation_id, "record_outcome", _record_outcome
        ),
    }


def build_tools(
    store: Store, journal: Journal, conversation_id: int, local_date: str
) -> list[Callable]:
    """The Slice 1 tool surface, decorated for the Tool Runner.

    Deliberately tiny — two tools. Their purpose is to make the Tool Runner
    path real from day one so later slices add tools to a working mechanism
    rather than building the mechanism alongside their first tool.
    """
    from anthropic import beta_tool

    callables = build_tool_callables(store, journal, conversation_id, local_date)

    @beta_tool
    def save_goal(text: str) -> str:
        """Save the user's single goal for today.

        Args:
            text: The goal, in the user's own words, as concretely as they gave it.
        """
        return callables["save_goal"](text=text)

    @beta_tool
    def record_outcome(status: str, notes: str = "") -> str:
        """Record how today's goal turned out.

        Args:
            status: One of "done", "partial", "missed", or "carried".
            notes: Optional short detail the user gave, in their own words.
        """
        return callables["record_outcome"](status=status, notes=notes or None)

    return [save_goal, record_outcome]
