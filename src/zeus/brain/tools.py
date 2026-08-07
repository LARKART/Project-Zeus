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

# A goal is one spoken sentence. The cap exists because nothing else bounds
# what lands here: `text` comes straight from a Whisper transcript of a 30s
# listen window, and whatever is stored replays into every later prompt AND
# into the journal, for good. Truncating rather than rejecting keeps a
# long-winded but genuine answer usable.
MAX_GOAL_CHARS = 500


def logged_tool(
    store: Store, conversation_id: int, name: str, fn: Callable
) -> Callable:
    """Wrap a tool so every call is timed and recorded.

    A raising tool is reported to the model as an ERROR, not as a successful
    result (spec §10: "Return is_error: true so the model can adapt; log to
    actions with ok=0"). It used to RETURN the error as a string, which the
    Tool Runner cannot distinguish from a normal result — so it emitted a
    plain tool_result with no `is_error`, and the model was told the call
    had worked. The action log was right the whole time (ok=False, error
    set); only the model was misinformed, and a model that thinks a failing
    call succeeded has no reason to stop retrying it.

    anthropic.lib.tools.ToolError rather than letting the original
    exception escape: the runner catches BOTH and sets is_error either way,
    but ToolError's content is used verbatim, so the model reads "The
    save_goal tool failed: disk full" instead of repr(RuntimeError(...)).
    The turn still does not collapse — the runner turns this into a
    tool_result and hands it back to the model.
    """
    from anthropic.lib.tools import ToolError

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
            raise ToolError(f"The {name} tool failed: {exc}") from exc

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
        # VALIDATED, because nothing upstream does it. text="" was stored
        # and journalled as "Goal set: " — a goal row that reads as an
        # answered morning check-in while carrying nothing, which the
        # evening check-in then recalls back to the user as an empty
        # sentence. And text="x" * 100000 was stored whole, replaying into
        # every later prompt and into the journal permanently.
        #
        # Empty is REJECTED (the model can ask again); too long is
        # TRUNCATED (the answer is real, just long-winded — losing it
        # entirely would be worse than losing its tail).
        text = (text or "").strip()
        if not text:
            return (
                "No goal text was given, so nothing was saved. Ask the user "
                "what the one thing is and save what they actually say."
            )
        if len(text) > MAX_GOAL_CHARS:
            log.warning(
                "goal text was %d characters; truncated to %d",
                len(text), MAX_GOAL_CHARS,
            )
            text = text[:MAX_GOAL_CHARS].rstrip()
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
