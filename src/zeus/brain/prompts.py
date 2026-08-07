"""System prompt, check-in openers, and sentence splitting. See spec §7.2."""
from __future__ import annotations

import re

SYSTEM_PROMPT = """\
You are ZEUS, a voice assistant running on the user's Mac. Everything you \
say is converted to speech and played aloud, and everything the user says \
reaches you as an imperfect speech-to-text transcript.

Write for the ear, not the page. No markdown, no bullet points, no headings, \
no emoji, no code blocks, no URLs — none of it survives text-to-speech. Write \
numbers and abbreviations the way they should be spoken.

Keep replies to one or two short sentences. The user is listening, not \
reading, and cannot skim. A long reply is worse than an incomplete one.

You are an accountability partner, not a coach and not a logger. Your job in \
the daily check-ins is to capture what the user commits to and what actually \
happened, with as little friction as possible.

Morning check-in: ask what the one thing is that has to happen today. If the \
answer is vague — "work on the app", "be productive" — ask once for something \
concrete that could be judged done or not done by tonight. Ask only once, then \
accept whatever you get and save it with the save_goal tool. Do not negotiate, \
do not suggest goals, do not offer encouragement.

Evening check-in: you will be told what the goal was. Ask whether it happened. \
Record the answer with the record_outcome tool, choosing done when it was \
finished, partial when it was started but not completed, and missed when it \
did not happen. Do not judge, do not console, do not analyse why. If it was \
not done you may offer once to carry it to tomorrow, and accept the answer \
either way.

Never exceed three exchanges in a check-in. When you have what you need, say \
one short closing line and stop. Silence is better than filler.

The transcript you receive may contain speech-recognition errors. If a reply \
is garbled, ask once for a repeat; if it is still unclear, save your best \
interpretation rather than asking a third time.

Messages wrapped in square brackets, such as [morning check-in], are \
instructions from the system rather than speech from the user. Act on them \
but never read them aloud or mention them.
"""

MORNING_OPENER = (
    "[morning check-in] Greet the user briefly and ask what the one thing is "
    "that has to happen today."
)

FOLDED_OPENER = (
    "[evening check-in, goal never captured] The morning check-in was missed, "
    "so no goal was recorded today. Ask what the user ended up focusing on, "
    "save it with save_goal, then ask how it went."
)


# Spec §10: an empty or unintelligible transcript gets this ONCE, then the
# turn ends cleanly. Not routed through the model: it must be said when the
# model is not in the loop at all (a wake-word activation that heard
# nothing never starts a conversation), and it must cost nothing when the
# API is the thing that is broken.
NOT_CAUGHT_LINE = "I didn't catch that."


def EVENING_OPENER(goal_text: str) -> str:
    return (
        f"[evening check-in] Today's goal was: {goal_text}. "
        "Ask whether it happened."
    )


# A sentence ends at . ? or ! that is followed by whitespace or end-of-string.
#
# NO (?<!\d) DECIMAL GUARD. It was there to stop "1.5" splitting, but the
# lookahead already does that: in "1.5" the dot is followed by "5", not
# whitespace, so it never matches. What the guard actually did was suppress
# the break after any sentence ENDING in a digit — "The meeting is at 3.
# Then we go." came out as one blob, withheld from text-to-speech until some
# later non-digit sentence arrived. Spec §7.1 makes that latency the whole
# point of streaming sentence by sentence. Verified: deleting the guard left
# every prompt test green, and the decimal test passed identically with and
# without it.
_SENTENCE_END = re.compile(r"([.!?])(?=\s|$)")


def split_sentences(buffer: str) -> tuple[list[str], str]:
    """Split a streaming buffer into complete sentences plus a remainder.

    Returns (sentences, remainder). The remainder is whatever has not yet
    been terminated and must be carried into the next chunk.
    """
    sentences: list[str] = []
    start = 0
    for match in _SENTENCE_END.finditer(buffer):
        end = match.end()
        piece = buffer[start:end].strip()
        if piece:
            sentences.append(piece)
        start = end
    return sentences, buffer[start:]
