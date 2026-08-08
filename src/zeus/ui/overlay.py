"""The floating HUD — ZEUS's face, the Siri-style panel that appears on wake.

WHY NATIVE AND NOT A WEB PAGE. The panel has to appear over whatever you are
already doing, including a fullscreen app, without stealing focus or
bouncing an icon in the Dock. Only a real NSPanel does that: a browser
window cannot float above fullscreen Spaces, cannot avoid taking key focus,
and cannot be summoned by a background process. PyObjC is already a
dependency (pyobjc-framework-quartz, for the idle-time check), and AppKit
comes with it, so this costs no new package.

THE MAIN THREAD IS APPKIT'S. Cocoa requires its run loop on thread 0, and
every UI mutation on it. ZEUS's own loop therefore moves to a background
thread when the overlay is on (see cli.cmd_run), and every method here is
safe to call from any thread -- each one marshals onto the main queue and
returns immediately, so a slow redraw can never stall the voice loop.

DEGRADES TO NOTHING. `build_overlay` returns a NullOverlay when AppKit is
unavailable or the overlay is switched off, so every caller can talk to the
same interface without asking whether a screen exists. That is what keeps
this importable under pytest on a headless runner.
"""
from __future__ import annotations

import logging
from typing import Protocol

log = logging.getLogger(__name__)

# States the panel can show. Kept as plain strings rather than an enum so a
# caller in another module never has to import this one just to name a state.
IDLE = "idle"
LISTENING = "listening"
THINKING = "thinking"
SPEAKING = "speaking"


class Overlay(Protocol):
    """What the ritual and the wake path are allowed to ask of the UI."""

    def show(self, state: str, text: str = "") -> None: ...
    def hide(self) -> None: ...
    def run_forever(self) -> None: ...
    def stop(self) -> None: ...


class NullOverlay:
    """No screen, no panel, no problem.

    Used by every test and by `zeus run --no-overlay`. It is not a stub for
    an unfinished feature -- a daemon under launchd with no login session
    genuinely has nowhere to draw, and must keep working.
    """

    def show(self, state: str, text: str = "") -> None:
        log.debug("overlay: %s %r", state, text)

    def hide(self) -> None:
        log.debug("overlay: hide")

    def run_forever(self) -> None:
        raise RuntimeError(
            "NullOverlay has no run loop; run the daemon on the main thread"
        )

    def stop(self) -> None:
        pass


def build_overlay(enabled: bool = True) -> Overlay:
    """A real panel if one can be drawn, otherwise a silent no-op."""
    if not enabled:
        return NullOverlay()
    try:
        return MacOverlay()
    except Exception:
        # An unusable overlay must never take the daemon down with it: ZEUS
        # without a face still hears, thinks and speaks.
        log.warning("could not create the overlay; continuing without it",
                    exc_info=True)
        return NullOverlay()


class MacOverlay:
    """A borderless, non-activating NSPanel that floats above everything."""

    WIDTH = 460.0
    HEIGHT = 132.0
    MARGIN = 48.0

    def __init__(self) -> None:
        # Imported HERE, not at module scope: this module is imported by the
        # CLI on every command, and importing AppKit connects to the window
        # server -- which fails, loudly and slowly, over SSH or in CI.
        from AppKit import (
            NSApplication, NSApplicationActivationPolicyAccessory, NSColor,
            NSFont, NSMakeRect, NSPanel, NSScreen, NSTextField,
            NSVisualEffectView,
        )

        self._app = NSApplication.sharedApplication()
        # ACCESSORY, not REGULAR: no Dock icon, no menu bar, no app switcher
        # entry. ZEUS is a daemon that occasionally draws, not an app you
        # command-tab to.
        self._app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)

        screen = NSScreen.mainScreen().frame()
        rect = NSMakeRect(
            (screen.size.width - self.WIDTH) / 2.0,
            screen.size.height - self.HEIGHT - self.MARGIN * 2,
            self.WIDTH, self.HEIGHT,
        )
        panel = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            rect, self._style_mask(), 2, False
        )
        panel.setOpaque_(False)
        panel.setBackgroundColor_(NSColor.clearColor())
        panel.setHasShadow_(True)
        panel.setLevel_(self._status_level())
        panel.setCollectionBehavior_(self._collection_behavior())
        # The panel must never become key: typing continues to go to whatever
        # you were working in while ZEUS listens.
        panel.setBecomesKeyOnlyIfNeeded_(True)
        panel.setHidesOnDeactivate_(False)

        effect = NSVisualEffectView.alloc().initWithFrame_(
            NSMakeRect(0, 0, self.WIDTH, self.HEIGHT)
        )
        effect.setMaterial_(self._material())
        effect.setBlendingMode_(0)        # behind-window: the Siri look
        effect.setState_(1)               # active regardless of app focus
        effect.setWantsLayer_(True)
        effect.layer().setCornerRadius_(22.0)
        effect.layer().setMasksToBounds_(True)
        panel.setContentView_(effect)

        self._state_label = self._label(
            NSMakeRect(28, self.HEIGHT - 52, self.WIDTH - 56, 22),
            NSFont.systemFontOfSize_weight_(13.0, 0.3),
            NSColor.secondaryLabelColor(), NSTextField,
        )
        self._text_label = self._label(
            NSMakeRect(28, 22, self.WIDTH - 56, 56),
            NSFont.systemFontOfSize_weight_(19.0, 0.2),
            NSColor.labelColor(), NSTextField,
        )
        effect.addSubview_(self._state_label)
        effect.addSubview_(self._text_label)

        self._panel = panel
        self._visible = False

    # -- construction helpers, each isolating one AppKit constant ---------
    @staticmethod
    def _style_mask() -> int:
        from AppKit import (
            NSWindowStyleMaskBorderless, NSWindowStyleMaskNonactivatingPanel,
        )
        return NSWindowStyleMaskBorderless | NSWindowStyleMaskNonactivatingPanel

    @staticmethod
    def _status_level() -> int:
        from AppKit import NSStatusWindowLevel
        return NSStatusWindowLevel

    @staticmethod
    def _collection_behavior() -> int:
        from AppKit import (
            NSWindowCollectionBehaviorCanJoinAllSpaces,
            NSWindowCollectionBehaviorFullScreenAuxiliary,
            NSWindowCollectionBehaviorStationary,
        )
        # All three matter: join-all-spaces so it follows you between
        # desktops, fullscreen-auxiliary so it draws OVER a fullscreen app
        # instead of switching away from it, stationary so Mission Control
        # does not fling it around.
        return (
            NSWindowCollectionBehaviorCanJoinAllSpaces
            | NSWindowCollectionBehaviorFullScreenAuxiliary
            | NSWindowCollectionBehaviorStationary
        )

    @staticmethod
    def _material() -> int:
        from AppKit import NSVisualEffectMaterialHUDWindow
        return NSVisualEffectMaterialHUDWindow

    @staticmethod
    def _label(frame, font, colour, NSTextField):
        field = NSTextField.alloc().initWithFrame_(frame)
        field.setBezeled_(False)
        field.setDrawsBackground_(False)
        field.setEditable_(False)
        field.setSelectable_(False)
        field.setFont_(font)
        field.setTextColor_(colour)
        field.setStringValue_("")
        return field

    # -- the interface ---------------------------------------------------
    def _on_main(self, block) -> None:
        """Run `block` on the main thread and do not wait for it.

        Not waiting is the point: show() is called from the wake thread in
        the middle of a conversation, and blocking it on a redraw would put
        UI latency directly into the voice loop.
        """
        from Foundation import NSOperationQueue
        NSOperationQueue.mainQueue().addOperationWithBlock_(block)

    def show(self, state: str, text: str = "") -> None:
        caption = {
            LISTENING: "Listening…",
            THINKING: "Thinking…",
            SPEAKING: "ZEUS",
            IDLE: "ZEUS",
        }.get(state, "ZEUS")

        def paint() -> None:
            self._state_label.setStringValue_(caption)
            self._text_label.setStringValue_(text)
            if not self._visible:
                # orderFrontRegardless, NOT makeKeyAndOrderFront: the panel
                # appears without ZEUS stealing focus from what you are doing.
                self._panel.orderFrontRegardless()
                self._visible = True

        self._on_main(paint)

    def hide(self) -> None:
        def dismiss() -> None:
            self._panel.orderOut_(None)
            self._visible = False

        self._on_main(dismiss)

    def run_forever(self) -> None:
        """Hand the main thread to AppKit. Never returns until stop()."""
        self._app.run()

    def stop(self) -> None:
        self._on_main(lambda: self._app.terminate_(None))
