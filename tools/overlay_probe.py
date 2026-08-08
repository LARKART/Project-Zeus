import sys, threading, time, json
sys.path.insert(0, "src")
from zeus.ui.overlay import MacOverlay, LISTENING, THINKING, SPEAKING

log = open("/tmp/overlay_probe.json", "w")
overlay = MacOverlay()
obs = []

def note(tag):
    p = overlay._panel
    obs.append({
        "step": tag,
        "visible": bool(p.isVisible()),
        "onActiveSpace": bool(p.isOnActiveSpace()),
        "alpha": round(float(p.alphaValue()), 2),
        "level": int(p.level()),
        "frame": [round(v) for v in (p.frame().origin.x, p.frame().origin.y,
                                     p.frame().size.width, p.frame().size.height)],
        "state": overlay._state_label.stringValue(),
        "text": overlay._text_label.stringValue(),
    })
    log.write(json.dumps(obs[-1]) + "\n"); log.flush()

def script():
    time.sleep(0.5); note("before-show")
    overlay.show(LISTENING, ""); time.sleep(1.2); note("listening")
    overlay.show(THINKING, "What's my one thing today?"); time.sleep(1.2); note("thinking")
    overlay.show(SPEAKING, "Ship Zeus. That's what you said this morning."); time.sleep(2.0); note("speaking")
    overlay.hide(); time.sleep(0.8); note("after-hide")
    log.write("DONE\n"); log.flush(); log.close()
    overlay.stop()

threading.Thread(target=script, daemon=True).start()
overlay.run_forever()
