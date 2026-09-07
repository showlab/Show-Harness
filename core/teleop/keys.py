"""Interactive multi-rollout keyboard gate, shared by the MVTOKEN entrypoints.

Data collection runs many rollouts back to back: the operator resets the scene, starts a
rollout, and either lets it end on its own (DONE / max_steps) or cuts it short when the arm
does something useless. Doing that by restarting the launcher each time loses the connected
session, the camera stream, and the vLLM warm-up, so both ``scripts/run_real_mvtoken.py`` (single-arm)
and ``scripts/run_real_dual_mvtoken.py`` (dual-arm) keep ONE process alive and drive the rollout
boundaries from stdin:

    Enter  -- between rollouts: the scene is set, start the next one
    't'    -- during a rollout: end it now and go back to the Enter gate
    'q'    -- finish (either place)

The watcher raises ``KeyboardInterrupt`` in the main thread via ``_thread.interrupt_main()``
for the during-rollout keys, which the runners already treat as a clean rollout end -- so no
runner needs to know this module exists.

Fail-safe by design: everything is a no-op when stdin is not a TTY (mock / CI / piped runs),
so ``wait_for_enter`` returns True immediately and nothing ever blocks an unattended run.
"""
from __future__ import annotations

import select
import sys
import threading
import _thread

try:  # POSIX-only single-key reading.
    import termios
    import tty

    _HAS_TERMIOS = True
except ImportError:  # pragma: no cover - non-POSIX
    _HAS_TERMIOS = False


class RolloutKeyWatcher(threading.Thread):
    """Background stdin watcher for interactive multi-rollout collection.

    While armed (during a rollout) it reads single keys in cbreak mode and, on the restart
    key or quit key, raises ``KeyboardInterrupt`` in the main thread -- which the rollout
    loops already catch and turn into a clean rollout end. The entrypoint then starts a new
    rollout (restart key) or exits ('q'). Disabled when stdin is not a TTY, so it never
    blocks mock / CI runs.
    """

    def __init__(self, restart_key: str = "t", quit_key: str = "q") -> None:
        super().__init__(daemon=True)
        self.restart_key = (restart_key or "t")[:1]
        self.quit_key = quit_key
        self.enabled = bool(_HAS_TERMIOS) and sys.stdin.isatty()
        self._armed = threading.Event()
        self._stop_event = threading.Event()
        self.restart = threading.Event()
        self.quit = threading.Event()
        # Between-rollout Enter-to-confirm gate, read on this same stdin-owning thread.
        self._confirm_mode = threading.Event()
        self._proceed = threading.Event()

    def arm(self) -> None:
        """Begin acting on keys for a new rollout (clears prior restart/quit flags)."""
        self.restart.clear()
        self.quit.clear()
        self._armed.set()

    def disarm(self) -> None:
        self._armed.clear()

    def wait_for_enter(self, prompt: str) -> bool:
        """Block between rollouts until Enter (proceed -> True) or the quit key (-> False).

        Read on the watcher thread, which already owns stdin in cbreak, so there is no second
        reader to contend with. No-op (returns True) when disabled (non-TTY / no termios)."""
        if not self.enabled:
            return True
        print(prompt, flush=True)
        self._proceed.clear()
        self._confirm_mode.set()
        try:
            while not self._stop_event.is_set():
                if self._proceed.wait(0.1):
                    return True
                if self.quit.is_set():
                    return False
        finally:
            self._confirm_mode.clear()
        return False

    def shutdown(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        if not self.enabled:
            return
        fd = sys.stdin.fileno()
        saved = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while not self._stop_event.is_set():
                if not select.select([sys.stdin], [], [], 0.2)[0]:
                    continue
                ch = sys.stdin.read(1)
                if self._confirm_mode.is_set():
                    # Between-rollout gate: Enter proceeds, the quit key finishes.
                    if ch in ("\n", "\r"):
                        self._proceed.set()
                    elif ch == self.quit_key:
                        self.quit.set()
                    continue
                if not self._armed.is_set():
                    continue  # ignore keys typed between rollouts
                if ch == self.quit_key:
                    self.quit.set()
                    self._armed.clear()
                    _thread.interrupt_main()
                elif ch == self.restart_key:
                    self.restart.set()
                    self._armed.clear()
                    _thread.interrupt_main()
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, saved)
