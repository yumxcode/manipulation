"""终端单键输入（无需 X11）：后台线程读 stdin，主循环用 poll() 取按键。"""

from __future__ import annotations

import queue
import select
import sys
import termios
import threading
import tty


class KeyReader:
    def __init__(self):
        self.q: queue.Queue[str] = queue.Queue()
        self._fd = sys.stdin.fileno()
        self._old = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        self._running = True
        self._t = threading.Thread(target=self._loop, daemon=True, name="keys")
        self._t.start()

    def _loop(self):
        while self._running:
            r, _, _ = select.select([sys.stdin], [], [], 0.05)
            if r:
                ch = sys.stdin.read(1)
                if ch:
                    self.q.put(ch)

    def poll(self) -> str | None:
        try:
            return self.q.get_nowait()
        except queue.Empty:
            return None

    def close(self):
        self._running = False
        self._t.join(timeout=0.5)
        termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
