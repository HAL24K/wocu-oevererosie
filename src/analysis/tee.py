"""Keep what a script prints: from ``tee_to(path)`` on, stdout also goes to a file."""

from __future__ import annotations

import atexit
import sys
from pathlib import Path


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for s in self.streams:
            s.write(text)

    def flush(self):
        for s in self.streams:
            s.flush()


def tee_to(path: Path) -> Path:
    """Copy everything printed from now on to ``path`` (for module-level scripts)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    f = path.open("w")
    terminal = sys.stdout
    sys.stdout = _Tee(terminal, f)

    def close():
        sys.stdout = terminal
        f.close()
        print(f"→ report kept in {path}")

    atexit.register(close)
    return path
