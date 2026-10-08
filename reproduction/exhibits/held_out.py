"""The certificate sweep over networks the study was never developed against.

Same code, same settings, same stopping rules, different networks and a different sink. Nothing here
selects a route, widens a tolerance or excludes a cell: what the solver does on a network nobody tuned
it against is the measurement, and a cell that fails or runs out of budget is a result of it.

``reproduction/exhibits/certificate_sweep.py`` holds the implementation. This module exists only to name the
declaration it runs against, so that the two sinks cannot be confused for one another and a held-out row
can never be read into the reported corpus.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from reproduction.exhibits.certificate_sweep import main

if __name__ == "__main__":
    raise SystemExit(main(exhibit_name="held_out"))
