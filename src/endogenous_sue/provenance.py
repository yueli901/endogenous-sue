"""Provenance stamped onto every result record.

A stored number is interpretable only against the code and the settings that produced it. Both are
recorded on every record, together with the package version, the commit, the platform and the resolved
arguments of the run, so that a record can be matched to its origin without consulting anything else.
"""
from __future__ import annotations

import platform
import subprocess
import time
from functools import lru_cache

from . import __version__
from .config import CODE_PATHS, GATE_REVISION, REPO_ROOT, Settings

__all__ = ["commit", "stamp"]


@lru_cache(maxsize=1)
def commit() -> str | None:
    """Current commit, with ``-dirty`` appended when the code differs from it.

    A commit alone is not enough: a result produced from edited but uncommitted code would otherwise be
    recorded as though it came from the committed version. Dirtiness is judged over the code directories
    rather than the whole tree, because the whole tree includes the result files a run is writing.
    """
    try:
        revision = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                                  capture_output=True, text=True, timeout=10)
        if revision.returncode != 0:
            return None
        modified = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "status", "--porcelain", "--", *CODE_PATHS],
            capture_output=True, text=True, timeout=10)
        suffix = "-dirty" if modified.stdout.strip() else ""
        return revision.stdout.strip()[:12] + suffix
    except (OSError, subprocess.SubprocessError):
        return None


def stamp(record: dict, settings: Settings, run_spec: dict | None = None) -> dict:
    """Return ``record`` with the provenance fields added.

    The platform is recorded rather than the machine: what a reader needs in order to compare a re-run is
    the operating system and the architecture, and a hostname identifies a private computer.
    """
    fields = dict(version=__version__,
                  commit=commit(),
                  gate_revision=GATE_REVISION,
                  settings=settings.as_record(),
                  platform=platform.platform(),
                  python=platform.python_version(),
                  timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    if run_spec is not None:
        fields["run_spec"] = run_spec
    return dict(record, **fields)
