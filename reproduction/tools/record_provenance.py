"""Record the environment a reproduction would run in, and checksum the results it would read.

Written to ``results/provenance/``. This is what makes "the numbers came out different" a question that
can be
answered rather than argued about.

    python reproduction/tools/record_provenance.py
"""
from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
sys.path.insert(0, str(ROOT / "src"))

PROVENANCE = ROOT / "results" / "provenance"
RESULTS = ROOT / "results" / "reproduce"
DATA = ROOT / "data" / "TransportationNetworks"


def package_versions() -> list[str]:
    lines = [f"python {platform.python_version()}"]
    for module in ("numpy", "scipy", "numba", "matplotlib"):
        try:
            lines.append(f"{module} {__import__(module).__version__}")
        except ImportError:
            lines.append(f"{module} not installed")
    try:
        import endogenous_sue
        lines.append(f"endogenous_sue {endogenous_sue.__version__}")
        from endogenous_sue.provenance import GATE_REVISION, commit
        lines.append(f"endogenous_sue commit {commit()}")
        lines.append(f"endogenous_sue gate_revision {GATE_REVISION}")
    except ImportError as exc:
        lines.append(f"endogenous_sue not importable: {exc}")
    return lines


def machine() -> list[str]:
    lines = [f"platform {platform.platform()}",
             f"machine {platform.machine()}",
             f"processor {platform.processor() or 'unknown'}"]
    try:
        import os
        lines.append(f"cpu_count {os.cpu_count()}")
    except Exception:
        pass
    return lines


def data_commit() -> str:
    if not (DATA / ".git").is_dir():
        return "network data absent"
    result = subprocess.run(["git", "-C", str(DATA), "rev-parse", "HEAD"],
                            capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def checksums() -> list[str]:
    lines = []
    for path in sorted(RESULTS.glob("*.jsonl")):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(ROOT)}  ({path.stat().st_size} bytes)")
    return lines


def main():
    PROVENANCE.mkdir(exist_ok=True)
    (PROVENANCE / "software_versions.txt").write_text("\n".join(package_versions()) + "\n")
    # The hostname is deliberately absent: it identifies a computer, not an environment.
    (PROVENANCE / "machine.txt").write_text(
        "\n".join(machine() + [f"network data commit {data_commit()}"]) + "\n")
    (PROVENANCE / "checksums.txt").write_text("\n".join(checksums()) + "\n")
    print(f"wrote {PROVENANCE / 'software_versions.txt'}")
    print(f"wrote {PROVENANCE / 'machine.txt'}")
    print(f"wrote {PROVENANCE / 'checksums.txt'} ({len(checksums())} result files)")


if __name__ == "__main__":
    main()
