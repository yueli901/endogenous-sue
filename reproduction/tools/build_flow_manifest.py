"""Deposit the digests of the certified flow arrays, so the link to them survives without them.

The arrays the certificate sweep writes are large and are not tracked. Records in the other sinks name the
array each was computed against, and that link is what lets a reader tell a record derived from the
deposited flow from one derived from a re-solve. Without the arrays the check has nothing to compare to,
so a fresh clone could not run it at all.

This writes the digests alone. They are small, they are tracked, and they let the consistency of every
derived record be checked on a clone that will never hold the arrays themselves.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
FLOWS = REPO_ROOT / "results" / "reproduce" / "flows"
MANIFEST = REPO_ROOT / "results" / "reproduce" / "flows.manifest.json"


def main() -> int:
    if not FLOWS.is_dir():
        print(f"no flow directory at {FLOWS}; nothing to record", file=sys.stderr)
        return 1
    entries = {}
    for path in sorted(FLOWS.glob("*.npz")):
        blob = path.read_bytes()
        entries[path.name] = {"sha256": hashlib.sha256(blob).hexdigest()[:16], "bytes": len(blob)}
    MANIFEST.write_text(json.dumps({"directory": "flows", "files": entries}, indent=1) + "\n")
    total = sum(entry["bytes"] for entry in entries.values())
    print(f"{len(entries)} flow arrays, {total / 1e6:.1f} MB, recorded in "
          f"{MANIFEST.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
