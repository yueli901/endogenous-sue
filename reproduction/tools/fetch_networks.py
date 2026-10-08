"""Fetch the benchmark networks at the commit the reported results were produced against.

The networks are the Transportation Networks for Research collection, which is not redistributed here.
This clones it at a pinned commit, so a reproduction reads the same files rather than whatever the
upstream repository holds today.

    python reproduction/tools/fetch_networks.py [--dest DIR] [--commit SHA]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
sys.path.insert(0, str(_ROOT / "src"))

from endogenous_sue.config import DATA_ROOT, PINNED_COMMIT

UPSTREAM = "https://github.com/bstabler/TransportationNetworks.git"


def run(*args):
    result = subprocess.run(args, text=True)
    if result.returncode != 0:
        raise SystemExit(f"command failed: {' '.join(args)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dest", default=str(DATA_ROOT))
    parser.add_argument("--commit", default=PINNED_COMMIT)
    args = parser.parse_args()

    dest = Path(args.dest)
    if (dest / ".git").is_dir():
        print(f"{dest} already exists; fetching and checking out {args.commit[:12]}")
        run("git", "-C", str(dest), "fetch", "--all", "--quiet")
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"cloning {UPSTREAM} into {dest}")
        run("git", "clone", "--quiet", UPSTREAM, str(dest))
    run("git", "-C", str(dest), "checkout", "--quiet", args.commit)

    head = subprocess.run(["git", "-C", str(dest), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    if head != args.commit:
        raise SystemExit(f"checked out {head}, expected {args.commit}")
    networks = sorted(p.name for p in dest.iterdir() if p.is_dir() and not p.name.startswith("."))
    print(f"ready at {head[:12]}: {len(networks)} network directories")


if __name__ == "__main__":
    main()
