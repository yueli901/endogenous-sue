"""Resolve a sweep's cells, so a job array and the reported corpus cannot disagree.

A job script that repeats the network names duplicates a declaration, and the copy drifts: the array this
replaces ran an excluded network and never ran Austin, which is a reported row. Here the cells come from
:mod:`endogenous_sue.config`, and an index outside the declared range is an error rather than a silently
skipped cell.

    python reproduction/tools/cells.py --list                 every sweep and how many cells it declares
    python reproduction/tools/cells.py corpus --count         the array size the job needs
    python reproduction/tools/cells.py corpus 3               the network that array index runs
    python reproduction/tools/cells.py certificate_sweep 12   the network and dispersion that index runs
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
sys.path.insert(0, str(_ROOT / "src"))

from endogenous_sue.config import CORPUS_CELLS, EXHIBIT_CELLS, PILOT_CELL

#: Every sweep a job array can index. The corpus array runs a whole network per task, so its cells are
#: networks; every other exhibit is indexed cell by cell.
SWEEPS = dict({"corpus": CORPUS_CELLS, "pilot": (PILOT_CELL,)}, **EXHIBIT_CELLS)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sweep", nargs="?", choices=sorted(SWEEPS))
    parser.add_argument("index", nargs="?", type=int)
    parser.add_argument("--count", action="store_true", help="print the number of cells and exit")
    parser.add_argument("--list", action="store_true", help="print every sweep and its size")
    args = parser.parse_args(argv)

    if args.list:
        for sweep in sorted(SWEEPS):
            cells = SWEEPS[sweep]
            print(f"{sweep:20} {len(cells):4} cells  "
                  f"{len({cell[0] for cell in cells}):3} networks")
        return 0
    if args.sweep is None:
        parser.error("name a sweep, or pass --list")

    cells = SWEEPS[args.sweep]
    if args.count:
        print(len(cells))
        return 0
    if args.index is None:
        parser.error("give an index, or --count")
    if not 0 <= args.index < len(cells):
        raise SystemExit(f"index {args.index} is outside the {len(cells)} cells of the {args.sweep} "
                         f"sweep; the job array and reproduction/tools/cells.py disagree")
    print(" ".join(f"{item:g}" if isinstance(item, float) else str(item)
                   for item in cells[args.index]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
