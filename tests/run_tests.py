"""Run the test suite without pytest.

The tests are written in the ordinary pytest style, so ``python -m pytest tests`` is the normal way to run
them. This runner exists so that the suite can also be run in an environment where pytest is not
installed, which is the situation a reader is most likely to be in on a first checkout.
"""
from __future__ import annotations

import importlib.util
import sys
import time
import traceback
from pathlib import Path
from unittest import SkipTest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE.parent))


def load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    files = ([HERE / f"test_{a}.py" if not a.endswith(".py") else Path(a) for a in argv]
             if argv else sorted(HERE.glob("test_*.py")))
    passed = failed = skipped = 0
    failures = []
    for path in files:
        try:
            module = load(path)
        except SkipTest as exc:
            skipped += 1
            print(f"  SKIP  {path.stem}  ({exc})", flush=True)
            continue
        for name in sorted(vars(module)):
            if not name.startswith("test_"):
                continue
            fn = getattr(module, name)
            if not callable(fn):
                continue
            started = time.perf_counter()
            try:
                fn()
                passed += 1
                print(f"  PASS  {path.stem}.{name}  ({time.perf_counter() - started:.2f}s)",
                      flush=True)
            except SkipTest as exc:
                skipped += 1
                print(f"  SKIP  {path.stem}.{name}  ({exc})", flush=True)
            except Exception:
                failed += 1
                failures.append((f"{path.stem}.{name}", traceback.format_exc()))
                print(f"  FAIL  {path.stem}.{name}  ({time.perf_counter() - started:.2f}s)",
                      flush=True)
    for name, tb in failures:
        print(f"\n{'=' * 78}\n{name}\n{'=' * 78}\n{tb}")
    summary = f"{passed} passed, {failed} failed"
    if skipped:
        summary += f", {skipped} skipped"
    print(f"\n{summary}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
