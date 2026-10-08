"""Whether the support window's envelope settles, for a cell whose support never does.

The inclusion scheme stops when one support recurs across the whole window. On the large cells it never
does: Chicago-Sketch at mu=1 saw 441,722 distinct supports in a million iterations. That is not by itself
a failure, and the strict-support criterion is the wrong gate to read it through. What the tied face is
actually built from is the *envelope* of the window -- the per-destination intersection and union -- and
:func:`endogenous_sue.tied.contested.contested_pairs` consumes only those two. A window whose envelope is
fixed defines one face, however much the individual support identity moves inside it.

So this asks the question the face solver asks. It runs the inclusion scheme in blocks, derives the base
(intersection) and union over each block independently, and reports whether they moved between blocks, by
how many links, and how large a face the envelope implies.

    python reproduction/tools/envelope_diagnostic.py --network Chicago-Sketch --mu 1 \
        --checkpoint checkpoints/certificate_sweep_Chicago-Sketch_mu1.npz \
        --iterations 8000 --interval 500 --out results/_envelope/chicago_mu1.json

A phase-3 checkpoint resumes the inclusion scheme natively. A phase-1 checkpoint holds a flow the
*two-phase* solver produced, which averages the cost and applies a dead band where this scheme does
neither, so its flow is not an iterate this scheme would have reached. It is still the right
neighbourhood to ask the question in, and the run is seeded from it rather than continued: `seeded` in
the output says which happened, and no result from a seeded run should be read as a continuation.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from hashlib import blake2b
from pathlib import Path

import numpy as np

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
sys.path.insert(0, str(_ROOT / "src"))

from endogenous_sue import corpus
from endogenous_sue.config import DEFAULTS, DeadlineReached
from endogenous_sue.equilibrium import load_checkpoint, save_checkpoint, solve_inclusion
from endogenous_sue.network import NetworkArrays

POPCOUNT = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(axis=1)


def _bits(packed: np.ndarray) -> int:
    """How many links one packed mask holds."""
    return int(POPCOUNT[packed].sum())


def _fingerprint(envelope: dict) -> str:
    """A stable name for one envelope, so two blocks can be compared without holding both."""
    hasher = blake2b(digest_size=8)
    for d in sorted(envelope):
        hasher.update(np.uint32(d).tobytes())
        hasher.update(envelope[d].tobytes())
    return hasher.hexdigest()


def _difference(left: dict, right: dict) -> int:
    """Links in one envelope and not the other, summed over destinations."""
    if left is None or right is None:
        return -1
    total = 0
    for d in set(left) | set(right):
        if d not in left or d not in right:
            total += _bits(right[d] if d not in left else left[d])
        else:
            total += _bits(left[d] ^ right[d])
    return total


def _face(topo, base: dict, union: dict) -> tuple[int, int, int]:
    """Contested pairs, the distinct links among them, and the tied node-pair groups they fall into.

    The pair count is what the active-set walk pays per round; the group count is what the ladder
    searches over, and the two differ by orders of magnitude on a large face. Reporting only the first
    made an affordable face look unaffordable once already.
    """
    pairs, links, groups = 0, set(), set()
    for d in base:
        contested = np.flatnonzero(np.unpackbits(union[d] & ~base[d]))
        pairs += contested.size
        for link in contested:
            links.add(int(link))
            tail, head = int(topo.tail[link]), int(topo.head[link])
            groups.add((min(tail, head), max(tail, head)))
    return pairs, len(links), len(groups)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--network", required=True)
    parser.add_argument("--mu", type=float, required=True)
    parser.add_argument("--checkpoint", default=None,
                        help="a phase-3 checkpoint to resume; omit to run the scheme from scratch")
    parser.add_argument("--iterations", type=int, default=8000)
    parser.add_argument("--interval", type=int, default=500,
                        help="iterations per block; each block's envelope is derived independently")
    parser.add_argument("--time-budget", type=float, default=None, help="hours")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    stored = load_checkpoint(args.checkpoint) if args.checkpoint else None
    if args.checkpoint and stored is None:
        print(f"no checkpoint at {args.checkpoint}", file=sys.stderr)
        return 2
    phase = int(stored.get("phase", 1)) if stored is not None else None
    if phase is not None and phase != 3:
        # A phase-1 flow is not a seed for this scheme, it is a different answer. Phase one applies the
        # potential dead band, whose whole purpose is to freeze the support onto one admissible
        # selection; seeded from it, Berlin-Prenzlauerberg-Center at mu=1 reports one distinct support
        # and no contested links, where the scheme run properly on the same cell finds nineteen
        # supports and thirteen contested pairs. Reading that as "no ties" would be exactly backwards.
        print(f"{args.checkpoint} is phase {phase}, not 3. Phase one freezes the support through the "
              f"dead band, so its flow reports ties that are not there to be found. Run without "
              f"--checkpoint to start the inclusion scheme from scratch instead.", file=sys.stderr)
        return 2

    net, od = corpus.load(args.network)
    topo = NetworkArrays.from_dict(net).topology()
    where = (f"resuming a phase-3 checkpoint at iteration {int(stored['iteration'])}"
             if stored is not None else "starting the inclusion scheme from scratch")
    print(f"{args.network} mu={args.mu:g}  links={topo.n_links}  {where}", flush=True)

    state = None
    if stored is not None:
        state = Path(args.out).with_suffix(".state.npz")
        save_checkpoint(state, phase=3, x=stored["x"], iteration=np.int64(stored["iteration"]),
                        seen_before=np.int64(stored.get("seen_before", 0)))

    settings = DEFAULTS
    started = time.perf_counter()
    deadline = None if args.time_budget is None else started + args.time_budget * 3600.0
    first = (int(stored["iteration"]) + 1) if stored is not None else 1
    # The last iteration of the last whole block, not one past it. Stopping one past leaves a block
    # holding a single iteration, whose base and union are that one support: it compares as a large
    # change against the settled envelope and as a face with nothing contested in it. Every verdict
    # in the first run of this script was inverted by that one iteration.
    whole = max(1, args.iterations // args.interval) * args.interval
    stop_after = first + whole - 1

    blocks: list[dict] = []
    block = {"base": None, "union": None, "seen": set(), "first": first, "move": 0.0}
    last_seen = [first - 1]
    previous = {"base": None, "union": None, "fingerprint": None}
    changed_at = {"base": None, "union": None}

    def close(last_iteration: int) -> None:
        if block["base"] is None:
            return
        partial = last_iteration - block["first"] + 1 < args.interval
        if partial:
            # Kept in the record so the run is fully accounted for, but never compared: an envelope
            # derived over fewer iterations than its neighbours is not evidence that it moved.
            blocks.append(dict(iterations=[block["first"], last_iteration], partial=True,
                               base_fingerprint=_fingerprint(block["base"]),
                               union_fingerprint=_fingerprint(block["union"]),
                               distinct_in_block=len(block["seen"]), move=block["move"]))
            return
        pairs, links, groups = _face(topo, block["base"], block["union"])
        base_moved = _difference(previous["base"], block["base"])
        union_moved = _difference(previous["union"], block["union"])
        if previous["base"] is not None and base_moved:
            changed_at["base"] = last_iteration
        if previous["union"] is not None and union_moved:
            changed_at["union"] = last_iteration
        blocks.append(dict(
            iterations=[block["first"], last_iteration], partial=False,
            base_fingerprint=_fingerprint(block["base"]),
            union_fingerprint=_fingerprint(block["union"]),
            base_changed=None if previous["base"] is None else bool(base_moved),
            union_changed=None if previous["union"] is None else bool(union_moved),
            base_symmetric_difference=None if previous["base"] is None else base_moved,
            union_symmetric_difference=None if previous["union"] is None else union_moved,
            last_base_change=changed_at["base"], last_union_change=changed_at["union"],
            distinct_in_block=len(block["seen"]),
            contested_pairs=pairs, contested_links=links, contested_groups=groups,
            move=block["move"]))
        print(f"  [{block['first']}..{last_iteration}] "
              f"base {blocks[-1]['base_fingerprint']} union {blocks[-1]['union_fingerprint']}  "
              f"symdiff base={base_moved if previous['base'] is not None else 'n.a.':>4} "
              f"union={union_moved if previous['union'] is not None else 'n.a.':>4}  "
              f"distinct={len(block['seen']):4d}  pairs={pairs} links={links} groups={groups}  "
              f"move={block['move']:.3e}", flush=True)
        previous["base"], previous["union"] = block["base"], block["union"]
        block.update(base=None, union=None, seen=set(), first=last_iteration + 1, move=0.0)

    def observe(iteration: int, packed: dict, move: float) -> None:
        if block["base"] is None:
            block["base"] = {d: bits.copy() for d, bits in packed.items()}
            block["union"] = {d: bits.copy() for d, bits in packed.items()}
        else:
            for d, bits in packed.items():
                block["base"][d] &= bits
                block["union"][d] |= bits
        hasher = blake2b(digest_size=8)
        for d in sorted(packed):
            hasher.update(packed[d].tobytes())
        block["seen"].add(hasher.digest())
        block["move"] = move
        last_seen[0] = iteration
        if iteration - block["first"] + 1 >= args.interval:
            close(iteration)
        if iteration >= stop_after:
            raise DeadlineReached(f"diagnostic ran its {args.iterations} iterations")

    stopped = "iterations"
    try:
        result = solve_inclusion(net, od, args.mu, settings, checkpoint=state,
                                 deadline=deadline, observer=observe,
                                 progress=lambda message: print(f"    {message}", flush=True))
        stopped = result.stopped_on
    except DeadlineReached as reached:
        stopped = str(reached)
    # The label has to be the last iteration the observer actually saw. The scheme can stop on its
    # own criterion well short of the request, and naming the partial block with the full range
    # reports a settled envelope as having been watched for far longer than it was.
    close(last_seen[0])

    whole_blocks = [b for b in blocks if not b.get("partial")]
    settled = whole_blocks[1:]
    enough = len(settled) >= 1
    verdict = dict(
        network=args.network, mu=args.mu,
        resumed_from=int(stored["iteration"]) if stored is not None else None,
        iterations_run=args.iterations,
        interval=args.interval, window=settings.inclusion_window, stopped_on=stopped,
        blocks_compared=len(settled),
        iterations_observed=last_seen[0] - first + 1,
        # ``None`` and not ``False`` where nothing could be compared. A single block answers the
        # question no more than zero do, and reporting that as "moving" inverts the finding.
        base_stable=(not any(b["base_changed"] for b in settled)) if enough else None,
        union_stable=(not any(b["union_changed"] for b in settled)) if enough else None,
        # Stability over every block is the wrong question for a run started from scratch: the first
        # blocks absorb the transient, where the flow is nowhere near equilibrium and the envelope is
        # correspondingly enormous. Berlin-Prenzlauerberg-Center at mu=1 reports 1,174 contested pairs
        # over the opening block and 13 once settled. What matters is when the envelope last moved and
        # how long it has held since.
        last_base_change=changed_at["base"], last_union_change=changed_at["union"],
        blocks_since_base_change=sum(1 for b in settled
                                     if changed_at["base"] is None
                                     or b["iterations"][0] > changed_at["base"]),
        blocks_since_union_change=sum(1 for b in settled
                                      if changed_at["union"] is None
                                      or b["iterations"][0] > changed_at["union"]),
        contested_pairs=whole_blocks[-1]["contested_pairs"] if whole_blocks else None,
        contested_links=whole_blocks[-1]["contested_links"] if whole_blocks else None,
        contested_groups=whole_blocks[-1]["contested_groups"] if whole_blocks else None,
        wall_seconds=time.perf_counter() - started, intervals=blocks)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(verdict, indent=2))
    if state is not None:
        state.unlink(missing_ok=True)

    def verdict_of(stable):
        return "STABLE" if stable else ("MOVING" if stable is not None else "NOT ESTABLISHED")

    print(f"\n{args.network} mu={args.mu:g}: "
          f"base {verdict_of(verdict['base_stable'])}, "
          f"union {verdict_of(verdict['union_stable'])} "
          f"over {verdict['blocks_compared']} compared block(s), "
          f"{verdict['iterations_observed']} iterations observed, stopped on {stopped}; "
          f"face carries {verdict['contested_pairs']} contested pairs over "
          f"{verdict['contested_links']} links in {verdict['contested_groups']} tied groups")
    print(f"  base last moved at {verdict['last_base_change']}, held for "
          f"{verdict['blocks_since_base_change']} block(s) since; "
          f"union last moved at {verdict['last_union_change']}, held for "
          f"{verdict['blocks_since_union_change']} block(s) since")
    if not enough:
        print("  the envelope was never compared against a second block: this is not a stability result")
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
