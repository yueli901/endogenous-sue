# ES-SUE — stochastic user equilibrium with endogenous efficient subnetworks

Reference implementation and reproduction artifact for the paper
**"Stochastic User Equilibrium with Endogenous Efficient Subnetworks: Existence, Computation, and
Certification."**

Dial's efficient subnetwork — the arcs along which the cost to a destination strictly decreases — is
usually taken as given, or built once from free-flow costs and held fixed. Under congestion it is
neither: it is a function of the flow, and where two routes tie it is **set-valued**. This code solves
the equilibrium that follows, and **certifies** the answer rather than asserting it.

```python
from endogenous_sue import corpus, solve

net, od = corpus.load("SiouxFalls")
equilibrium = solve(net, od, mu=1.0)
equilibrium.x          # link flows
equilibrium.residual   # measured against the support the solver froze -- see below
```

---

## Quick start

```sh
python -m pip install -e ".[fast,figures,dev]"   # install
python reproduction/tools/fetch_networks.py      # benchmark networks, at a pinned commit
python tests/run_tests.py                        # 110 tests, a few minutes
```

Then reproduce something small and rebuild the paper's tables from the stored records:

```sh
make reproduce                                       # the exhibits that finish in minutes
python reproduction/tools/validate_results.py --strict
python reproduction/tools/build_tables.py
```

`numba` is optional. Every compiled kernel has a pure-Python counterpart with identical semantics, so
the package runs without it and only the speed changes — the test suite runs both ways in CI.

## What is here

```
src/endogenous_sue/   the method
  config                every setting, tolerance and corpus constant, in one file
  tntp, corpus          reading benchmark networks, and the study's network list
  network, costs        per-network arrays, adjacency, and the link cost
  shortestpath          cost-to-destination potentials
  subnetwork            which links are efficient, and in what order
  loading               recursive-logit loading on an acyclic support
  equilibrium           the two-phase solver, and the inclusion scheme
  certificate           testing a returned flow against the definition
  witness               the five-node network on which no strict equilibrium exists
  tied/                 the set-valued case: contested links, atoms, faces, the guaranteed branch

reproduction/         everything that regenerates the paper
  exhibits/             one script per exhibit, named for what it produces
  tools/                fetch data, validate results, build tables and figures
  cluster/              the Slurm scripts the long sweeps were dispatched with

results/              everything the paper reports
  reproduce/            the canonical sweep, one record per cell
  held_out/             a prospective run on networks the method was not developed against
  tables/               the generated LaTeX table bodies and numeric macros
  figures/              the paper's figures, as included

tests/  docs/  data/
```

Every setting lives in `src/endogenous_sue/config.py`. No tolerance, budget, threshold, grid or corpus
membership is defined anywhere else, and nothing is read from the environment, so a result cannot depend
on a shell. `config.py` also declares what each exhibit runs, so a sweep cannot quietly cover a different
set of networks from the one the tables report.

## How the method works

**Phase 1** builds the support from a running average of the congested cost while loading at the current
cost. Averaging the cost that *defines* the support is what stabilises the efficiency boundary: the
support stops changing and freezes in finite time whenever the equilibrium is off the tie boundary.
Rebuilding it from the instantaneous cost instead makes the map discontinuous and the support oscillates
indefinitely — that contrast is one of the paper's exhibits, measured by
`reproduction/exhibits/convergence.py`.

Phase 1 does not stop on an iteration count. Its job is to freeze the support, not to converge the flow.
It stops when the support is unchanged across several consecutive rebuilds, or earlier when the exactness
bound certifies that no efficiency gap can still change sign. The iteration limits in `config.py` mark a
run unconverged; they never define an answer.

**Phase 2** holds the frozen support fixed and solves the now-smooth fixed point to machine precision.

Where the support is genuinely set-valued the equilibrium is a **mixture**, and `tied/` finds one. A
guaranteed simplicial branch sits underneath the whole search: the accelerators above it inherit its
guarantee and never replace it.

## Why the certificate matters

A small residual is not an equilibrium. The two-phase solver reports its residual against the support it
*froze*; the model is defined against the support the flow's own cost *induces*. Those differ, and the
gap is not small:

```python
from endogenous_sue.certificate import certify

# Omitting the mixture tests the flow as a strict equilibrium against the supports its own cost
# induces. On many cells a two-phase solve alone does not pass this -- which is the point.
certify(net, od, 1.0, equilibrium.x).certified
```

Among the converged candidates in the reported sweep, most have a frozen-support residual below `1e-8`
while failing a reload on the support induced by their returned cost. Passing those cells needs the
tied-face solve, which returns the mixture that `certify` then checks.

`certify` takes the network, demand, dispersion, flow and mixture — and **nothing from the solver** —
then checks support admissibility, the mixture reconstruction, tie weights and demand conservation.
`conservation_error` and `reload_on_own_support` expose the two underlying questions separately: does
the flow carry the demand, and is it the loading on its own support?

## Results and provenance

`results/` holds one record per cell. Every record carries the package version, the commit, the
platform, the timestamp, the resolved run arguments, the gate revision and the full solver settings it
was produced under. A record naming a stored flow also carries that file's digest.

`reproduction/tools/validate_results.py --strict` checks all of it before any number is read off the
files. It refuses a file that mixes gate revisions or settings, reports partial coverage rather than
letting it read as complete, and names any record that stopped on an iteration limit instead of
converging.

## Scope

The method is an equilibrium formulation and certification framework with an efficient fixed-support
loading primitive. It is **not** a general-purpose fast solver for large networks: on the larger
benchmark networks the implementation does not reach certification within its declared computational
budget, usually because the support is still changing when the budget expires. `docs/known_limitations.md`
is specific about what does and does not hold.

## Documentation

| | |
|---|---|
| [docs/](docs/) | result schema, data sources, the solved-face contract, known limitations, the held-out protocol |
| [reproduction/README.md](reproduction/README.md) | how to run every exhibit, and what each costs |
| [CHANGELOG.md](CHANGELOG.md) | release history |
| [CONTRIBUTING.md](CONTRIBUTING.md) | reporting a bug, opening a pull request |

## Citing

See [CITATION.cff](CITATION.cff).

## Licence

MIT, see [LICENSE](LICENSE). The benchmark networks are not ours and are not redistributed; see
[docs/data_sources.md](docs/data_sources.md).
