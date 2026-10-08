# Naming

One name per thing. Nothing here is an abbreviation invented for convenience: each form below is either
the name itself or that name under a transformation the tooling requires.

## The names

| what | name | why this form |
|---|---|---|
| the equilibrium concept, in prose and in the paper | **ES-SUE** | the concept: stochastic user equilibrium in which the efficient subnetwork is endogenous |
| the repository, and the installed distribution | **endogenous-sue** | the concept spelled out, hyphenated as distribution names are |
| the import package | **endogenous_sue** | the distribution name with the underscore Python requires |
| environment variables | **ENDOGENOUS_SUE_*** | the import name, upper-cased as environment variables are |

So `pip install endogenous-sue` then `import endogenous_sue`, and the thing the package computes is called
ES-SUE.

The paper's title states the concept in full and does not use the abbreviation in its prose. The
abbreviation exists for the repository, for tables, and for talking about the work.

## Environment variables

| variable | effect |
|---|---|
| `ENDOGENOUS_SUE_DATA` | read the benchmark networks from this directory instead of `data/TransportationNetworks` |

No solver setting is read from the environment. Published records carry the resolved `Settings`
dataclass, and the test suite enforces that `ENDOGENOUS_SUE_DATA` is the only environment variable read by
the package.

## Vocabulary

Terms that carry a specific meaning here, in the sense the paper uses them.

| term | meaning |
|---|---|
| **efficient subnetwork** | the arcs along which the cost to a given destination strictly decreases. Under congestion it is a function of the flow, which is what "endogenous" refers to |
| **support** | the efficient subnetwork a particular assignment is carried on. Set-valued at a tie |
| **tie** | two routes of equal cost to a destination, so the strict and closed efficient sets differ and no single support is determined |
| **contested arc** | an arc on which the supports late in the stabilising phase disagree, so the tie there is unresolved |
| **tied face** | the set of mixtures over admissible supports that the equilibrium may lie in when ties are present |
| **relaxed equilibrium** | the equilibrium concept the paper defines: a mixture over admissible supports, reducing to the ordinary notion where there are no ties |
| **certificate** | the a posteriori test that a returned flow satisfies the definition. Two clauses: admissibility of the supports carrying flow, and conservation of demand |
| **dispersion** | the logit parameter, written mu. Larger means a sharper choice, hence a tighter concentration on efficient routes |
| **exclusion margin** | the smallest efficiency gap over arcs the support excludes |
| **isolation margin** | the smallest gap over every arc whose efficiency could still change. Defined only where the tie set is empty |
| **corpus** | the seventeen benchmark networks the study reports, listed in `endogenous_sue.corpus.CORPUS` |
| **cell** | one (network, dispersion) combination. The unit a result row describes |
| **vintage** | the pair of solver settings that have moved results across the whole corpus. Recorded on every row so that rows from either side are never read together |

## Rule

A rename is global and verified. When a name changes, sweep every file, filename, figure and code path in
one pass, then grep for the old form and confirm zero occurrences remain. Old names are recorded in
`archive/README.md`, which is not part of the release, so that an old reference can still be resolved
without the superseded name reappearing in an active file.
