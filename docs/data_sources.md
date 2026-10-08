# Data

## The networks

All networks come from the Transportation Networks for Research collection at
<https://github.com/bstabler/TransportationNetworks>, in TNTP format. They are not redistributed here.

`reproduction/tools/fetch_networks.py` clones the collection and checks out commit `d1639b4e` (19 June 2025), which
is the version the reported results were produced against and which is pinned in `endogenous_sue.corpus`. The
script verifies the checkout and refuses a mismatch. Set `ENDOGENOUS_SUE_DATA` to read from a clone elsewhere.

The corpus is seventeen networks, listed in `endogenous_sue.corpus.CORPUS`, from a four-link example to a
75,379-link city.

## What the reader does with a network file

Two details of the format matter enough to be worth stating, because both silently produce wrong output
rather than an error.

**Node numbering.** Everything downstream indexes per-node arrays by `id - 1`, which assumes the file
numbers its nodes from one. Munich numbers its by OpenStreetMap id. `tntp.parse_net` detects this and
renumbers, preserving order so that centroids stay first, and returns the mapping. `parse_trips` must be
given that same mapping, and it refuses to guess one rather than deriving its own: a mapping derived from
the trips file alone would number origins by their own sort and attach each origin's demand to a different
node.

**Centroids.** A zero-cost connector leaves the cost-to-destination unchanged, so the strict-decrease rule
never admits it. Three separate questions follow — may flow leave a zone, may it arrive at its
destination, may it transit a foreign centroid — and they are answered by three separate conditions in
`subnetwork.active_for_dest`. Bundling any two of them loses demand, and it does so invisibly, because the
equilibrium certificate asks whether the arcs carrying flow are efficient and never whether the flow
arrived.

## Excluded networks

Two corpus networks are outside the reported tables. Barcelona and Winnipeg fold capacity into the BPR
coefficient and leave unit capacity on every link, with exponents up to 16.83, so no fixed
representability guard and the strictly-increasing-cost assumption can both hold. They are filtered where
records are loaded rather than in each table, so regenerating a table cannot reintroduce them.

Munich is not in the corpus, and the criterion says so rather than the list: its trips file covers 284 of
the 742 zones the network declares, so 89.7% of its demand reaches no destination at free flow, far above
the tolerance any reported total could absorb.

`python reproduction/tools/corpus_verdicts.py` runs the same gate over every network the collection offers and prints
the verdict with the values behind it. Run it with `--check` to have it exit non-zero wherever the gate
and the declared corpus disagree. As of the last run three networks -- Berlin-Center, Philadelphia and
chicago-regional -- are admitted by the gate and absent from the corpus, which is a scope decision made by
omission rather than by a stated criterion. That is recorded here rather than hidden: either a criterion
that excludes them belongs in this file, or they belong in the corpus.
