"""Stochastic user equilibrium with endogenous efficient supports.

The efficient support -- the links along which the cost to a destination strictly decreases -- is usually
taken as given. Under congestion it is not: it is a function of the flow, and where two routes tie it is
set-valued. This package solves the equilibrium that follows and certifies the answer.

    from endogenous_sue import corpus, solve, certify

    net, od = corpus.load("SiouxFalls")
    equilibrium = solve(net, od, mu=1.0)
    certificate = certify(net, od, 1.0, equilibrium.x)

Modules
-------
``config``          every setting, tolerance and corpus constant
``tntp``            reading benchmark network files
``corpus``          locating and loading the benchmark networks
``network``         per-network arrays and adjacency
``costs``           the link cost function and its derivative
``shortestpath``    cost-to-destination potentials
``subnetwork``      which links are efficient, and in what order
``loading``         recursive-logit loading on an acyclic support
``equilibrium``     the two-phase solver and the inclusion scheme
``certificate``     testing a returned flow against the definition
``witness``         the five-node network on which no strict equilibrium exists
``tied``            the set-valued case: contested links, faces, and their certificates
"""
from __future__ import annotations

__version__ = "1.0.0"

from . import config, corpus, costs, tntp, witness
from .certificate import (
                          Certificate,
                          admissibility_violations,
                          certify,
                          conservation_error,
                          reload_on_own_support,
                          supports_from_potential,
)
from .config import DEFAULTS, Settings
from .costs import bpr_cost, bpr_cost_prime
from .equilibrium import (
                          Equilibrium,
                          InclusionResult,
                          Phase1Result,
                          SupportRule,
                          polish,
                          solve,
                          solve_inclusion,
                          solve_phase1,
)
from .loading import (
                          LoadingReport,
                          absorbing,
                          load,
                          load_destination,
                          load_parallel,
                          load_prepared,
                          load_with_masks,
                          prepare_masks,
)
from .network import NetworkArrays, Topology, topology_of, unpack
from .shortestpath import potentials
from .subnetwork import active_for_dest, cycle_links, isolation_margin, kahn_order
from .tntp import parse_net, parse_trips, read_network

__all__ = [
    "__version__",
    "config", "corpus", "costs", "tntp", "witness",
    "Settings", "DEFAULTS",
    "NetworkArrays", "Topology", "topology_of", "unpack",
    "parse_net", "parse_trips", "read_network",
    "bpr_cost", "bpr_cost_prime",
    "potentials",
    "active_for_dest", "kahn_order", "cycle_links", "isolation_margin",
    "LoadingReport", "load", "load_parallel", "load_destination", "load_with_masks",
    "load_prepared", "prepare_masks", "absorbing",
    "SupportRule", "Phase1Result", "Equilibrium", "InclusionResult",
    "solve", "solve_phase1", "solve_inclusion", "polish",
    "Certificate", "certify", "conservation_error", "admissibility_violations",
    "supports_from_potential", "reload_on_own_support",
]
