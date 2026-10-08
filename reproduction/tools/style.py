"""Shared presentation: network names, number formatting, and figure style.

Presentation only. Corpus membership, the dispersion grid and the exclusion rule are settings and live in
:mod:`endogenous_sue.config`; duplicating them here is how a table comes to disagree with the data.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
sys.path.insert(0, str(_ROOT / "src"))

from endogenous_sue.config import DISPERSIONS, EXCLUDED_NETWORKS  # noqa: E402,F401

# Display names. A network absent here prints under its own name, which is the safe default: an entry
# that guesses from a prefix will eventually label one network as another.
DISPLAY_NAME = {
    "SiouxFalls": "Sioux Falls",
    "Eastern-Massachusetts": "Eastern Massachusetts",
    "Berlin-Mitte-Prenzlauerberg-Friedrichshain-Center": "Berlin-M.-P.-F.-Center",
    "Braess-Example": "Braess example",
    "chicago-regional": "Chicago-Regional",
}

PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"]

RC_PARAMS = {
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.2,
    "pdf.fonttype": 42,
}


def display(network: str) -> str:
    return DISPLAY_NAME.get(network, network)


def sci(value, digits: int = 1) -> str:
    """Scientific notation as LaTeX, so a table never mixes ``1.81e-05`` with a typeset power of ten."""
    if value is None:
        return "---"
    if value == 0:
        return "$0$"
    exponent = int(np.floor(np.log10(abs(value))))
    mantissa = value / 10.0 ** exponent
    if round(mantissa, digits) >= 10:
        mantissa, exponent = mantissa / 10.0, exponent + 1
    return f"${mantissa:.{digits}f}\\times10^{{{exponent}}}$"


def num(value, digits: int = 1) -> str:
    """A number for a LaTeX column, in one notation throughout."""
    if value is None:
        return "---"
    compact = f"{value:.3g}"
    if value != 0 and (abs(value) < 1e-3 or "e" in compact.lower()):
        return sci(value, digits)
    return f"${compact}$"


def span(values, digits: int = 1, tol: float = 5e-3) -> str:
    """A single value where the range is negligible, otherwise the range."""
    if not values:
        return "---"
    low, high = min(values), max(values)
    return num(low, digits) if abs(high - low) < tol else f"{num(low, digits)}--{num(high, digits)}"
