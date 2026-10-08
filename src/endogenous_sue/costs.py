"""Separable BPR link cost and its derivative.

The cost of link ``e`` carrying flow ``x_e`` is

    t_e(x_e) = fft_e (1 + b_e (x_e / cap_e) ** power_e).

The load ratio is clamped before the exponentiation. Clamping after it does not help: the overflow has
already produced an infinity, and under automatic differentiation a value repaired afterwards still
carries a non-finite derivative. The clamp level is set from the float64 exponent range rather than
chosen, because above it the cost is constant in flow and the strict monotonicity the theory assumes
fails; the level is therefore the weakest guard the arithmetic admits, not a modelling choice.
"""
from __future__ import annotations

import numpy as np

from .config import FLOAT64_EXPONENT_HEADROOM, MIN_CAPACITY

__all__ = ["bpr_cost", "bpr_cost_prime", "ratio_cap"]


def ratio_cap(power: np.ndarray) -> np.ndarray:
    """Largest load ratio admitted into ``ratio ** power`` while the result stays representable."""
    power = np.maximum(np.asarray(power, dtype=float), 1e-12)
    return 10.0 ** (FLOAT64_EXPONENT_HEADROOM / power)


def bpr_cost(x: np.ndarray, fft: np.ndarray, cap: np.ndarray,
             b: np.ndarray, power: np.ndarray) -> np.ndarray:
    """Link cost at flow ``x``."""
    ratio = np.clip(x / np.clip(cap, MIN_CAPACITY, None), 0.0, ratio_cap(power))
    return fft * (1.0 + b * ratio ** power)


def bpr_cost_prime(x: np.ndarray, fft: np.ndarray, cap: np.ndarray,
                   b: np.ndarray, power: np.ndarray) -> np.ndarray:
    """Derivative of :func:`bpr_cost` with respect to flow.

    Zero on a clamped link, and zero at ``x = 0`` whenever ``power > 1``. Both are why no positive lower
    bound on the derivative is assumed anywhere in this package.
    """
    capped = np.clip(cap, MIN_CAPACITY, None)
    limit = ratio_cap(power)
    ratio = np.clip(x / capped, 0.0, limit)
    with np.errstate(invalid="ignore", divide="ignore"):
        derivative = fft * b * power * np.where(ratio > 0.0, ratio ** (power - 1.0), 0.0) / capped
    return np.where(ratio < limit, np.nan_to_num(derivative), 0.0)
