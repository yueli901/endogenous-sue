"""Optional numba acceleration.

Every compiled kernel in this package has a pure-Python counterpart with identical semantics, so the
package imports and runs without numba installed. Only the speed changes.
"""
from __future__ import annotations

try:
    from numba import njit, prange
    HAVE_NUMBA = True
except ImportError:
    HAVE_NUMBA = False
    prange = range

    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def wrap(fn):
            return fn
        return wrap

__all__ = ["njit", "prange", "HAVE_NUMBA"]
