"""The tied face: what happens where the efficient support is set-valued.

At a tie the strict and closed efficient sets differ, so the support is not a single set and no
single-valued assignment maps the cost to a flow. The equilibrium is then a mixture over admissible
supports, and this subpackage finds one and certifies it.

The face is parameterised by tied group, a group being a tied edge of which a support admits exactly one
orientation. Where the efficiency gap does not depend on the destination the search collapses to one
weight per contested link, and the collapse is checked at the point returned rather than assumed over the
face: a class whose lifted gaps disagree is split and re-solved. Correctness never rests on the collapse
being globally valid.
"""
from .contested import contested_pairs
from .solver import solve_face_admissible, solve_tied

__all__ = ["contested_pairs", "solve_tied", "solve_face_admissible"]
