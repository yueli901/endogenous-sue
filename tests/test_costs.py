"""The link cost: monotonicity, the derivative, and the representability clamp."""
import numpy as np

from endogenous_sue.config import FLOAT64_EXPONENT_HEADROOM
from endogenous_sue.costs import bpr_cost, bpr_cost_prime, ratio_cap


def test_the_cost_increases_with_flow_below_the_clamp():
    fft, cap = np.array([1.0]), np.array([100.0])
    b, power = np.array([0.15]), np.array([4.0])
    flows = np.linspace(0.0, 500.0, 50)
    costs = [float(bpr_cost(np.array([x]), fft, cap, b, power)[0]) for x in flows]
    assert all(later >= earlier for earlier, later in zip(costs, costs[1:]))


def test_the_clamp_is_set_by_the_exponent_range_and_not_by_a_chosen_level():
    for power in (1.0, 4.0, 16.83):
        limit = float(ratio_cap(np.array([power]))[0])
        assert abs(np.log10(limit) * power - FLOAT64_EXPONENT_HEADROOM) < 1e-6


def test_the_cost_stays_finite_at_an_extreme_load_ratio():
    """Unit capacity with a large exponent is what the excluded networks carry."""
    fft, cap = np.array([1.0]), np.array([1.0])
    b, power = np.array([0.15]), np.array([16.83])
    cost = bpr_cost(np.array([1e6]), fft, cap, b, power)
    assert np.all(np.isfinite(cost))
    assert np.all(np.isfinite(bpr_cost_prime(np.array([1e6]), fft, cap, b, power)))


def test_the_derivative_vanishes_where_the_clamp_binds_and_at_zero_flow():
    fft, cap = np.array([1.0]), np.array([1.0])
    b, power = np.array([0.15]), np.array([4.0])
    assert float(bpr_cost_prime(np.array([0.0]), fft, cap, b, power)[0]) == 0.0
    beyond = float(ratio_cap(power)[0]) * 10.0
    assert float(bpr_cost_prime(np.array([beyond]), fft, cap, b, power)[0]) == 0.0


def test_the_monotonicity_diagnostic_separates_the_two_ways_a_cost_stops_increasing():
    """A zero-cost connector and a clamped link both have a zero derivative and are not the same thing.

    A link with zero free-flow time has a cost identically zero, so its derivative vanishes at every
    flow: that is the connector convention the model already admits and a property of the benchmark
    data. A link with positive free-flow time whose derivative has reached zero has lost the uniqueness
    argument the method relies on. Counting them together would report the second whenever the first is
    present, which on Chicago-Sketch would read as 772 failures where there are none.
    """
    import numpy as np

    from endogenous_sue.network import NetworkArrays
    from endogenous_sue.preflight import cost_is_increasing

    net = dict(tail=np.array([1, 2, 3]), head=np.array([2, 3, 4]),
               fft=np.array([0.0, 1.0, 1.0]), capacity=np.array([100.0, 100.0, 100.0]),
               b=np.array([0.15, 0.15, 0.15]), power=np.array([4.0, 4.0, 4.0]),
               n_nodes=4, n_zones=2)
    arrays = NetworkArrays.from_dict(net)

    # Link 0 carries flow and has no free-flow time; link 1 carries flow and a real cost; link 2 is
    # empty, so it is flat for neither reason and must not be counted at all.
    by_convention, despite, smallest = cost_is_increasing(arrays, np.array([10.0, 10.0, 0.0]))
    assert by_convention == 1, f"the zero-cost connector was not counted: {by_convention}"
    assert despite == 0, f"a link with a real cost was called flat: {despite}"
    assert smallest > 0.0

    # Drive the second link's load ratio far enough below the representable range that its derivative
    # underflows while its free-flow time stays positive. That is the case the second count is for.
    # The ratio is cubed, so it must fall far enough that the cube underflows, not merely
    # the ratio: at 1e-90 over a capacity of 100 the cube is 1e-276 and still representable.
    tiny = np.array([10.0, 1e-120, 0.0])
    by_convention, despite, _ = cost_is_increasing(arrays, tiny)
    assert by_convention == 1
    assert despite == 1, (
        "a positive-cost link whose derivative underflowed was not reported separately; "
        "the two causes are being conflated")
