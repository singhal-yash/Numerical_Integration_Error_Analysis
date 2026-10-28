import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from error_analysis import (
    absolute_error,
    compute_errors,
    convergence_orders,
    convergence_study,
    format_table,
    observed_order,
)
from trapezoidal import composite_trapezoidal


# The reference_simpson function helps in calculating the integral using Simpson's rule.
# It checks if n is even and then calculates the approximate answer.
def reference_simpson(f, a, b, n):
    if n % 2 != 0:
        raise ValueError("n must be even.")
    h = (b - a) / n
    total = f(a) + f(b)
    for i in range(1, n):
        total += (4 if i % 2 else 2) * f(a + i * h)
    return h / 3 * total

# This test helps to check whether the absolute_error function is calculating
# the correct difference between the approximate and actual values.
def test_absolute_error_basic():
    assert absolute_error(2.5, 2.0) == 0.5
    assert absolute_error(1.5, 2.0) == 0.5
    assert absolute_error(3, 3) == 0


# This test checks whether the absolute_error function also works properly
# when we give numbers using NumPy data types.
def test_absolute_error_accepts_numpy_scalars():
    np = pytest.importorskip("numpy")
    assert absolute_error(np.float64(2.5), np.float32(2.0)) == 0.5

# This test helps to check whether the function rejects values which are
# not real numbers, such as text, None, complex numbers or boolean values.
@pytest.mark.parametrize("bad", ["1", None, [1], 1j, True])
def test_absolute_error_rejects_non_real(bad):
    with pytest.raises(TypeError):
        absolute_error(bad, 1.0)
    with pytest.raises(TypeError):
        absolute_error(1.0, bad)


# This test checks whether the function rejects values like infinity and NaN,
# since these are not valid finite numbers for calculating the error.
@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_absolute_error_rejects_non_finite(bad):
    with pytest.raises(ValueError):
        absolute_error(bad, 1.0)
    with pytest.raises(ValueError):
        absolute_error(1.0, bad)

# This test helps to check whether observed_order calculates the correct
# convergence order when the error follows a known power of the step size.
@pytest.mark.parametrize("p", [1, 2, 3, 4, 0.5])
def test_observed_order_recovers_known_power(p):
    h1, h2 = 0.1, 0.05
    assert observed_order(h1, 3.0 * h1 ** p, h2, 3.0 * h2 ** p) == pytest.approx(p)

# This test checks whether we get the same convergence order even if we
# change the order in which the two step sizes and errors are given.
def test_observed_order_argument_order_is_symmetric():
    assert observed_order(0.1, 1e-2, 0.05, 2.5e-3) == pytest.approx(
        observed_order(0.05, 2.5e-3, 0.1, 1e-2))

# This test checks whether the function returns None when one or both errors
# are zero, because the convergence order cannot be calculated in that case.
def test_observed_order_zero_error_is_not_meaningful():
    assert observed_order(0.1, 0.0, 0.05, 1e-3) is None
    assert observed_order(0.1, 1e-3, 0.05, 0.0) is None
    assert observed_order(0.1, 0, 0.05, 0) is None

# This test helps to check whether the function raises an error for invalid
# values, such as equal step sizes, negative errors, zero or infinity.
def test_observed_order_invalid_values():
    with pytest.raises(ValueError):
        observed_order(0.1, 1e-3, 0.1, 1e-4)        
    with pytest.raises(ValueError):
        observed_order(0.0, 1e-3, 0.05, 1e-4)       
    with pytest.raises(ValueError):
        observed_order(-0.1, 1e-3, 0.05, 1e-4)      
    with pytest.raises(ValueError):
        observed_order(0.1, -1e-3, 0.05, 1e-4)      
    with pytest.raises(ValueError):
        observed_order(0.1, math.nan, 0.05, 1e-4)
    with pytest.raises(ValueError):
        observed_order(math.inf, 1e-3, 0.05, 1e-4)

# This test checks whether the function rejects incorrect input types,
# such as text, boolean values or None.
def test_observed_order_invalid_types():
    with pytest.raises(TypeError):
        observed_order("0.1", 1e-3, 0.05, 1e-4)
    with pytest.raises(TypeError):
        observed_order(0.1, True, 0.05, 1e-4)
    with pytest.raises(TypeError):
        observed_order(0.1, 1e-3, None, 1e-4)

# This test helps to check whether compute_errors gives the correct values
# of n, step size and absolute error for different subdivisions.
def test_compute_errors_rows():
    rows = compute_errors(composite_trapezoidal, math.sin, 0, math.pi, 2.0, [4, 8])
    assert [r["n"] for r in rows] == [4, 8]
    assert rows[0]["h"] == pytest.approx(math.pi / 4)
    assert rows[1]["h"] == pytest.approx(math.pi / 8)
    for r in rows:
        assert r["error"] == pytest.approx(abs(r["approx"] - 2.0))
    assert rows[1]["error"] < rows[0]["error"]

# This test checks whether compute_errors accepts a generator as input
# and keeps the subdivision counts in the same order as we give them.
def test_compute_errors_keeps_given_order_and_accepts_any_iterable():
    rows = compute_errors(composite_trapezoidal, math.sin, 0, math.pi, 2.0,
                          (n for n in [16, 4, 8]))
    assert [r["n"] for r in rows] == [16, 4, 8]

# This test checks whether compute_errors accepts integer values from NumPy
# and returns the subdivision counts as normal Python integers.
def test_compute_errors_accepts_numpy_ints():
    np = pytest.importorskip("numpy")
    rows = compute_errors(composite_trapezoidal, math.sin, 0, math.pi, 2.0,
                          np.array([4, 8, 16]))
    assert [r["n"] for r in rows] == [4, 8, 16]
    assert all(type(r["n"]) is int for r in rows)

# This test helps to check whether the step size stays positive even when
# we reverse the integration limits, and the calculated error stays small.
def test_compute_errors_reversed_interval_uses_positive_h():
    rows = compute_errors(composite_trapezoidal, math.sin, math.pi, 0, -2.0, [8])
    assert rows[0]["h"] > 0
    assert rows[0]["error"] < 0.05

# This test checks whether compute_errors passes along an error raised by
# the numerical method, such as giving an odd n to Simpson's rule.
def test_compute_errors_propagates_method_errors():
    with pytest.raises(ValueError):
        compute_errors(reference_simpson, math.sin, 0, 1, 1 - math.cos(1), [3])

# This test helps to check whether compute_errors rejects invalid n values,
# such as empty inputs, zero, negative, repeated or non-integer values.
@pytest.mark.parametrize("n_values, exc", [
    ([], ValueError),
    ([0, 2], ValueError),
    ([-2, 2], ValueError),
    ([2, 2], ValueError),
    ([2.0, 4], TypeError),
    ([True, 4], TypeError),
    (["2", 4], TypeError),
])
def test_compute_errors_rejects_bad_n_values(n_values, exc):
    with pytest.raises(exc):
        compute_errors(composite_trapezoidal, math.sin, 0, 1, 1.0, n_values)

# This test checks whether compute_errors rejects incorrect arguments,
# such as invalid functions, text inputs, infinity or equal interval limits.
def test_compute_errors_rejects_bad_arguments():
    ok = dict(method=composite_trapezoidal, f=math.sin, a=0, b=1, exact=0.5,
              n_values=[2, 4])
    with pytest.raises(TypeError):
        compute_errors(**{**ok, "method": 5})
    with pytest.raises(TypeError):
        compute_errors(**{**ok, "f": "sin"})
    with pytest.raises(TypeError):
        compute_errors(**{**ok, "a": "0"})
    with pytest.raises(TypeError):
        compute_errors(**{**ok, "exact": None})
    with pytest.raises(ValueError):
        compute_errors(**{**ok, "b": math.inf})
    with pytest.raises(ValueError):
        compute_errors(**{**ok, "exact": math.nan})
    with pytest.raises(ValueError):
        compute_errors(**{**ok, "b": 0})           

# This test checks whether compute_errors raises an error when n_values
# is just a single number instead of a collection of subdivision counts.
def test_compute_errors_n_values_not_iterable():
    with pytest.raises(TypeError):
        compute_errors(composite_trapezoidal, math.sin, 0, 1, 1.0, 4)

# This test helps to check whether the trapezoidal rule gives a convergence
# order close to 2 when we increase the number of subdivisions.
def test_trapezoidal_observed_order_is_two():
    rows = convergence_study(composite_trapezoidal, math.sin, 0, math.pi, 2.0,
                             [4, 8, 16, 32, 64, 128])
    orders = [r["order"] for r in rows[1:]]
    assert rows[0]["order"] is None
    assert all(o == pytest.approx(2.0, abs=0.02) for o in orders)

# This test checks whether Simpson's rule gives a convergence order close
# to 4, which is what we expect for this smooth function.
def test_simpson_observed_order_is_four():
    rows = convergence_study(reference_simpson, math.exp, 0, 1, math.e - 1,
                             [2, 4, 8, 16, 32])
    orders = [r["order"] for r in rows[1:]]
    assert all(o == pytest.approx(4.0, abs=0.05) for o in orders)

# This test helps to check whether the Simpson's rule from our actual module
# also gives an order close to 4, if that module is available.
def test_real_simpson_module_if_available():
    np = pytest.importorskip("numpy")
    simpson = pytest.importorskip("simpson")
    rows = convergence_study(simpson.composite_simpson, np.exp, 0, 1,
                             math.e - 1, [2, 4, 8, 16, 32])
    assert all(r["order"] == pytest.approx(4.0, abs=0.05) for r in rows[1:])

# This test checks whether Simpson's rule gives almost zero error for a
# cubic polynomial and does not report a misleading convergence order.
def test_polynomial_exactness_is_not_given_an_order():
    rows = convergence_study(reference_simpson, lambda x: x ** 3, 0, 2, 4.0,
                             [2, 4, 8, 16])
    assert all(r["error"] < 1e-13 for r in rows)
    assert all(r["order"] is None for r in rows)

# This test checks whether the trapezoidal rule reports no convergence order
# when it already gives the exact answer for a linear function.
def test_trapezoidal_exact_for_linear_gives_no_order():
    rows = convergence_study(composite_trapezoidal, lambda x: 3 * x + 1, 0, 2, 8.0,
                             [2, 4, 8])
    assert all(r["order"] is None for r in rows)

# This test helps to check whether the function stops reporting the order
# when the error becomes extremely small and may not be reliable.
def test_roundoff_floor_stops_order_reporting():
    exact = 2 * math.pi * 1.2660658777520082
    rows = convergence_study(composite_trapezoidal, lambda x: math.exp(math.cos(x)),
                             0, 2 * math.pi, exact, [4, 8, 16, 32, 64])
    assert rows[-1]["error"] < 1e-13
    assert rows[-1]["order"] is None

# This test checks whether setting the error_floor to zero allows us to
# calculate the convergence order as long as the errors are not zero.
def test_floor_zero_reports_orders_for_nonzero_errors():
    rows = convergence_study(reference_simpson, math.sin, 0, math.pi, 2.0,
                             [2, 4, 8, 16, 32, 64, 128, 256], error_floor=0)
    assert rows[-1]["order"] == pytest.approx(4.0, abs=0.1)

# This test checks whether convergence_study arranges the n values in
# increasing order and leaves the first convergence order as None.
def test_study_sorts_n_and_first_order_is_none():
    rows = convergence_study(composite_trapezoidal, math.sin, 0, math.pi, 2.0,
                             [32, 8, 16])
    assert [r["n"] for r in rows] == [8, 16, 32]
    assert rows[0]["order"] is None
    assert rows[1]["order"] == pytest.approx(2.0, abs=0.05)

# This test helps to check whether convergence_study raises an error
# when we give only one subdivision count instead of at least two.
def test_study_needs_two_subdivisions():
    with pytest.raises(ValueError):
        convergence_study(composite_trapezoidal, math.sin, 0, math.pi, 2.0, [4])

# This test checks whether the trapezoidal rule gives a lower convergence
# order, close to 1.5, when we use the square root function.
def test_study_non_smooth_function_gives_lower_order():
    rows = convergence_study(composite_trapezoidal, math.sqrt, 0, 1, 2 / 3,
                             [16, 32, 64, 128, 256])
    assert rows[-1]["order"] == pytest.approx(1.5, abs=0.05)


# This test helps to check the error results for a function with a sharp
# corner, and makes sure the calculated errors are positive.
def test_study_kink_function_order_between_one_and_two():
    f = lambda x: abs(x - 1 / 3)
    exact = (1 / 3) ** 2 / 2 + (2 / 3) ** 2 / 2
    rows = convergence_study(composite_trapezoidal, f, 0, 1, exact,
                             [10, 20, 40, 80, 160, 320])
    assert all(r["error"] > 0 for r in rows)

# This test checks whether convergence_orders sorts the results by step
# size before calculating the convergence orders.
def test_convergence_orders_sorts_by_step_size():
    rows = [
        {"n": 8, "h": 0.125, "error": 0.125 ** 2},
        {"n": 2, "h": 0.5, "error": 0.5 ** 2},
        {"n": 4, "h": 0.25, "error": 0.25 ** 2},
    ]
    orders = convergence_orders(rows)
    assert len(orders) == 2
    assert all(o == pytest.approx(2.0) for o in orders)

# This test checks whether convergence_orders returns None for a pair
# of results when one of the errors is below the given error limit.
def test_convergence_orders_marks_floor_rows_none():
    rows = [
        {"h": 0.4, "error": 1e-3},
        {"h": 0.2, "error": 6e-5},
        {"h": 0.1, "error": 1e-16},
    ]
    orders = convergence_orders(rows)
    assert orders[0] == pytest.approx(math.log2(1e-3 / 6e-5))
    assert orders[1] is None

# This test helps to check whether convergence_orders rejects incorrect
# results, missing values, invalid step sizes or invalid error limits.
def test_convergence_orders_validation():
    good = [{"h": 0.2, "error": 1e-3}, {"h": 0.1, "error": 2e-4}]
    with pytest.raises(ValueError):
        convergence_orders(good[:1])
    with pytest.raises(ValueError):
        convergence_orders([])
    with pytest.raises(TypeError):
        convergence_orders("abc")
    with pytest.raises(TypeError):
        convergence_orders([1, 2])
    with pytest.raises(ValueError):
        convergence_orders([{"h": 0.2}, {"h": 0.1, "error": 1}])
    with pytest.raises(ValueError):
        convergence_orders(good, error_floor=-1)
    with pytest.raises(TypeError):
        convergence_orders(good, error_floor="1e-14")
    with pytest.raises(ValueError):
        convergence_orders([{"h": 0.2, "error": -1}, {"h": 0.1, "error": 1}])
    with pytest.raises(ValueError):
        convergence_orders([{"h": 0.2, "error": 1}, {"h": 0.2, "error": 1}])
    with pytest.raises(ValueError):
        convergence_orders([{"h": 0, "error": 1}, {"h": 0.2, "error": 1}])


# This test checks whether convergence_orders can also accept results
# given as a tuple instead of only using a list.
def test_convergence_orders_accepts_tuple():
    rows = ({"h": 0.2, "error": 4e-2}, {"h": 0.1, "error": 1e-2})
    assert convergence_orders(rows)[0] == pytest.approx(2.0)

# This test helps to check whether format_table creates the expected
# table with the correct headings, rows and convergence order values.
def test_format_table_contents():
    rows = convergence_study(composite_trapezoidal, math.sin, 0, math.pi, 2.0,
                             [4, 8])
    text = format_table(rows)
    lines = text.splitlines()
    assert len(lines) == 3
    assert "order" in lines[0]
    assert lines[1].rstrip().endswith("-")
    assert "2.0" in lines[2].split()[-1]

# This test checks whether format_table still returns the heading row
# when we give it an empty list of results.
def test_format_table_empty():
    assert len(format_table([]).splitlines()) == 1