import math
from numbers import Integral, Real

# The _check_real function helps to if the given input value is a real number or not. 
# It will reject any sort of text, or boolean values like True or False, infinity or NaN values.
def _check_real(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number.")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number.")

# The _check_positive function helps to check whether given value is positive nuumber or not.
def _check_positive(value, name):
    _check_real(value, name)
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero.")

# In this absolute_error function, we are calculating the absolute error, which means how 
# is our calculated answer with respect to the actual correct answer.
def absolute_error(approx, exact):
    _check_real(approx, "approx")
    _check_real(exact, "exact")
    return abs(approx - exact)

# The observed_order function helps to measure the fact that how quickly the error rate
# rate decreases when we use a smaller step size. It further helps in understanding the speed
# of the convergence of our given numeric method.
def observed_order(h1, error1, h2, error2):
    _check_positive(h1, "h1")
    _check_positive(h2, "h2")
    _check_real(error1, "error1")
    _check_real(error2, "error2")
    if h1 == h2:
        raise ValueError("h1 and h2 must be different.")
    if error1 < 0 or error2 < 0:
        raise ValueError("errors must be non-negative.")

    if error1 == 0 or error2 == 0:
        return None
    return math.log(error1 / error2) / math.log(h1 / h2)

# The compute_errors functions helps in running the given numerical method for different
# numbers of subdivisions which is defined as n, and it calculates the approximation using
# the step size and the absolute error for each one of those n values. 
def compute_errors(method, f, a, b, exact, n_values):
    if not callable(method):
        raise TypeError("method must be a callable function.")
    if not callable(f):
        raise TypeError("f must be a callable function.")
    _check_real(a, "a")
    _check_real(b, "b")
    _check_real(exact, "exact")
    if a == b:
        raise ValueError("a and b must be different.")

    n_list = list(n_values)
    if not n_list:
        raise ValueError("n_values must not be empty.")
    for n in n_list:
        if isinstance(n, bool) or not isinstance(n, Integral):
            raise TypeError("every n in n_values must be an integer.")
        if n <= 0:
            raise ValueError("every n in n_values must be greater than zero.")
    if len(set(n_list)) != len(n_list):
        raise ValueError("n_values must not contain duplicates.")

    results = []
    for n in n_list:
        n = int(n)
        approx = float(method(f, a, b, n))
        results.append({
            "n": n,
            "h": abs(b - a) / n,
            "approx": approx,
            "error": absolute_error(approx, exact),
        })
    return results

# The convergence_orders function helps to compare the errors from consecutive step sizes
# to find the convergence order. It will particularly skips those calculations when the 
# errors become too small, this might lead to errors becoming unrealible.
def convergence_orders(results, error_floor=1e-14):
    if not isinstance(results, (list, tuple)) or not all(
            isinstance(r, dict) for r in results):
        raise TypeError("results must be a list of dicts from compute_errors.")
    if len(results) < 2:
        raise ValueError("at least two results are needed to estimate an order.")
    _check_real(error_floor, "error_floor")
    if error_floor < 0:
        raise ValueError("error_floor must be non-negative.")
    for r in results:
        if "h" not in r or "error" not in r:
            raise ValueError("each result must contain 'h' and 'error'.")
        _check_positive(r["h"], "h")
        _check_real(r["error"], "error")
        if r["error"] < 0:
            raise ValueError("errors must be non-negative.")

    rows = sorted(results, key=lambda r: r["h"], reverse=True)
    orders = []
    for coarse, fine in zip(rows, rows[1:]):
        if coarse["error"] <= error_floor or fine["error"] <= error_floor:
            orders.append(None)
        else:
            orders.append(observed_order(
                coarse["h"], coarse["error"], fine["h"], fine["error"]))
    return orders

# The convergence_study function uses the above-mentioned convergence_orders and compute_errors
# function to perform the end-to-end error analysis. It will help in calculating the 
# approximations, errors, and convergence orders for a different counts of subdivisions.
def convergence_study(method, f, a, b, exact, n_values, error_floor=1e-14):
    results = compute_errors(method, f, a, b, exact, n_values)
    if len(results) < 2:
        raise ValueError("at least two subdivision counts are needed.")
    rows = sorted(results, key=lambda r: r["n"])
    orders = convergence_orders(rows, error_floor)
    rows[0]["order"] = None
    for row, order in zip(rows[1:], orders):
        row["order"] = order
    return rows

# This format_table function is simple, it takes in the calculated results from the 
# converegence_study function and formats them into a properly, neat, and clean table.
def format_table(rows):
    lines = [f"{'n':>8} {'h':>12} {'approximation':>20} {'abs error':>12} {'order':>8}"]
    for r in rows:
        order = "-" if r.get("order") is None else f"{r['order']:.4f}"
        lines.append(
            f"{r['n']:>8d} {r['h']:>12.6f} {r['approx']:>20.12f} "
            f"{r['error']:>12.3e} {order:>8}")
    return "\n".join(lines)

if __name__ == "__main__":
    from trapezoidal import composite_trapezoidal

    table = convergence_study(
        composite_trapezoidal, math.sin, 0, math.pi, 2.0, [4, 8, 16, 32, 64])
    print("Composite Trapezoidal Rule, sin(x) on [0, pi], exact value 2.0")
    print(format_table(table))

    # Simpson's rule is developed on its own branch, so skip this part
    # (instead of failing) when simpson.py is not present in src/.
    try:
        from simpson import composite_simpson
    except ImportError:
        print("\nsimpson.py not found in src/: skipping Simpson's rule demo.")
    else:
        # Simpson's rule needs an even n; the same n values are used for a
        # direct comparison with the trapezoidal table above.
        table = convergence_study(
            composite_simpson, math.sin, 0, math.pi, 2.0, [4, 8, 16, 32, 64])
        print("\nComposite Simpson's 1/3 Rule, sin(x) on [0, pi], exact value 2.0")
        print(format_table(table))
