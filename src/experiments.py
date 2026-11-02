"""
Project 7 - Member 4: Experiments and Visualization
Compares the composite Trapezoidal rule and composite Simpson's rule.

Run with:   python experiments.py
Needs:      numpy, matplotlib  (pip install numpy matplotlib)
Files needed in the same folder: trapezoidal.py, simpson.py
Output:     tables printed on screen + saved in the 'results' folder
            (results/tables.txt and several .png plots)
"""

import math
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")          # lets us save plots without opening a window
import matplotlib.pyplot as plt

from trapezoidal import composite_trapezoidal
from simpson import composite_simpson

# Errors smaller than this are just floating-point rounding noise,
# so we do not trust a convergence order computed from them.
ERROR_FLOOR = 1e-13

# All the n values are even, because Simpson's rule needs an even n.
N_VALUES = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

# Everything we print is also stored here and saved to a file at the end.
report_lines = []


def show(text=""):
    """Print a line and also remember it for the results file."""
    print(text)
    report_lines.append(text)


# ----------------------------------------------------------------------
# Small helper functions
# ----------------------------------------------------------------------
def absolute_error(approx, exact):
    return abs(approx - exact)


def observed_order(h1, e1, h2, e2):
    """Estimate p from  error ~ C * h^p  using two step sizes.
    Returns None when the error is too small to be meaningful."""
    if e1 < ERROR_FLOOR or e2 < ERROR_FLOOR:
        return None
    return math.log(e1 / e2) / math.log(h1 / h2)


def run_experiment(f, a, b, exact, n_values):
    """Run both methods for every n and return a list of result rows."""
    rows = []
    for n in n_values:
        h = (b - a) / n
        trap = composite_trapezoidal(f, a, b, n)
        simp = composite_simpson(f, a, b, n)
        rows.append({
            "n": n,
            "h": h,
            "trap_err": absolute_error(trap, exact),
            "simp_err": absolute_error(simp, exact),
        })

    # Orders need the previous row, so the first row has no order.
    for i in range(len(rows)):
        if i == 0:
            rows[i]["trap_order"] = None
            rows[i]["simp_order"] = None
        else:
            prev = rows[i - 1]
            cur = rows[i]
            cur["trap_order"] = observed_order(prev["h"], prev["trap_err"],
                                               cur["h"], cur["trap_err"])
            cur["simp_order"] = observed_order(prev["h"], prev["simp_err"],
                                               cur["h"], cur["simp_err"])
    return rows


def order_text(p):
    if p is None:
        return "-"
    return f"{p:.3f}"


def print_table(rows):
    show(f"{'n':>6} {'h':>12} {'Trap error':>14} {'Trap order':>11} "
         f"{'Simp error':>14} {'Simp order':>11}")
    show("-" * 74)
    for r in rows:
        show(f"{r['n']:>6} {r['h']:>12.6f} {r['trap_err']:>14.3e} "
             f"{order_text(r['trap_order']):>11} {r['simp_err']:>14.3e} "
             f"{order_text(r['simp_order']):>11}")
    show()


def make_plot(rows, title, filename):
    """Log-log plot of error against step size h, with reference lines."""
    h_list = [r["h"] for r in rows]
    # log plots cannot show 0, so replace zero errors by a tiny number
    trap_err = [max(r["trap_err"], 1e-17) for r in rows]
    simp_err = [max(r["simp_err"], 1e-17) for r in rows]

    plt.figure(figsize=(7, 5))
    plt.loglog(h_list, trap_err, "o-", label="Trapezoidal")
    plt.loglog(h_list, simp_err, "s-", label="Simpson")

    # reference lines: slope 2 and slope 4, starting at the largest h
    h0 = h_list[0]
    plt.loglog(h_list, [trap_err[0] * (h / h0) ** 2 for h in h_list],
               "--", color="gray", label="slope 2 ($h^2$)")
    plt.loglog(h_list, [simp_err[0] * (h / h0) ** 4 for h in h_list],
               ":", color="black", label="slope 4 ($h^4$)")

    plt.xlabel("step size h")
    plt.ylabel("absolute error")
    plt.title(title)
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("results", filename), dpi=150)
    plt.close()


# ----------------------------------------------------------------------
# Experiment 1 + 3: smooth functions and observed convergence order
# ----------------------------------------------------------------------
def experiment_smooth():
    show("=" * 74)
    show("EXPERIMENT 1 & 3: SMOOTH FUNCTIONS AND OBSERVED CONVERGENCE ORDER")
    show("=" * 74)

    tests = [
        ("exp(x) on [0, 1]", math.exp, 0.0, 1.0, math.e - 1,
         "smooth_exp.png"),
        ("sin(x) on [0, pi]", math.sin, 0.0, math.pi, 2.0,
         "smooth_sin.png"),
    ]
    for name, f, a, b, exact, png in tests:
        show(f"Function: {name}      exact value = {exact:.12f}")
        rows = run_experiment(f, a, b, exact, N_VALUES)
        print_table(rows)
        make_plot(rows, "Error vs step size: " + name, png)


# ----------------------------------------------------------------------
# Experiment 2: polynomial exactness
# ----------------------------------------------------------------------
def experiment_polynomials():
    show("=" * 74)
    show("EXPERIMENT 2: POLYNOMIAL EXACTNESS  (f(x) = x^k on [0, 1], n = 4)")
    show("=" * 74)
    show("Exact integral of x^k on [0,1] is 1/(k+1).")
    show("'EXACT' means the error is only floating-point rounding (< 1e-13).")
    show()
    show(f"{'degree k':>9} {'Trap error':>14} {'Trap':>7} "
         f"{'Simp error':>14} {'Simp':>7}")
    show("-" * 56)

    n = 4
    for k in range(0, 6):
        f = lambda x, k=k: x ** k
        exact = 1.0 / (k + 1)
        trap_err = absolute_error(composite_trapezoidal(f, 0.0, 1.0, n), exact)
        simp_err = absolute_error(composite_simpson(f, 0.0, 1.0, n), exact)
        trap_word = "EXACT" if trap_err < ERROR_FLOOR else "no"
        simp_word = "EXACT" if simp_err < ERROR_FLOOR else "no"
        show(f"{k:>9} {trap_err:>14.3e} {trap_word:>7} "
             f"{simp_err:>14.3e} {simp_word:>7}")
    show()
    show("Expected: Trapezoidal is exact for degree <= 1,")
    show("          Simpson is exact for degree <= 3.")
    show()


# ----------------------------------------------------------------------
# Experiment 4: more challenging integrands
# ----------------------------------------------------------------------
def experiment_challenging():
    show("=" * 74)
    show("EXPERIMENT 4: MORE CHALLENGING INTEGRANDS")
    show("=" * 74)

    tests = [
        ("sin(20x) on [0, 1]  (rapid oscillation)",
         lambda x: math.sin(20 * x), 0.0, 1.0,
         (1 - math.cos(20)) / 20, "hard_sin20x.png"),
        ("1/(1+25x^2) on [-1, 1]  (sharp peak at x = 0)",
         lambda x: 1 / (1 + 25 * x * x), -1.0, 1.0,
         2 * math.atan(5) / 5, "hard_peak.png"),
        ("sqrt(x) on [0, 1]  (infinite slope at x = 0)",
         math.sqrt, 0.0, 1.0, 2.0 / 3.0, "hard_sqrt.png"),
    ]
    for name, f, a, b, exact, png in tests:
        show(f"Function: {name}      exact value = {exact:.12f}")
        rows = run_experiment(f, a, b, exact, N_VALUES)
        print_table(rows)
        make_plot(rows, "Error vs step size: " + name, png)


# ----------------------------------------------------------------------
# Extra: one plot comparing all functions at the same n
# ----------------------------------------------------------------------
def summary_plot():
    """Bar-style comparison: error of both methods at n = 16 for each test."""
    tests = [
        ("exp(x)", math.exp, 0.0, 1.0, math.e - 1),
        ("sin(x)", math.sin, 0.0, math.pi, 2.0),
        ("sin(20x)", lambda x: math.sin(20 * x), 0.0, 1.0,
         (1 - math.cos(20)) / 20),
        ("1/(1+25x^2)", lambda x: 1 / (1 + 25 * x * x), -1.0, 1.0,
         2 * math.atan(5) / 5),
        ("sqrt(x)", math.sqrt, 0.0, 1.0, 2.0 / 3.0),
    ]
    n = 16
    names, trap_errs, simp_errs = [], [], []
    for name, f, a, b, exact in tests:
        names.append(name)
        trap_errs.append(absolute_error(composite_trapezoidal(f, a, b, n), exact))
        simp_errs.append(absolute_error(composite_simpson(f, a, b, n), exact))

    positions = list(range(len(names)))
    width = 0.38
    plt.figure(figsize=(8, 5))
    plt.bar([p - width / 2 for p in positions], trap_errs, width,
            label="Trapezoidal")
    plt.bar([p + width / 2 for p in positions], simp_errs, width,
            label="Simpson")
    plt.yscale("log")
    plt.xticks(positions, names)
    plt.ylabel("absolute error (log scale)")
    plt.title(f"Both methods on every test function (n = {n})")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join("results", "summary_n16.png"), dpi=150)
    plt.close()


# ----------------------------------------------------------------------
# Main program
# ----------------------------------------------------------------------
if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)

    experiment_smooth()
    experiment_polynomials()
    experiment_challenging()
    summary_plot()

    with open(os.path.join("results", "tables.txt"), "w") as file:
        file.write("\n".join(report_lines))
    print("Done. Tables and plots are saved in the 'results' folder.")
