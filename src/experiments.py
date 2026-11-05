"""
Experiments comparing the composite Trapezoidal rule and Simpson's rule.

Uses compute_errors, convergence_study and format_table from error_analysis.py
(no duplicated error/order code). Run from the project root:

    python src/experiments.py

Tables and plots are saved in the 'results' folder next to 'src'.
"""

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # save plots without opening a window
import matplotlib.pyplot as plt

from error_analysis import compute_errors, convergence_study, format_table
from simpson import composite_simpson
from trapezoidal import composite_trapezoidal

# Output folder is always <project root>/results, whatever the current directory.
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# Same floor as the default in error_analysis.py: errors at or below this
# are rounding noise, so no order is reported for them.
ERROR_FLOOR = 1e-14

# All n are even because Simpson's rule needs an even n.
N_VALUES = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

report_lines = []   # everything printed is also saved to results/tables.txt


def show(text=""):
    print(text)
    report_lines.append(text)


def make_plot(trap_rows, simp_rows, title, filename):
    """Log-log plot of error against step size h, with slope-2 and slope-4 lines."""
    h_list = [r["h"] for r in trap_rows]
    # log axes cannot show 0, so use a tiny value instead of an error of 0
    trap_err = [max(r["error"], 1e-17) for r in trap_rows]
    simp_err = [max(r["error"], 1e-17) for r in simp_rows]

    plt.figure(figsize=(7, 5))
    plt.loglog(h_list, trap_err, "o-", label="Trapezoidal")
    plt.loglog(h_list, simp_err, "s-", label="Simpson")

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
    plt.savefig(RESULTS_DIR / filename, dpi=150)
    plt.close()


def study_function(name, f, a, b, exact, png):
    """Run both methods on one function: print two tables and save one plot."""
    show(f"Function: {name}      exact value = {exact:.12f}")
    trap_rows = convergence_study(composite_trapezoidal, f, a, b, exact,
                                  N_VALUES, ERROR_FLOOR)
    simp_rows = convergence_study(composite_simpson, f, a, b, exact,
                                  N_VALUES, ERROR_FLOOR)
    show("Composite Trapezoidal rule")
    show(format_table(trap_rows))
    show()
    show("Composite Simpson's rule")
    show(format_table(simp_rows))
    show()
    make_plot(trap_rows, simp_rows, "Error vs step size: " + name, png)


# Experiments 1 and 3: smooth functions and observed convergence order
def experiment_smooth():
    show("=" * 74)
    show("EXPERIMENT 1 & 3: SMOOTH FUNCTIONS AND OBSERVED CONVERGENCE ORDER")
    show("=" * 74)
    study_function("exp(x) on [0, 1]", math.exp, 0.0, 1.0, math.e - 1,
                   "smooth_exp.png")
    study_function("sin(x) on [0, pi]", math.sin, 0.0, math.pi, 2.0,
                   "smooth_sin.png")


# Experiment 2: polynomial exactness
def polynomial_value(coeffs, x):
    """Value of c0 + c1*x + c2*x^2 + ... at x (coeffs = [c0, c1, c2, ...])."""
    total = 0.0
    for power in range(len(coeffs)):
        total += coeffs[power] * x ** power
    return total


def polynomial_integral(coeffs, a, b):
    """Exact integral of the polynomial from a to b, found term by term."""
    total = 0.0
    for power in range(len(coeffs)):
        total += coeffs[power] * (b ** (power + 1) - a ** (power + 1)) / (power + 1)
    return total


def is_exact(error, exact):
    """True if the error is only floating-point rounding.
    The limit grows a little with the size of the exact value."""
    return error <= ERROR_FLOOR * max(1.0, abs(exact))


def experiment_polynomials():
    show("=" * 74)
    show("EXPERIMENT 2: POLYNOMIAL EXACTNESS")
    show("=" * 74)
    show("A rule is EXACT for a polynomial if its error is only rounding noise")
    show(f"(error <= {ERROR_FLOOR:g} * max(1, |exact value|)).")
    show()

    # Part A: monomials x^k on [0, 1] with n = 4
    show("Part A: f(x) = x^k on [0, 1], n = 4   (exact integral = 1/(k+1))")
    show(f"{'k':>2} {'exact':>12} {'Trap approx':>13} {'Trap error':>11} "
         f"{'exact?':>7} {'Simp approx':>13} {'Simp error':>11} {'exact?':>7}")
    show("-" * 74)

    degrees = [0, 1, 2, 3, 4, 5]
    trap_errors = []
    simp_errors = []
    trap_exact_flags = []
    simp_exact_flags = []
    for k in degrees:
        f = lambda x, k=k: x ** k
        exact = 1.0 / (k + 1)
        trap_row = compute_errors(composite_trapezoidal, f, 0.0, 1.0, exact, [4])[0]
        simp_row = compute_errors(composite_simpson, f, 0.0, 1.0, exact, [4])[0]
        trap_ok = is_exact(trap_row["error"], exact)
        simp_ok = is_exact(simp_row["error"], exact)
        trap_errors.append(trap_row["error"])
        simp_errors.append(simp_row["error"])
        trap_exact_flags.append(trap_ok)
        simp_exact_flags.append(simp_ok)
        show(f"{k:>2} {exact:>12.8f} {trap_row['approx']:>13.8f} "
             f"{trap_row['error']:>11.3e} {'EXACT' if trap_ok else 'no':>7} "
             f"{simp_row['approx']:>13.8f} {simp_row['error']:>11.3e} "
             f"{'EXACT' if simp_ok else 'no':>7}")
    show()

    # Highest degree up to which the rule is exact for every lower degree too
    trap_best = -1
    while trap_best + 1 < len(degrees) and trap_exact_flags[trap_best + 1]:
        trap_best += 1
    simp_best = -1
    while simp_best + 1 < len(degrees) and simp_exact_flags[simp_best + 1]:
        simp_best += 1
    show(f"Result: Trapezoidal is exact up to degree {trap_best}.")
    show(f"        Simpson is exact up to degree {simp_best}.")
    show("Expected: Trapezoidal up to degree 1, Simpson up to degree 3.")
    show()

    # Part B: general polynomials (not just x^k), on other intervals, several n
    show("Part B: general polynomials, several values of n")
    show("(exactness does not depend on n, as long as n is even)")
    show()
    show(f"{'polynomial':<24} {'n':>3} {'exact':>10} {'Trap error':>11} "
         f"{'exact?':>7} {'Simp error':>11} {'exact?':>7}")
    show("-" * 74)
    polynomials = [
        ("2x^3-3x^2+x+5 on [0,2]", [5, 1, -3, 2], 0.0, 2.0),
        ("x^4-2x+1 on [0,2]", [1, -2, 0, 0, 1], 0.0, 2.0),
    ]
    for name, coeffs, a, b in polynomials:
        f = lambda x, coeffs=coeffs: polynomial_value(coeffs, x)
        exact = polynomial_integral(coeffs, a, b)
        for n in [4, 8, 16]:
            trap_err = compute_errors(composite_trapezoidal, f, a, b,
                                      exact, [n])[0]["error"]
            simp_err = compute_errors(composite_simpson, f, a, b,
                                      exact, [n])[0]["error"]
            show(f"{name:<24} {n:>3} {exact:>10.6f} {trap_err:>11.3e} "
                 f"{'EXACT' if is_exact(trap_err, exact) else 'no':>7} "
                 f"{simp_err:>11.3e} "
                 f"{'EXACT' if is_exact(simp_err, exact) else 'no':>7}")
    show()

    make_polynomial_plot(degrees, trap_errors, simp_errors)


def make_polynomial_plot(degrees, trap_errors, simp_errors):
    """Error against polynomial degree: the jump shows where exactness ends."""
    # log axes cannot show 0, so use a tiny value instead of an error of 0
    trap_plot = [max(e, 1e-17) for e in trap_errors]
    simp_plot = [max(e, 1e-17) for e in simp_errors]

    plt.figure(figsize=(7, 5))
    plt.semilogy(degrees, trap_plot, "o-", label="Trapezoidal")
    plt.semilogy(degrees, simp_plot, "s-", label="Simpson")
    plt.axhline(ERROR_FLOOR, color="gray", linestyle="--",
                label="rounding level (1e-14)")
    plt.xlabel("polynomial degree k  (f(x) = x^k on [0, 1], n = 4)")
    plt.ylabel("absolute error (0 is drawn as 1e-17)")
    plt.title("Polynomial exactness")
    plt.xticks(degrees)
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "polynomial_exactness.png", dpi=150)
    plt.close()


# Experiment 4: challenging integrands
def experiment_challenging():
    show("=" * 74)
    show("EXPERIMENT 4: MORE CHALLENGING INTEGRANDS")
    show("=" * 74)
    study_function("sin(20x) on [0, 1]  (rapid oscillation)",
                   lambda x: math.sin(20 * x), 0.0, 1.0,
                   (1 - math.cos(20)) / 20, "hard_sin20x.png")
    study_function("1/(1+25x^2) on [-1, 1]  (sharp peak at x = 0)",
                   lambda x: 1 / (1 + 25 * x * x), -1.0, 1.0,
                   2 * math.atan(5) / 5, "hard_peak.png")
    study_function("sqrt(x) on [0, 1]  (infinite slope at x = 0)",
                   math.sqrt, 0.0, 1.0, 2.0 / 3.0, "hard_sqrt.png")


# Extra: both methods on every test function at the same n
def summary_plot():
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
        trap_errs.append(compute_errors(composite_trapezoidal, f, a, b,
                                        exact, [n])[0]["error"])
        simp_errs.append(compute_errors(composite_simpson, f, a, b,
                                        exact, [n])[0]["error"])

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
    plt.savefig(RESULTS_DIR / "summary_n16.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    RESULTS_DIR.mkdir(exist_ok=True)

    experiment_smooth()
    experiment_polynomials()
    experiment_challenging()
    summary_plot()

    (RESULTS_DIR / "tables.txt").write_text("\n".join(report_lines))
    print(f"Done. Tables and plots are saved in {RESULTS_DIR}")