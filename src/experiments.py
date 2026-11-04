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
def experiment_polynomials():
    show("=" * 74)
    show("EXPERIMENT 2: POLYNOMIAL EXACTNESS  (f(x) = x^k on [0, 1], n = 4)")
    show("=" * 74)
    show("Exact integral of x^k on [0,1] is 1/(k+1).")
    show(f"'EXACT' means the error is at most {ERROR_FLOOR:g} (rounding only).")
    show()
    show(f"{'degree k':>9} {'Trap error':>14} {'Trap':>7} "
         f"{'Simp error':>14} {'Simp':>7}")
    show("-" * 56)

    for k in range(0, 6):
        f = lambda x, k=k: x ** k
        exact = 1.0 / (k + 1)
        trap_err = compute_errors(composite_trapezoidal, f, 0.0, 1.0,
                                  exact, [4])[0]["error"]
        simp_err = compute_errors(composite_simpson, f, 0.0, 1.0,
                                  exact, [4])[0]["error"]
        trap_word = "EXACT" if trap_err <= ERROR_FLOOR else "no"
        simp_word = "EXACT" if simp_err <= ERROR_FLOOR else "no"
        show(f"{k:>9} {trap_err:>14.3e} {trap_word:>7} "
             f"{simp_err:>14.3e} {simp_word:>7}")
    show()
    show("Expected: Trapezoidal exact for degree <= 1, Simpson for degree <= 3.")
    show()


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