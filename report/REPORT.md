## Experiments and Results

### Setup
Both composite rules were tested with n = 2, 4, 8, ..., 1024 subintervals (always even, as Simpson's rule requires). Absolute errors and observed convergence orders come from `error_analysis.py` (`convergence_study`, `format_table`). The observed order between two consecutive runs is p = log(e1/e2) / log(h1/h2). Errors at or below 1e-14 are rounding noise, so no order is reported for them. Full tables are in `results/tables.txt` and the plots are the `results/*.png` files.

### 1. Smooth functions (e^x on [0,1], sin x on [0,pi])
The observed order is 2.000 for the Trapezoidal rule and 4.000 for Simpson's rule, as theory predicts (error proportional to h^2 and h^4). For e^x with n = 16 the errors are 5.6e-4 (Trapezoidal) and 1.5e-7 (Simpson), so Simpson is about 3800 times more accurate for the same number of subintervals.

### 2. Polynomial exactness (x^k on [0,1], n = 4)
The Trapezoidal rule is exact for degree 0 and 1 and has errors from degree 2 onward. Simpson's rule is exact up to degree 3 and has errors from degree 4. The Trapezoidal error depends on f'' and Simpson's on the fourth derivative f'''', which is zero for every polynomial of degree 3 or lower.

### 3. Challenging integrands

**sin(20x) on [0,1]: why the order is negative at n = 4.**
One period of sin(20x) is about 0.31, but the step sizes at n = 2 and n = 4 are h = 0.5 and 0.25, which are larger than one period. The grid points cannot resolve the oscillation, so the error depends on where the points happen to fall and not on h. The Trapezoidal error even grows from 7.3e-2 (n = 2) to 1.29e-1 (n = 4), giving an order of -0.81. This is pre-asymptotic behaviour: the h^2 and h^4 error formulas only describe the error once h is small compared with the oscillation. Orders close to 2 and 4 appear from n = 32 (h = 0.031, about 10 points per period).

**1/(1+25x^2) on [-1,1]: why Simpson is worse than Trapezoidal at n = 8 to 16.**
The peak at x = 0 has a width of about 0.2, and the fourth derivative there is large (f''''(0) = 15000). At n = 8 (h = 0.25) and n = 16 (h = 0.125) the grid has not yet resolved the peak, so the error bound of Simpson's rule, proportional to h^4 f'''' , is far too large to be useful. Here Simpson's error is 2.6e-2 and 2.7e-3, while the Trapezoidal rule gives 7.5e-3 and 1.4e-4. The Trapezoidal values at these n are partly lucky: its orders (3.8, then 5.8, then 1.5) are not constant either. This is visible in `summary_n16.png`. Neither rule is in its asymptotic regime there, so the usual ranking (Simpson better) does not apply yet.

**Why Simpson's order jumps to 7.2 and 11.0 at n = 32 and 64.**
Between n = 16 and n = 64 Simpson's error drops from 2.7e-3 to 1.8e-5 to 9.1e-9, a factor of about 150 and then 2000, instead of the factor 16 that order 4 gives. The grid is crossing from "peak not resolved" to "peak resolved", and the error falls faster than any fixed power of h during that transition. These orders are not real convergence rates. Only after n >= 128 does the estimate settle at 4.126, 3.999 and 4.000. An order estimated from two points is only meaningful in the asymptotic regime.

**sqrt(x) on [0,1]: why both methods lose their expected orders.**
The error formulas assume the second derivative (Trapezoidal) and fourth derivative (Simpson) are bounded. For sqrt(x), f'(x) = 1/(2 sqrt(x)) is infinite at x = 0, so this assumption fails. The error is dominated by the first subinterval [0, h], where the function behaves like sqrt(x) and the local error is proportional to h^(3/2). Both methods therefore converge at about order 1.5 (observed 1.496 and 1.500 at n = 1024). Simpson's rule is still about 2.5 times more accurate than the Trapezoidal rule, but it loses its order-4 advantage. The standard remedy is the substitution x = t^2, which turns the integral into the polynomial 2t^2 on [0,1].

### Conclusions
1. For smooth functions Simpson's rule is much more accurate, and the observed orders (2 and 4) match the theory.
2. The theoretical orders only appear once h is small enough to resolve the function. Before that, the error can be erratic, and Simpson can even be worse than the Trapezoidal rule.
3. Both rules depend on smoothness. A singular derivative such as in sqrt(x) reduces both to order 1.5.
4. Orders computed from errors near 1e-14 or lower are meaningless because of floating-point rounding.