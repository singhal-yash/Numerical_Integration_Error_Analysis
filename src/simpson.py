
import numpy as np


def simpson(func, a, b, n):
    """Approximate the integral of func over [a, b] using Simpson's rule."""
    if not callable(func):
        raise TypeError("func must be a callable function.")
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        raise TypeError("a and b must be numeric.")

    if not isinstance(n, int) or isinstance(n, bool):
        raise TypeError("n must be an integer.")

    if a >= b:
        raise ValueError("a must be less than b.")

    if n <= 0 or n % 2 != 0:
        raise ValueError("n must be a positive even integer.")

    h = (b - a) / n # this is the width of each subinterval
    x = np.linspace(a, b, n + 1) # generate n+1 equally spaced points between a and b
    y = np.asarray(func(x)) # evaluate the function at each x value

    integral = (h / 3) * (
        y[0]
        + 4 * np.sum(y[1:n:2])
        + 2 * np.sum(y[2:n:2])
        + y[n]
    )

    return integral


if __name__ == "__main__":
    result = simpson(np.sin, 0, np.pi, 10)
    print(f"Integral ≈ {result:.8f}")
