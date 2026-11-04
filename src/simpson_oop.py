from numbers import Real

import numpy as np

from integrator_base import Function, NumericalIntegrator


class SimpsonIntegrator(NumericalIntegrator):
    """Integrate functions using Simpson's 1/3 rule."""

    def integrate(self, func: Function, a: Real, b: Real, n: int) -> float:
        self.validate_inputs(func, a, b, n)
        if n % 2:
            raise ValueError("n must be a positive even integer.")

        step = (b - a) / n
        points = np.linspace(a, b, n + 1)
        values = np.array([func(point) for point in points])
        result = (step / 3) * (
            values[0]
            + 4 * np.sum(values[1:n:2])
            + 2 * np.sum(values[2:n:2])
            + values[n]
        )
        return float(result)
