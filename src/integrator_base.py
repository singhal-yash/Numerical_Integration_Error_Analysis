from abc import ABC, abstractmethod
from numbers import Integral, Real
from typing import Callable
import math


Function = Callable[[float], float]


class NumericalIntegrator(ABC):
    """Base interface for composite numerical integration methods."""

    @staticmethod
    def validate_inputs(func: Function, a: Real, b: Real, n: int) -> None:
        if not callable(func):
            raise TypeError("func must be a callable function.")
        if not isinstance(a, Real) or not isinstance(b, Real):
            raise TypeError("a and b must be real numbers.")
        if not math.isfinite(a) or not math.isfinite(b):
            raise ValueError("a and b must be finite numbers.")
        if not isinstance(n, Integral) or isinstance(n, bool):
            raise TypeError("n must be an integer.")
        if n <= 0:
            raise ValueError("n must be greater than zero.")

    @abstractmethod
    def integrate(self, func: Function, a: Real, b: Real, n: int) -> float:
        """Approximate the integral of ``func`` over ``[a, b]``."""

    def __call__(self, func: Function, a: Real, b: Real, n: int) -> float:
        return self.integrate(func, a, b, n)
