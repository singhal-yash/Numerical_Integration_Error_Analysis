import math
import numpy as np


class Simpson:
    """
    A class to perform numerical integration using Simpson's rule. 
    """
    def __init__(self, func, a, b, n):
        self.func = func
        self.a = a  
        self.b = b
        self.n = n

    def integrate(self):
        if self.n % 2 == 1:
            raise ValueError("n must be an even integer.")
        
        h = (self.b - self.a) / self.n
        x = np.linspace(self.a, self.b, self.n + 1)
        y = self.func(x)

        integral = (h / 3) * (y[0] + 2 * np.sum(y[2:self.n:2]) + 4 * np.sum(y[1:self.n:2]) + y[self.n])
        return integral

    def __str__(self):
        return f"Simpson's Rule: a={self.a}, b={self.b}, n={self.n}"

    def __repr__(self):
        return f"Simpson(func={self.func.__name__}, a={self.a}, b={self.b}, n={self.n})"

    def __eq__(self, other):
        if not isinstance(other, Simpson):
            return NotImplemented
        return (self.func.__name__ == other.func.__name__ and
                self.a == other.a and
                self.b == other.b and
                self.n == other.n)


if __name__ == "__main__":
    def f(x):
        return np.sin(x)

    a = 0
    b = math.pi
    n = 10 

    simpson = Simpson(f, a, b, n)
    result = simpson.integrate()
    print(f"Integral of sin(x) from {a} to {b} is approximately: {result:.5f}.")

