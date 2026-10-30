from numbers import Real, Integral
import numpy as np
import math 

'''
    Approximate the definite integral of f over [a, b] using the composite Trapezoidal Rule.

    Parameters
        f : Callable function to integrate.
        a : dtype(float); Lower limit of integration.
        b : dtype(float); Upper limit of integration.
        n : dtype(int); Number of subintervals (must be a positive integer).

    Mathematical formula
        use the composite trapezoidal rule for n equal subintervals, 
        step size, h = (b-a)/n
        integral = h*(f(a)+ 2*sum_{i=1}^{n-1}(f(a+h*i)) + f(b))/2 

    Returns 
        integral: dtype(float); Approximation of the definite integral.

    Raises
        TypeError: If either n is not an integer or n is of bool type. If f is not callable. If a, b are not real numbers.  
        ValueError: If n <= 0. If a,b are not finite real numbers. 
'''

def composite_trapezoidal(f,a,b,n): 
    if not callable(f):
        raise TypeError("f must be a callable function.")
    
    if not isinstance(a, Real) or not isinstance(b, Real):
        raise TypeError("a and b must be real numbers.")

    if not math.isfinite(a) or not math.isfinite(b):
        raise ValueError("a and b must be real finite numbers.")

    if not isinstance(n, Integral) or isinstance(n, bool):
        raise TypeError("n must be positive integer.")

    if n <= 0:
        raise ValueError("n must be greater than zero.")

    h = (b-a)/n

    boundary_value_sum = (f(a)+f(b))/2 

    for i in range(1,n):
        boundary_value_sum += f(a+(i*h))

    return h * boundary_value_sum

if __name__ == "__main__" :
    result = composite_trapezoidal(math.sin, False , np.float64(math.pi), np.int64(100))
    exact_value = 2
    print("Composite Trapezoidal Rule Approximation:", result)
    print("Exact value:", exact_value)
    print("Absolute error:", abs(result - exact_value))
