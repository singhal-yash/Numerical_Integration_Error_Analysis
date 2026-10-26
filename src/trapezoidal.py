from numbers import Real
import math 

'''
    Approximate the definite integral of f over [a, b] using the composite Trapezoidal Rule.

    Parameters
        f : Function to integrate.
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
        TypeError: If n is not an integer. If f is not callable. If a, b are not real. If n,a,b are bool type.  
        ValueError: If n <= 0. If a,b not finite real number. 
'''

def composite_trapezoidal(f,a,b,n): 
    if not callable(f):
        raise TypeError("f must be a callable function.")
    
    if not isinstance(n, int) or isinstance(n, bool):
        raise TypeError("n must be positive integer.")

    if (isinstance(a, bool) or isinstance(b,bool) or not isinstance(a, Real) or not isinstance(b, Real)):
        raise TypeError("a and b must be real numbers.")

    if not math.isfinite(a) or not math.isfinite(b):
        raise ValueError("a and b must be real finite numbers.")

    if n <= 0:
        raise ValueError("n must be greater than zero.")

    h = (b-a)/n

    boundary_value_sum = (f(a)+f(b))/2 

    for i in range(1,n):
        boundary_value_sum += f(a+(i*h))

    return h * boundary_value_sum

if __name__ == "__main__" :
    result = composite_trapezoidal(math.sin, 0, math.pi, 100)

    print("Composite Trapezoidal Rule for function: sin(x) over [0, pi] with Subintervals: 100")
    print("Approximation:", result)
    print("Exact value:", 2.0)
    print("Absolute error:", abs(result - 2.0))
