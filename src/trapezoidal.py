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
        TypeError: If n is not an integer.
        ValueError: If n <= 0.
'''

def composite_trapezoidal(f,a,b,n): 
    if not isinstance(n, int):
        raise TypeError("n must be positive integer.")

    if n <= 0:
        raise ValueError("n must be greater than zero.")

    h = (b-a)/n

    boundary_value_sum = (f(a)+f(b))/2 

    for i in range(1,n):
        boundary_value_sum += f(a+(i*h))

    return h * boundary_value_sum
