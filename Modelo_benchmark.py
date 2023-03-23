"""
Example of a program containg a function to be minimized
@author: Renilton
"""
from numpy import sin, cos, sqrt, exp, e


def modelo_benchmark(x, function, num_dim):
    """
    Function chosen to be optimized (minimize)

    Parameters
        ----------
        x : char
            function variable
        function : int
            index of the function to choose
        num_dim : int
            number of dimensions of the function
    """
    # Exponencial
    if function == 1 or function == 'Exponencial':
        total = 0
        for i in range(num_dim):
            total += x[i] ** 2
        return total
    # Rastrigin
    elif function == 2 or function == 'Rastrigin':
        total = 0
        for i in range(num_dim):
            total += x[i]**2-10*cos(2*3.14*x[i])+10
        return total
    # Shaffer's f6
    elif function == 3 or function == 'Shaffer':
        total = 0
        for i in range(num_dim):
            total += 0.5+((sin(sqrt(x[i]**2)))**2-0.5)/(1.0 + 0.001*(x[i]**2))**2
        return total
    # Ackley
    elif function == 4 or function == 'Ackley':
        total = 0
        for i in range(num_dim):
            total += -20.0 * exp(-0.2 * sqrt((x[i]**2)/2)) - exp(cos(2*3.14*x[i]))/2 + 20 + e
        return total
