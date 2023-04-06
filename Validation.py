"""
Auxiliary def for verification
@author: Renilton
"""
from numpy import shape, ndarray

def validation(bounds, num_part, maxiter, **kwargs):
    if (type(bounds) != ndarray):
        raise TypeError("The variable 'bounds' must be an array")
    elif type(num_part) != int:
        raise TypeError("The variable 'num_part' must be an integer.")
    elif type(maxiter) != int:
        raise TypeError("The variable 'maxiter' must be an integer.")
    elif (shape(bounds)[1] != 2):
        raise TypeError("The variable 'bounds' must have 2 columns")
