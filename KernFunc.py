import math
import numpy as np

def distance(x):
    """
  x - 1D vector

  Return:
  r - matrix N x N of distance between all particles 
    """
    D = x.shape[0]

    temp = x.reshape((D, 1))

    r = temp - temp.T

    return r

def W( x, h ):
    """
  x - 1D vector
  h - smoothing parameter
  Return:
  w - matrix N x N of W(r, h)
    """

    r = distance(x)
    w = (1.0 / (h*math.sqrt(np.pi))) * np.exp( -((r*r) / (h*h)))

    return w

def gradW( x, h ):
    """
    x - 1D vector
    h - smoothing parameter

    Return:
    dw - matrix; grad of kernel function
    """
    r = distance(x)
    dw = -2 * np.exp( -((r*r) / (h*h)))* (1. / (math.sqrt(np.pi) * h**3))*r

    return dw

def lap_W( x,h ):
    """
    x - 1D vector
    h - smoothing parameter

    Return:
    lapw - matrix; second deriv of kernel function
    """

    r = distance(x)
    lapw = -2 * np.exp( -((r*r) / (h*h))) * (1. / (math.sqrt(np.pi) * h**3)) * (1 - (2*r*r)/(h*h))

    return lapw