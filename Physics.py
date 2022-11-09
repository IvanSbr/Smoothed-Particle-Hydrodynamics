from KernFunc import *
import numpy as np
import math

def get_density(x, m, h):
    """
  x - vector or matrix of distance
  m - mass
  h - smoothing parameter

  Return:
  rho - density x_i (on every particle)
    """

    D = x.shape[0]
    rho = np.sum(m * W(x, h), 1).reshape((D, 1))
    
    return rho

def get_grad_density(x, m, h):
    """
  x - vector or matrix of distance
  m - mass
  h - smoothing parameter

  Return:
  d_rho -  derivative of density x_i (on every particle)
    """

    D = x.shape[0]
    r = distance(x)
    d_rho = np.sum(m * gradW(x, h), 1).reshape((D, 1))

    return d_rho

def get_lap_density(x, rho, m, h):
    """
  x - vector or matrix of distance
  m - mass
  h - smoothing parameter

  Return:
  lap_rho - second derivative of density x_i (on every particle)
    """

    D = x.shape[0]
    lap_rho = np.sum(m/rho.T * (rho.T - rho)*lap_W(x, h), 1).reshape((D, 1))
    
    return lap_rho

def pressure(x, rho, d_rho, lap_rho, m, h):
    """
  x - distance
  rho -  density
  d_rho - derivative of density
  lap_rho - second derivative of density
  m - mass
  h - smoothing parameter

  Return:
  P - pressure on particles 
    """

    D = x.shape[0]
    r = distance(x)
    P = np.sum(m/rho.T * 0.25 * ((d_rho.T * d_rho.T)/rho.T - lap_rho.T) * W(x, h), 1).reshape((D,1))

    return P

def Acceleration(x, P, rho, vel, gamma, m ,h):
    """
  x - distance
  P - pressuare
  rho -  density
  vel - velocity
  gamma - damping parameter
  m - mass
  h - smoothing parameter

  Return:
  acc - acceleration
    """

    r = distance(x)
    acc = - np.sum(m*( P/(rho* rho) + P.T/(rho.T * rho.T))* gradW(x, h), 1)

    acc -= (gamma * vel + x)

    return acc