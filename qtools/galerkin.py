import numpy as np
from math import pi
import copy
from scipy.special import erf
import tucker3d as tuck

def newton(x, eps, ind):
    
    # galerkin tensor for convolution as a hartree potential
    
    if ind == 6:
        a, b, r = -15., 10, 80
    elif ind == 8:
        a, b, r = -20., 15, 145
    elif ind == 10:
        a, b, r = -25., 20, 220
    elif ind == 12:
        a, b, r = -30., 25, 320
    else:
        raise Exception("wrong ind parameter")
    
    
    N = x.shape
    
    hr = (b-a)/(r - 1)
    h = x[1]-x[0]
    
    s = np.array(range(r), dtype = np.complex128)
    s = a + hr * (s - 1)
    
    w = np.zeros(r, dtype = np.complex128)
    for alpha in xrange(r):
        w[alpha] = 2*hr * np.exp(s[alpha]) / np.sqrt(pi)
    w[0]   = w[0]/2
    w[r-1] = w[r-1]/2
    
    
    U = np.zeros((N[0], r), dtype = np.complex128)
    for alpha in xrange(r):
        U[:, alpha] = (  func_int(x-h/2, x[0]-h/2, np.exp(2*s[alpha])) -
                       func_int(x+h/2, x[0]-h/2, np.exp(2*s[alpha])) +
                       func_int(x+h/2, x[0]+h/2, np.exp(2*s[alpha])) -
                       func_int(x-h/2, x[0]+h/2, np.exp(2*s[alpha]))  )
    
    newton = tuck.can2tuck(w, U, U, U)
    newton = tuck.round(newton, eps)
    
    return (1./h**3) * newton




def func_int(x, y, a):
    
    if (a*(2*np.max(x))**2 > 1e-10):
        f = -(np.exp(-a*(x-y)**2)-1)/(2*a) + np.sqrt(pi/a)/2 * (
                                                                (y - x) * erf(np.sqrt(a) * (x-y))  )
    else:
        f = (-(x-y)**2/2) 
    return f    

