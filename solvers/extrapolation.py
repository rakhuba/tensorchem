import numpy as np
import time
from math import pi
import copy
from scipy.special import erf
import tucker3d as tuck
import qtools as qt
import solvers

#Aitkens extrapolation scheme

def extrapolation(E, E_exact = 0.):
    
    n = len(E)
    Nlev = n/2
    
    if n == 2:
        return (4*E[1] - E[0])/3
    elif n == 1:
        raise Exception('Length of the input array must be greater than 1')

    res = [E]
    if E_exact <> 0.:
        err = [[abs(E_exact - E[i])/abs(E_exact ) for i in xrange(n)]]
    else:
        err = [[]]
    Wlev = n
    for i in xrange(1, Nlev+1):
        Wlev -= 2
        res.append([None] * Wlev)
        err.append([None] * Wlev)
        
        for j in xrange(Wlev):
            x2 = res[i-1][j+2]
            x1 = res[i-1][j+1]
            x0 = res[i-1][j]
            
            res[i][j] = x2 - (x2-x1)**2 / (x2-2*x1+x0)
            if E_exact <> 0.:
                err[i][j] = abs(res[i][j] - E_exact)/abs(E_exact)


    return res, err