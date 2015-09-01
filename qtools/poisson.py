import numpy as np
import time
from math import pi
import copy
import tucker3d as tuck

################# poisson solver ######################



def poisson(F, mu, N, h, eps, r_add = 4, solver='Fourier'):   # check N+1

    #solves (-\Delta + mu) = F with F.shape = N \times N \times N  
    
    U1_L = np.ones((F.n[0], 4), dtype = np.float64)
    U2_L = U1_L.copy()
    U3_L = U1_L.copy()
    
    if solver=='Fourier':
        U1_L[:, 0] = 4. * np.sin(1./(N+1)*(np.array(range(F.n[0]))+1)*pi/2)**2 / h**2
    elif solver=='spectral':
        U1_L[:, 0] = np.pi**2 * (np.array(range(F.n[0]))+1)**2
    else:
        raise Exception('Incorrect poisson solver name')
    U1_L[:, 3] = mu * U1_L[:, 3].copy()
    U2_L[:, 1] = U1_L[:, 0].copy()
    U3_L[:, 2] = U1_L[:, 0].copy()
    g = np.ones(F.n[0], dtype = np.float64)


    L = tuck.can2tuck(g, U1_L, U2_L, U3_L)
    #L = tuck.round(L, 1e-14)
    
    f = tuck.dst(F)

    frac = tuck.cross.multifun([f, L], eps, lambda (a, b): a/b, r_add = r_add)
    sol = tuck.dst(frac)

    return sol


