import numpy as np
from math import pi
import tucker3d as tuck

def coulomb(x, vec, ind, eps):  # 1/|r-vec| in Tucker format

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
    h = (b-a)/(r - 1)

    s = np.array(range(r))
    s = a + h * (s - 1)

    w = np.zeros(r, dtype = np.float64)
    for alpha in xrange(r):
        w[alpha] = 2*h * np.exp(s[alpha]) / np.sqrt(pi)
    w[0]   = w[0]/2
    w[r-1] = w[r-1]/2


    U1 = np.zeros((N[0], r), dtype = np.float64)
    U2 = np.zeros((N[0], r), dtype = np.float64)
    U3 = np.zeros((N[0], r), dtype = np.float64)

    for alpha in xrange(r):
        U1[:, alpha] = np.exp(-(x - vec[0])**2 * np.exp(2*s[alpha]))
        U2[:, alpha] = np.exp(-(x - vec[1])**2 * np.exp(2*s[alpha]))
        U3[:, alpha] = np.exp(-(x - vec[2])**2 * np.exp(2*s[alpha]))

    
    newton = tuck.can2tuck(w, U1, U2, U3)
    newton = tuck.round(newton, eps)
    
    return newton


def exchange(psi1, psi2, i, eps_exchange, T, molecule):

    exchange = tuck.zeros(psi1[0].n)
    for j in xrange(molecule.orbitals):
        conv = tuck.cross.multifun([psi1[i], psi2[j]], eps_exchange, lambda (a,b): a*b, y0 = psi1[i])
        conv = tuck.cross.conv(T, conv, eps_exchange, y0 = conv)
        conv = tuck.round(tuck.real(conv),eps_exchange)
        conv = tuck.cross.multifun([psi2[j], conv], eps_exchange, lambda (a,b): a*b,y0 = psi2[j])
        exchange = exchange + conv
        exchange = tuck.round(exchange, eps_exchange)

    return exchange
       
    
