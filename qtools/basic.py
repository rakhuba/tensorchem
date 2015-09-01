import numpy as np
from math import pi
import copy
import tucker3d as tuck

def gaussian(x, gamma): # Tucker representation of gaussian exp(-gamma * r**2)

    n = x.shape[0]
    a = tuck.ones((n, n, n))
    a.u[0][:,0] = np.exp(-gamma * x**2)
    a.u[1][:,0] = a.u[0][:,0]
    a.u[2][:,0] = a.u[0][:,0]

    return a
 
def inner(a, b, h):   # inner
    return h**3 * tuck.dot(a, b)

def normalize(a, h):
    return 1./np.sqrt(inner(a, a, h)) * a

def gram(psi, h): # can be faster !!!!!!!!!!!! - gram = gram.T

    Norb = len(psi)
    gram = np.zeros((Norb, Norb), dtype = np.float64)

    for i in xrange(Norb):
        for j in xrange(Norb):
            gram[i,j] = inner(psi[i], psi[j], h)

    return gram

def bilinear(psi1, psi2):

    Norb = len(psi1)
    bil = np.zeros((Norb, Norb), dtype = np.float64)

    for i in xrange(Norb):
        for j in xrange(Norb):
            bil[i,j] = tuck.dot(psi1[i], psi2[j])

    return bil

def LT_prod(psi, L, eps): #psi by L - Lower Triangular matrix multiplication

    Norb = len(psi) 
    psi_new = [tuck.zeros(psi[0].n)]*Norb
    for i in xrange(Norb):
        for j in xrange(i+1):
            psi_new[i] = psi_new[i] + L[i,j]*psi[j]
            psi_new[i] = tuck.round(psi_new[i], eps)
    return psi_new

def UT_prod(psi, R, eps): #psi by R - Upper Triangular matrix multiplication
    
    Norb = len(psi)
    psi_new = [tuck.zeros(psi[0].n)]*Norb
    for i in xrange(Norb):
        for j in xrange(i+1):
            psi_new[i] = psi_new[i] + R[j,i]*psi[j]
            psi_new[i] = tuck.round(psi_new[i], eps)
    return psi_new

def prod(psi, S, density, eps): #psi by S matrix multiplication

    Norb = len(psi)
    psi_new = [tuck.zeros(psi[0].n)]*Norb
    for i in xrange(Norb):
        for j in xrange(Norb):
            psi_new[i] = psi_new[i] + S[i,j]*psi[j]
            psi_new[i] = tuck.round(psi_new[i], eps)
    
        #def psi_S_i(psi):
        #    res = 0.
        #    for j in xrange(Norb):
        #        res += (S[i, j]) * psi[j]
        #    return np.real(res)
        
        #psi_new[i] = tuck.cross.multifun(psi, eps, lambda a: psi_S_i(a), y0 = density, pr = 1)
    return psi_new
