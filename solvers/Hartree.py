import numpy as np
import time
from math import pi
import copy
from scipy.special import erf
import tucker3d as tuck
import qtools as qt



def Hartree(molecule, psi_0, E_0, grid, T, num_iter, eps, ind):
    
    alpha = 0.0
    
    a = grid[0]
    b = grid[1]
    N = grid[2]
    
    h = (b-a)/(N-1)
    
    x = np.zeros(N)
    for i in xrange(N):
        x[i] = a + i*h

    sf = h ** (3./2) # scaling factor

    Norb = molecule.orbitals
    num_atoms = molecule.num_atoms
    
    
    ############################# Programm body ######################################
    
    psi = psi_0
    E = E_0
    V = [0]*Norb
    psi_new = [0]*Norb
    ################## Coulomb potential calculation ####################
            
    prod = lambda A,B: tuck.cross.multifun([A,B], eps, lambda (a,b): a*b)
    
    print "Coulomb potential..."
    
    pot_coulomb = tuck.zeros((N,N,N))
    for i in xrange(num_atoms):
        vec = molecule.atoms[i].rad
        charge = molecule.atoms[i].charge
        pot_coulomb = pot_coulomb - charge * qt.pots.coulomb(x, vec, ind, eps)
        pot_coulomb = tuck.round(pot_coulomb, eps)

    ################## Iteration process ####################
    print "Iteration process..."

    for k in xrange(num_iter):
        
        density = tuck.zeros((N,N,N))  # density calculation
        for i in xrange(Norb):
            density = density + prod(tuck.conj(psi[i]), psi[i]) # !can be faster!
            density = tuck.round(density, eps)
        pot_hartree = tuck.cross.conv(T, density, eps)
        pot_hartree = tuck.round(tuck.real(pot_hartree),eps)

        for i in xrange(Norb):
            V[i] =  prod(tuck.round(pot_coulomb + pot_hartree, eps), psi[i])
            V[i] = tuck.round(V[i], eps)
            
            psi_new[i] = -qt.poisson(2*V[i], -2*E[i], N, h, eps)

        ################# Fock matrix ###################

        L = np.linalg.cholesky(qt.bilinear(psi_new, psi_new)*sf**2) # orthogonalization
        psi_Q = qt.UT_prod(psi_new, H(np.linalg.inv(L)), eps)

        # Fock matrix

        V_new = [0]*Norb
        for i in xrange(Norb):
            V_new[i] =  prod(tuck.round(pot_coulomb + pot_hartree,eps), psi_Q[i])
            V_new[i] = tuck.round(V_new[i], eps)

        Energ1 = np.dot(np.diag(E), H(np.linalg.inv(L)))
        Energ2 = qt.bilinear(psi_new, psi_new)*sf**2
        Energ2 = np.dot(np.linalg.inv(L), Energ2)
        Energ = np.dot(Energ2, Energ1)
        F = qt.bilinear(psi_Q, V_new)*sf**2 - np.dot(qt.bilinear(psi_Q, V)*sf**2, H(np.linalg.inv(L))) + Energ
        print F
        #F = np.real((F + F.T)/2)

        E_new = np.zeros(Norb, dtype = np.complex128)
        E_new, S = np.linalg.eigh(F)
        #E_new = alpha*E + (1-alpha)*E_new
        
        print np.array(E_new)
        
        psi_new = qt.prod(psi_Q, S.T, eps)
        for i in xrange(Norb):
            psi[i] = alpha*psi[i] + (1-alpha)*qt.normalize(psi_new[i], h)
            psi[i] = tuck.round(psi[i], eps)
        
        E = E_new.copy()
        
        for i in xrange(Norb):
            if E[i]>0:
                E[i] = -0.0    
    
    return E, psi

def H(A):
    return np.transpose(np.conjugate(A))