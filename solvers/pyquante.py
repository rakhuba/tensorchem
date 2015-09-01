import numpy as np
import time
from math import pi
import qtools as qt
import tucker3d as tuck
import solvers

# In pyquante fixed file PyQuante-1.6.4-py2.7-macosx-10.5-x86_64.egg/PyQuante/NumWrap.py which causes "libint extension not found, switching to normal ERI computation"
from PyQuante.Molecule import Molecule
from PyQuante import SCF
from PyQuante import CGBF
from PyQuante.PyQuante2 import SubspaceSolver
import numpy as np
from numpy.linalg import svd,cholesky,solve

import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt


def pyquante(mol, x, eps, basis = "sto-3g"):

    
    atoms = [None]*mol.num_atoms
    for i in xrange(mol.num_atoms):
        atoms[i] = (mol.atoms[i].charge,
                    (mol.atoms[i].rad[0],mol.atoms[i].rad[1],mol.atoms[i].rad[2]))
        
    mol_pq = Molecule(mol.name, atoms, units='Bohr')
    start = time.time()
    solver = SCF(mol_pq,method="HF",ConvCriteria=1e-7,MaxIter=40,basis=basis)
    solver.iterate()
    end = time.time()
    print 'SCF iteration - done'
    print (end - start)



    norb=int(solver.solver.nclosed)
    w=solver.basis_set.get()

    
    cf=solver.solver.orbs[:,0:norb]

    psi = [None]*norb
    for i in xrange(norb):
        psi[i] = gto2tuck(w, cf[:,i], x, eps)
        print psi[i].r
        #psi[i] = local(psi[i],1e-14)
        #psi[i] = tuck.round(psi[i],eps)

    E = solver.solver.orbe[:norb]

    return psi,E

def local(a,eps):
    
    b = tuck.zeros(a.n, dtype = np.float64)
    b.n = a.n

    u1,v1,r1 = svd_trunc(a.u[0],eps)
    u2,v2,r2 = svd_trunc(a.u[1],eps)
    u3,v3,r3 = svd_trunc(a.u[2],eps)
    
    b.u[0] = np.real(u1)
    b.u[1] = np.real(u2)
    b.u[2] = np.real(u3)
    b.r = (r1,r2,r3)
    

    g = np.dot(a.core, np.transpose(v3))
    g = np.transpose(g, [2,0,1])
    g = np.dot(g, np.transpose(v2))
    g = np.transpose(g, [0,2,1])
    g = np.dot(g, np.transpose(v1))
    b.core = np.real(np.transpose(g, [2,1,0]))
  


    return b



def svd_trunc(A, eps = 1e-14):
    
    u, s, v = np.linalg.svd(A,full_matrices = False)
    
    N1, N2 = A.shape
    
    eps_svd = eps*s[0]/np.sqrt(3)
    r = min(N1, N2)
    for i in xrange(min(N1, N2)):
        if s[i] <= eps_svd:
            r = i
            break
    #print s/s[0]
    u = u[:,:r].copy()
    v = v[:r,:].copy()
    s = s[:r].copy()
    
    return u, np.dot(np.diag(s),(v)), r


from numba import autojit
@autojit
def gto2tuck(w,cf,x, eps):
    
    n = len(x)
    r = len(cf)
    
    k=0
    for alpha in xrange(r):
        k += len(w[alpha].prims)
    
    U1 = np.zeros((n,k*r))
    U2 = np.zeros((n,k*r))
    U3 = np.zeros((n,k*r))
    c = np.zeros(k*r)
    
    nG = 0
    for alpha in xrange(r):
        for beta in xrange(len(w[alpha].prims)):
            prim = w[alpha].prims[beta]
            i,j,k = prim.powers
            x0,y0,z0 = prim.origin
            #print [x0,y0,z0]
            U1[:,beta + nG] = pow(x-x0,i)*np.exp(-prim.exp*(x-x0)**2)
            U2[:,beta + nG] = pow(x-y0,j)*np.exp(-prim.exp*(x-y0)**2)
            U3[:,beta + nG] = pow(x-z0,k)*np.exp(-prim.exp*(x-z0)**2)
            c[beta + nG] = w[alpha].norm*prim.norm*prim.coef*cf[alpha]
        nG += len(w[alpha].prims)


    u1, v1 = tuck.cross.cross2d_full(U1, eps)
    u2, v2 = tuck.cross.cross2d_full(U2, eps)
    u3, v3 = tuck.cross.cross2d_full(U3, eps)

            
    v1 = np.real(H(v1))
    v2 = np.real(H(v2))
    v3 = np.real(H(v3))
            
    r1 = v1.shape[0]
    r2 = v2.shape[0]
    r3 = v3.shape[0]

    maxrank = max([r1, r2, r3])
    core = np.zeros((maxrank, maxrank, maxrank))
    for a1 in xrange(r1):
        for a2 in xrange(r2):
            for a3 in xrange(r3):
                for a in xrange(len(c)):
                    core[a1, a2, a3] += c[a] * v1[a1, a] * v2[a2, a] * v3[a3, a]

    u1_new = np.zeros((n, maxrank))
    u2_new = np.zeros((n, maxrank))
    u3_new = np.zeros((n, maxrank))

    u1_new[:, :r1] = u1
    u2_new[:, :r2] = u2
    u3_new[:, :r3] = u3

    b = tuck.tensor()
    b.core = core.copy()
    b.u[0] = np.real(u1_new)
    b.u[1] = np.real(u2_new)
    b.u[2] = np.real(u3_new)

    b.r = core.shape
    b.n = (n, n, n)

    return b



def H(A):
    return np.transpose(np.conjugate(A))
