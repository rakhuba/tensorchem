import numpy as np
import time
from math import pi
import copy
from scipy.special import erf
import tucker3d as tuck
import qtools as qt
import solvers

# m is number of mixed vectors

def mixing(solver, molecule, psi_0, E_0, eps, grid, T, ind, max_iter, m, max_iter_inscf,  pr):
    
    count = 0
    
    a = grid[0]
    b = grid[1]
    N = grid[2]
    
    h = (b-a)/(N-1)
    
    x = np.zeros(N)
    for i in xrange(N):
        x[i] = a + i*h
    
    Norb = molecule.orbitals
    num_atoms = molecule.num_atoms
    
    pot_coulomb = tuck.zeros((N,N,N))
    for i in xrange(num_atoms):
        vec = molecule.atoms[i].rad
        charge = molecule.atoms[i].charge
        pot_coulomb = pot_coulomb - charge * qt.pots.coulomb(x, vec, ind, eps)
        pot_coulomb = tuck.round(pot_coulomb, eps)
    
    G = lambda psi, E, rho: solvers.dft_dens.lda(molecule, psi, E, rho, grid, T, eps, pot_coulomb, max_iter = max_iter_inscf, pr=pr)[0:4]


    Norb = molecule.orbitals

    rho = psi2dens(psi_0, eps)

    rho_m = []
    Grho_m = []
    D_m = []

    rho_m.append(rho)

    res = G(psi_0, E_0, rho)
    psi = res[0]
    E = res[1]
    Grho_m.append(res[2])


    D_m.append(tuck.round(Grho_m[0] - rho, eps))
    rho = Grho_m[0]

    err_and = np.zeros(max_iter)
    for k in xrange(1, max_iter):
        mk = min(k, m)
    
        f = np.zeros(mk + 1)
        f[mk] = 1.
    
        A = np.ones((mk + 1, mk + 1))
        A[mk, mk] = 0.
    
    
        for p in xrange(mk):
            for q in xrange(mk):
                A[p, q] += 2 * qt.inner(D_m[p], D_m[q], h)

    
        alpha = np.linalg.pinv(A).dot(f)
        #alpha = np.array([0.5, 0.5, 1.])

        if len(alpha) == 3:
            if alpha[0]> 1. + 1e-2:
                alpha[0] = .1
                alpha[1] = .9

            if alpha[0]< 0. - 1e-2:
                alpha[0] = 0.1
                alpha[1] = .9

        rho_temp = tuck.zeros((N, N, N))
        for p in xrange(mk):
            rho_temp += tuck.round(alpha[p] * Grho_m[p], eps/10)
        rho = tuck.round(rho_temp, eps/10)

        #print alpha


        rho_m.append(rho)
        rho_m = rho_m[-min(k + 1, m):]


        res = G(psi, E, rho)
        psi = res[0]
        E_new = res[1]

        err_and[k] = max(np.abs(E - E_new)/np.abs(E_new))
            #if err_and[k] < eps:
                #break


        solver.psi = psi
        solver.orb_energies = E_new

        print solver.psi[0].r, solver.psi[-1].r
        print E_new
        err = abs((E-E_new)/E)
        E = copy.copy(E_new)
        #solver.iterative.orb_energies.append(E_new)
        #solver.iterative.convergence.append(err)
        print 'Iteration', k, 'accuracy = %.2e' % max(err)

        if max(err) < 4 * eps:
            count += 1
            if count == 4:
                print 'Process converged with', i, 'iterations'
                break


        Grho_m.append(res[2])
        Grho_m = Grho_m[-min(k + 1, m):]

        D_m.append(tuck.round(Grho_m[-1] - rho_m[-1], eps))
        D_m = D_m[-min(k + 1, m):]

    Eel = solvers.dft_dens.lda(molecule, psi, E, rho, grid, T, eps*1e-2, pot_coulomb, max_iter = max_iter_inscf, pr=pr)[1]
    Efull = solvers.dft_dens.lda_full_energy(molecule, psi, E, rho, grid, T, eps*1e-2, ind, pr = None) + 2*np.sum(Eel)
    print Efull
    solver.energy = Efull
            
    return psi, E, rho, Efull, err_and


#err_and[k-1]  = np.linalg.norm( sol - E)

def psi2dens(psi, eps):
    
    density = tuck.zeros(psi[0].n)
    for i in xrange(len(psi)):
        density = density + tuck.cross.multifun([psi[i], psi[i]], eps/10, lambda (a,b): a*b, y0 = psi[i])
        density = tuck.round(density, eps)
    density = 2*density
    
    return density