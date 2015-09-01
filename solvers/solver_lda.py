import numpy as np
import time
from math import pi
import copy
from scipy.special import erf
import tucker3d as tuck
import qtools as qt

gamma = -0.1423
beta1 = 1.0529
beta2 = 0.3334
A = 0.0311
B = -0.048
C = 0.0020191519406228
D = -0.0116320663789130



def solver_lda(solver, molecule, psi, E, grid, T, ind, eps0, pot_coulomb, max_iter, max_iter_scf=1):
    solver.iterative.orb_energies = []
    solver.iterative.convergence = []
    k=0
    eps = eps0
    for i in xrange(max_iter):
        solver.psi, E_new = lda_scf(solver, molecule, solver.psi, E,
                                    grid, T, eps, pot_coulomb, pr=None, max_iter=max_iter_scf)
        print solver.psi[0].r, solver.psi[-1].r
        err = abs((E-E_new)/E)
        solver.iterative.orb_energies.append(E_new)
        solver.iterative.convergence.append(err)
        print 'Iteration', i, 'accuracy = %.2e' % max(err)
        #if max(err)<eps*10:
        #    print eps
        #    eps = eps * 1e-2
        #    if eps < eps0:
        #        eps = eps0
        if eps == eps0 and max(err) < 5 * eps:
            k += 1
            if k == 4:
                print 'Process converged with', i, 'iterations'
                break
        E = E_new
        print E
#print E

    if i == max_iter - 1:
        print ''
        print 'Process did not converge with eps precision'
        print ''
    
    solver.orb_energies = E_new
    E_full = lda_full_energy(molecule, solver.psi, E_new, pot_coulomb, grid, T, eps, ind, pr = None)
    solver.energy = E_full



def lda_scf(solver, molecule, psi_0, E, grid, T, eps, pot_coulomb, pr=None, max_iter=1):
    
    E_correct = molecule.energy
    
    
    a = grid[0]
    b = grid[1]
    N = grid[2]
    
    h = (b-a)/(N-1)
    
    x = np.zeros(N)
    for i in xrange(N):
        x[i] = a + i*h
    
    accuracy = np.zeros(max_iter)
    timing = np.zeros(max_iter)
    
    sf = h ** (3./2) # scaling factor
    
    Norb = molecule.orbitals
    num_atoms = molecule.num_atoms

            ##### Output #####
    #output_E = np.zeros((Norb+1, max_iter))
    #output_r = np.zeros((3, max_iter))
    #fname = molecule.name + '_' + 'lda' + '_'  + str(N) + '_' + str(eps)
    
    ############################# Programm body ######################################
    
    psi = psi_0
    V = [0]*Norb
    psi_new = [0]*Norb

    prod = lambda A,B,C: tuck.cross.multifun([A,B], eps, lambda (a,b): a*b, y0=C, pr=pr)

    density = psi2dens(psi_0, eps)

    ################## Iteration process ####################
    
    pot_hartree = tuck.cross.conv(T, density, eps, y0=density, pr=pr)
    pot_hartree = tuck.round(tuck.real(pot_hartree), eps)
    pot_V = tuck.round(pot_coulomb + pot_hartree, eps)
    
    for k in xrange(max_iter):
    
        #print "1 scf iter", E
        
        for i in xrange(Norb):
            V[i] =  prod(pot_V, psi[i], psi[i])
            psi[i] = tuck.real(psi[i])
            psi[i] = tuck.round(psi[i], eps)
        

        for i in xrange(Norb):
            Vxc = tuck.cross.multifun(psi_0 + [psi[i]], 10*eps, lambda a: ((cpot_fun(density_fun(a[:-1])) + xpot_fun(density_fun(a[:-1])))*(a[-1])), y0 = psi[i], pr=pr)
            V[i] = tuck.round(V[i] + Vxc, eps)
            psi_new[i] = -qt.poisson(2*V[i], -2*E[i], N, h, eps)
    

                

        
        ################# Fock matrix ###################
        
        L = np.linalg.cholesky(qt.bilinear(psi_new, psi_new)*sf**2) # orthogonalization
        psi_Q = qt.UT_prod(psi_new, H(np.linalg.inv(L)), eps)
        
        # Fock matrix
        
        V_new = [0]*Norb
        for i in xrange(Norb):
            V_new[i] = prod(pot_V, psi_Q[i], psi_Q[i])
            psi_Q[i] = tuck.real(psi_Q[i])
            psi_Q[i] = tuck.round(psi_Q[i], eps)
    
        for i in xrange(Norb):
            Vxc_new = tuck.cross.multifun(psi_0 + [psi_Q[i]], 10*eps, lambda a: ((cpot_fun(density_fun(a[:-1])) + xpot_fun(density_fun(a[:-1])))*(a[-1])), y0 = psi_Q[i], pr=pr)
            V_new[i] = tuck.round(V_new[i] + Vxc_new, eps)
    
        
        Energ1 = np.dot(np.diag(E), H(np.linalg.inv(L)))
        Energ2 = qt.bilinear(psi_new, psi_new)*sf**2
        Energ2 = np.dot(np.linalg.inv(L), Energ2)
        Energ = np.dot(Energ2, Energ1)
        F = ((qt.bilinear(psi_Q, V_new)*sf**2 - np.dot(qt.bilinear(psi_Q, V)*sf**2, H(np.linalg.inv(L))) + Energ ))
        #print 'Fock Matrix:'
        #print F
        #F = np.real((F + F.T)/2)
        
        #E_new = np.zeros(Norb, dtype = np.float64)
        E_new, S = np.linalg.eigh(F)
        #S+= 1e-15
        #output_E[:Norb,k] = E_new
        #output_E[Norb,k] = E_full
        #output_r[:,k] = density.r
        #np.save('experiments/' + fname, [output_E[:,:k], output_r[:,:k]])
        
        #print 'Orbital Energies:'
        #print np.array(E_new)

        #psi = copy.copy(psi_Q)
        psi = qt.prod(psi_Q, S.T, density, eps)
        for i in xrange(Norb):
            psi[i] = qt.normalize(psi[i], h)
            psi[i] = tuck.round(psi[i], eps)
            
        #if np.linalg.norm(E - E_new)/np.linalg.norm(E_new) < 2*eps:
        #    break
        
        E = E_new.copy()
        

        for i in xrange(Norb):
            if E[i]>0:
                E[i] = -0.0
    

                    
    #density = tuck.zeros((N,N,N))  # density calculation
    #density = tuck.cross.multifun(psi, eps/10, lambda a: density_fun(a), y0 = psi_Q[i], pr=pr)

    return psi, E




def lda_full_energy(molecule, psi, E, pot_coulomb, grid, T, eps, ind, pr = None):
    
    a = grid[0]
    b = grid[1]
    N = grid[2]
    
    h = (b-a)/(N-1)
    
    x = np.zeros(N)
    for i in xrange(N):
        x[i] = a + i*h

    prod = lambda A,B,C: tuck.cross.multifun([A,B], eps, lambda (a,b): a*b, y0=C, pr=pr)

    sf = h ** (3./2) # scaling factor
    
    
    Norb = molecule.orbitals
    num_atoms = molecule.num_atoms

    density = psi2dens(psi, eps)

    pot_hartree = tuck.cross.conv(T, density, eps)
    pot_hartree = tuck.real(pot_hartree)
    pot_hartree = tuck.round(pot_hartree, eps)


    E_electr = 0
    e_xc = tuck.cross.multifun([density, density], eps, lambda (a, b): ((cen_fun(np.abs(np.real(a))) + xen_fun(np.abs(np.real(a))))*np.abs(b)), y0 = density, pr = pr)
    de_xc = tuck.cross.multifun([density, density],  eps, lambda (a, b): ((cpot_fun(np.abs(np.real(a))) + xpot_fun(np.abs(np.real(a))))*(b)), y0 = density, pr=pr)
    e_h = 0.5 * prod(pot_hartree, density, density)
    
    E_h = sf**2*tuck.dot(e_h, tuck.ones(e_xc.n, dtype=np.float64))
    E_xc = sf**2*tuck.dot(e_xc, tuck.ones(e_xc.n, dtype=np.float64))
    dE_xc = sf**2*tuck.dot(de_xc, tuck.ones(e_xc.n, dtype=np.float64))
    #print E_xc, dE_xc, E_h
    
    

    for i in xrange(Norb):
        E_electr += 2*E[i]

    E_nuc = 0
    for i in xrange(num_atoms):
        vec_i = molecule.atoms[i].rad
        charge_i = molecule.atoms[i].charge
        for j in xrange(i):
            vec_j = molecule.atoms[j].rad
            charge_j = molecule.atoms[j].charge
            E_nuc = (E_nuc + charge_i*charge_j/
                 np.sqrt((vec_i[0] - vec_j[0])**2 + (vec_i[1] - vec_j[1])**2 + (vec_i[2] - vec_j[2])**2))

    E_full = E_electr - E_h + E_nuc + E_xc - dE_xc

    print 'Full Energy = %s' % (E_full)
    print E_electr, - E_h + E_nuc + E_xc - dE_xc
    #accuracy[k] =  ((E_correct - E_full)/abs(E_correct))
    #print 'Relative Precision of the full Energy = %s' % (accuracy[k])

    return E_full



def cen_fun(a):
    return -3./4 * (3./pi)**(1./3) * a**(1./3)

def cpot_fun(a):
    return -(3./pi)**(1./3) * a**(1./3)

def xen_high_fun(rho, (A,B,C,D) = (A,B,C,D)):
    rs = (3./(4*pi*rho))**(1./3)
    return A*np.log(rs) + B + C*rs*np.log(rs) + D*rs

def xen_low_fun(rho, (gamma, beta1, beta2) = (gamma, beta1, beta2)):
    rs = (3./(4*pi*rho))**(1./3) #rho = 1/(4/3 * pi * rs^3)  1/rho = 4/3 pi R^3 R = (3/(4*pi*rho))^(1/3)
    return gamma/(1. + beta1 * np.sqrt(rs) + beta2 * rs)

def xpot_high_fun(rho, (A,B,C,D) = (A,B,C,D)):
    rs = (3./(4*pi*rho))**(1./3)
    return A*np.log(rs) + (B-A/3) + 2./3*C*rs*np.log(rs) + 1./3*(2*D-C)*rs

def xpot_low_fun(rho, (gamma, beta1, beta2) = (gamma, beta1, beta2)):
    rs = (3./(4*pi*rho))**(1./3)
    return gamma*(1. + 7./6 * beta1 * np.sqrt(rs) + 4./3 * beta2 * rs)/(1. + beta1 * np.sqrt(rs) + beta2 * rs)**2
#1./rs * gamma*(1./rs + 7./6 * beta1 * 1./np.sqrt(rs) + 4./3 * beta2 )/(1./rs + beta1 * 1./np.sqrt(rs) + beta2 )**2
#gamma*(1. + 7./6 * beta1 * np.sqrt(rs) + 4./3 * beta2 * rs)/(1. + beta1 * np.sqrt(rs) + beta2 * rs)**2

def xpot_fun(rho):
    
    rho_temp = rho.copy()
    rs = (3./(4*pi*rho))**(1./3)
    
    mask_high = rs<=1
    mask_low = rs>1

    np.place(rho_temp, mask_high, xpot_high_fun(rho[mask_high]))
    np.place(rho_temp, mask_low, xpot_low_fun(rho[mask_low]))

    return rho_temp

def xen_fun(rho):
    
    rho_temp = rho.copy()
    rs = (3./(4*pi*rho))**(1./3)
    
    mask_high = rs<=1
    mask_low = rs>1
    
    np.place(rho_temp, mask_high, xen_high_fun(rho_temp[mask_high]))
    np.place(rho_temp, mask_low, xen_low_fun(rho_temp[mask_low]))
    
    return rho_temp

def psi2dens(psi, eps):
    
    density = tuck.zeros(psi[0].n)
    for i in xrange(len(psi)):
        density = density + tuck.cross.multifun([psi[i]], eps/10, lambda a: a[0]**2, y0 = psi[i])
        density = tuck.round(density, eps/10)
    density = 2*density
    
    return density

def density_fun(psi):
    temp = np.array(psi)
    temp = np.abs(np.real(temp))
    return 2*np.sum(temp**2, axis=0)

def H(A):
    return np.transpose(np.conjugate(A))