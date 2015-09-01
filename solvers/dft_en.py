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

def lda(molecule, psi_0, E_0, grid, T, num_iter, eps, ind):
    
    pr = None
    
    E_correct = molecule.energy
    
    eps_exchange = eps
    
    a = grid[0]
    b = grid[1]
    N = grid[2]
    
    h = (b-a)/(N-1)
    
    x = np.zeros(N)
    for i in xrange(N):
        x[i] = a + i*h
    
    accuracy = np.zeros(num_iter)
    timing = np.zeros(num_iter)
    
    sf = h ** (3./2) # scaling factor
    
    Norb = molecule.orbitals
    num_atoms = molecule.num_atoms

            ##### Output #####
    #output_E = np.zeros((Norb+1, num_iter))
    #output_r = np.zeros((3, num_iter))
    #fname = molecule.name + '_' + 'lda' + '_'  + str(N) + '_' + str(eps)
    
    ############################# Programm body ######################################
    

    V = [0]*Norb
    psi_new = [0]*Norb
    ################## Coulomb potential calculation ####################

    prod = lambda A,B,C: tuck.cross.multifun([A,B], eps, lambda (a,b): a*b, y0=C)
    
    E_nuc = 0
    for i in xrange(num_atoms):
        vec_i = molecule.atoms[i].rad
        charge_i = molecule.atoms[i].charge
        for j in xrange(i):
            vec_j = molecule.atoms[j].rad
            charge_j = molecule.atoms[j].charge
            E_nuc = (E_nuc + charge_i*charge_j/
                     np.sqrt((vec_i[0] - vec_j[0])**2 + (vec_i[1] - vec_j[1])**2 + (vec_i[2] - vec_j[2])**2))
    
    print "Coulomb potential..."
    
    pot_coulomb = tuck.zeros((N,N,N))
    for i in xrange(num_atoms):
        vec = molecule.atoms[i].rad
        charge = molecule.atoms[i].charge
        pot_coulomb = pot_coulomb - charge * qt.pots.coulomb(x, vec, ind, eps)
        pot_coulomb = tuck.round(pot_coulomb, eps)

    ################## Iteration process ####################
    print "Iteration process..."
    
    time_hartree = 0
    time_exchange = 0
    time_other = 0
    time_whole = 0
    time_iter = 0
    
    check = 0
    
    E_full = 0
    E_full_0 = 0
    E_full_1 = 1
    
    
    
    for k in xrange(num_iter):
        print "############################################ "
        print "Iteration Number %s" % (k + 1)
        print "############################################ "
        
        time_hartree_iter = 0
        time_exchange_iter = 0
        start_iter = time.time()
        
        start_hartree_iter = time.time()
        density = tuck.zeros((N,N,N))  # density calculation
        for i in xrange(Norb):
            density = density + prod(tuck.conj(psi[i]), psi[i], psi[i]) # !can be faster!
            density = tuck.round(density, eps)
        density = 2*density
        pot_hartree = tuck.cross.conv(T, density, eps,pr=None)
        pot_hartree = tuck.round(tuck.real(pot_hartree), eps)

    #density = tuck.real(density)
    #density = tuck.round(density , eps)
        end_hartree_iter = time.time()
        time_hartree_iter += (end_hartree_iter - start_hartree_iter)
        time_hartree += (end_hartree_iter - start_hartree_iter)

    
        E_electr = 0
        e_xc = tuck.cross.multifun([density, density], 3*eps, lambda (a, b): ((cen_fun(np.abs(np.real(a))) + xen_fun(np.abs(np.real(a))))*np.abs(b)), y0 = density, pr = pr)
        de_xc = tuck.cross.multifun([density, density],  3*eps, lambda (a, b): ((cpot_fun(np.abs(np.real(a))) + xpot_fun(np.abs(np.real(a))))*(b)), y0 = density, pr=pr)
        e_h = 0.5 * prod(pot_hartree, density, density)
        
        E_h = sf**2*tuck.dot(e_h, tuck.ones(e_xc.n))
        E_xc = sf**2*tuck.dot(e_xc, tuck.ones(e_xc.n))
        dE_xc = sf**2*tuck.dot(de_xc, tuck.ones(e_xc.n))
        #print E_xc, dE_xc, E_h
        
        
        for i in xrange(Norb):
            V[i] =  prod(tuck.round(pot_coulomb + pot_hartree, eps), psi[i], psi[i])
            psi[i] = tuck.real(psi[i])
            psi[i] = tuck.round(psi[i], eps)
        
        list = [None]*(Norb+1)
        list[1:] = psi
        for i in xrange(Norb):
            list[0] = psi[i]
            Vxc = tuck.cross.multifun(list, 10*eps, lambda (a): ((cpot_fun(density_fun(a[1:])) + xpot_fun(density_fun(a[1:])))*(a[0])), y0 = density, pr=pr) # Somehow it can not converge if precision eps. That is why 3*eps is used
            V[i] = tuck.round(V[i] + Vxc, eps)
            
            # Full energy computation
            #V_energ = prod(pot_hartree, psi[i], psi[i])
            E_electr += 2*E[i]# - sf**2*tuck.dot(psi[i], V_energ)/2 #- sf**2*tuck.dot(e_xc, tuck.ones(Exc.n))
            
            psi_new[i] = -qt.poisson(2*V[i], -2*E[i], N, h, eps)
    
        E_full_0 = E_full
        E_full = E_electr - E_h + E_nuc + E_xc - dE_xc
        E_full_1 = E_full
        print 'Full Energy = %s' % (E_full)
        print 'Correct Energy = %s' %(E_correct)
        accuracy[k] =  ((E_correct - E_full)/abs(E_correct))
        print 'Relative Precision of the full Energy = %s' % (accuracy[k])
                

        
        ################# Fock matrix ###################
        
        L = np.linalg.cholesky(qt.bilinear(psi_new, psi_new)*sf**2) # orthogonalization
        psi_Q = qt.UT_prod(psi_new, H(np.linalg.inv(L)), eps)
        
        # Fock matrix
        
        V_new = [0]*Norb
        for i in xrange(Norb):
            V_new[i] = prod(tuck.round(pot_coulomb + pot_hartree,eps), psi_Q[i], psi_Q[i])
            psi_Q[i] = tuck.real(psi_Q[i])
            psi_Q[i] = tuck.round(psi_Q[i], eps)
    
        list = [None]*(Norb+1)
        list[1:] = psi_Q
        for i in xrange(Norb):
            list[0] = psi_Q[i]
            Vxc_new = tuck.cross.multifun(list, eps*10, lambda (a): ((cpot_fun(density_fun(a[1:])) + xpot_fun(density_fun(a[1:])))*(a[0])), y0 = density, pr = pr)
            V_new[i] = tuck.round(V_new[i] + Vxc_new, eps)
    
    
        Energ1 = np.dot(np.diag(E), H(np.linalg.inv(L)))
        Energ2 = qt.bilinear(psi_new, psi_new)*sf**2
        Energ2 = np.dot(np.linalg.inv(L), Energ2)
        Energ = np.dot(Energ2, Energ1)
        F = qt.bilinear(psi_Q, V_new)*sf**2 - np.dot(qt.bilinear(psi_Q, V)*sf**2, H(np.linalg.inv(L))) + Energ
        print 'Fock Matrix:'
        print F
        #F = np.real((F + F.T)/2)
        
        E_new = np.zeros(Norb, dtype = np.complex128)
        E_new, S = np.linalg.eigh(F)
        
        output_E[:Norb,k] = E_new
        output_E[Norb,k] = E_full
        output_r[:,k] = density.r
        np.save('experiments/'+fname, [output_E[:,:k], output_r[:,:k]])
        
        print 'Orbital Energies:'
        print np.array(E_new)
        
        psi = qt.prod(psi_Q, S.T, eps)
        for i in xrange(Norb):
            psi[i] = qt.normalize(psi[i], h)
            psi[i] = tuck.round(psi[i], eps)
        
        E = E_new.copy()
        
        for i in xrange(Norb):
            if E[i]>0:
                E[i] = -0.0
    
        end_iter = time.time()
        print '1 Iteration Time: %s' % (end_iter - start_iter)
        print 'Hartree Time on this Iteration: %s' % (time_hartree_iter)
                    #print 'Exchange Time on this Iteration: %s' % (time_exchange_iter)
        time_whole += (end_iter - start_iter)
        timing[k] = end_iter - start_iter
            
        if abs( (E_full_1-E_full_0)/E_full_1 ) <= 10 * eps:
            check += 1
            if check == 3:
                break

    print 'The Whole Time: %s' % (time_whole)                
    return psi, E, accuracy, timing, E_full



def lda_full_energy(molecule, psi, E, grid, T, eps, eps_exchange):
    
    a = grid[0]
    b = grid[1]
    N = grid[2]
    
    h = (b-a)/(N-1)
    
    sf = h ** (3./2)
    
    Norb = molecule.orbitals
    num_atoms = molecule.num_atoms
    
    prod = lambda A,B: tuck.cross.multifun([A,B], eps, lambda (a,b): a*b)
    
    density = tuck.zeros((N,N,N))  # density calculation
    for i in xrange(Norb):
        density = density + prod(psi[i], psi[i], psi[i]) # !can be faster!
        density = tuck.round(density, eps)
    
    pot_hartree = tuck.cross.conv(T, density, eps)
    
    V = [0] * Norb
    for i in xrange(Norb):
        V[i] =  (prod(tuck.round(2*pot_hartree, eps), psi[i], psi[i]) -
                 qt.pots.exchange(psi, psi, i, eps_exchange, T, molecule))
        V[i] = tuck.round(V[i], eps)
    
    
    E_electr = 0
    for i in xrange(Norb):
        E_electr = E_electr + 2*E[i] - sf**2*tuck.dot(psi[i], V[i])
    
    
    E_nuc = 0
    for i in xrange(num_atoms):
        vec_i = molecule.atoms[i].rad
        charge_i = molecule.atoms[i].charge
        for j in xrange(i):
            vec_j = molecule.atoms[j].rad
            charge_j = molecule.atoms[j].charge
            E_nuc = (E_nuc + charge_i*charge_j/
                     np.sqrt((vec_i[0] - vec_j[0])**2 + (vec_i[1] - vec_j[1])**2 + (vec_i[2] - vec_j[2])**2))
    
    return E_nuc + E_electr


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

def density_fun(psi):
    temp = np.array(psi)
    temp = np.abs(np.real(temp))
    return 2*np.sum(temp**2, axis = 0)

def H(A):
    return np.transpose(np.conjugate(A))