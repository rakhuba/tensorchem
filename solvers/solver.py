import numpy as np
import time
from math import pi
import copy
from scipy.special import erf
import tucker3d as tuck
import qtools as qt
import solvers

class proto:
    pass

class solver:

    def __init__(self, molecule, method='hf', eps=1e-6,
                 maxiter=50, max_iter_inscf=1, meshsize=1024, boxsize=10., mixing=None,
                 pr=None, psi0=None, orb_energies0=None):
        
        self.molecule = molecule
        self.method = method
        
        self.params = proto()
        self.pots = proto()
        self.iterative = proto()
        
        self.params.eps = eps
        self.params.maxiter = maxiter
        self.params.meshsize = meshsize
        self.params.boxsize = boxsize
        self.params.mixing = mixing
        self.params.max_iter_inscf = max_iter_inscf
        self.params.grid = np.linspace(-boxsize, boxsize, meshsize)
        self.params.pr = pr
        
        if psi0 == None and orb_energies0 == None:
            self.psi, self.orb_energies = solvers.pyquante(molecule, self.params.grid, eps)
            self.energy = None
        else:
            self.psi = psi0
            self.orb_energies = orb_energies0
            self.energy = None
                
        if eps >= 1e-6:
            self.params.ind = 6
        elif eps >= 1e-8:
            self.params.ind = 8
        elif eps >= 1e-10:
            self.params.ind = 10
        else:
            self.params.ind = 12

        galerkin_kernel = qt.galerkin.newton(self.params.grid, eps, self.params.ind)
        self.params.galerkin_kernel = tuck.cross.toepl2circ(galerkin_kernel)
        self.get_coulomb()
                
    def solve(self):

        if self.method == 'hf':
            solvers.solver_hf(self, self.molecule, self.psi, self.orb_energies, self.params.eps,
                       [-self.params.boxsize, self.params.boxsize, self.params.meshsize],
                       self.params.galerkin_kernel, self.params.ind, self.params.maxiter, self.pots.coulomb)
        
        elif self.method == 'lda':
            solvers.solver_lda(self, self.molecule, self.psi, self.orb_energies, 
                              [-self.params.boxsize, self.params.boxsize, self.params.meshsize],
                               self.params.galerkin_kernel, self.params.ind,
                               self.params.eps, self.pots.coulomb,
                               self.params.maxiter, self.params.max_iter_inscf)

        elif self.method == 'mixing':
            solvers.mixing(self, self.molecule, self.psi, self.orb_energies, self.params.eps,
                           [-self.params.boxsize, self.params.boxsize, self.params.meshsize], self.params.galerkin_kernel, self.params.ind, self.params.maxiter, self.params.mixing, self.params.max_iter_inscf,  self.params.pr)
            

    def get_coulomb(self):
        molecule = self.molecule
        x = self.params.grid
        N = len(self.params.grid)
        Norb = self.molecule.orbitals
        num_atoms = self.molecule.num_atoms
        
        pot_coulomb = tuck.zeros((N, N, N))
        for i in xrange(num_atoms):
            vec = molecule.atoms[i].rad
            charge = molecule.atoms[i].charge
            pot_coulomb = pot_coulomb - charge * qt.pots.coulomb(x, vec, self.params.ind, self.params.eps)
            pot_coulomb = tuck.round(pot_coulomb, self.params.eps)
        self.pots.coulomb = pot_coulomb


    def gradient_diatomic(self):

        # pot squared

        molecule = self.molecule
        x = self.params.grid
        N = len(self.params.grid)
        Norb = self.molecule.orbitals
        num_atoms = self.molecule.num_atoms
        
        d_coulomb = tuck.zeros((N, N, N))
        tensor_x = tuck.ones((N, N, N))
        tensor_x.u[0] = tensor_x.u[0] * x
        for i in xrange(num_atoms):
            vec = molecule.atoms[i].rad
            charge = molecule.atoms[i].charge
            tensor_x = charge*tuck.round(vec[0]*tuck.ones((N, N, N)) - tensor_x, self.params.eps)
            pot_squared = qt.pots.coulomb(x, vec, self.params.ind, self.params.eps, beta=2.0)
            d_coulomb = tuck.round(d_coulomb, self.params.eps)
        
            d_coulomb -= tuck.cross.multifun([tensor_x, pot_squared], self.params.eps, lambda x: x[0]*x[1])

        self.get_rho()
        
        
        
        dE_nuc = 0.0
        for i in xrange(num_atoms):
            vec_i = molecule.atoms[i].rad
            charge_i = molecule.atoms[i].charge
            for j in xrange(i):
                vec_j = molecule.atoms[j].rad
                charge_j = molecule.atoms[j].charge
                dE_nuc = (dE_nuc - (vec_i[0] - vec_j[0])*charge_i*charge_j/
                np.sqrt((vec_i[0] - vec_j[0])**2 + (vec_i[1] - vec_j[1])**2 + (vec_i[2] - vec_j[2])**2)**3)


    def get_rho(self):

        self.rho = tuck.zeros(self.phi[0].n)
        for i in xrange(len(self.phi)):
            self.rho += tuck.cross.multifun([self.phi[i]], self.params.eps, lambda x: x[0]**2)
            self.rho = tuck.round(self.rho, self.params.eps)










