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





