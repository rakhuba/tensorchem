import numpy as np
class proto:
    pass

def molecule(name, charge = 0, rad = 0): # Warning: rad must be in angstrem
                                            # possible molecules: H2O,
                                            # if name = '' => manual setting of charge and rad

    const = 1./0.52917721092  # angstrem to bohr constant

    molecule = proto()
    
    if name == 'He':

        charge = [2.]
        rad = np.array([0.000000,  0.000000,  0.000000])
        molecule.energy = -2.861679995612

    if name == 'Li':

        charge = [3.]
        rad = np.array([0.000000,  0.000000,  0.000000])
        molecule.energy = -7.236384

    elif name == 'H2':

        charge = [1., 1.]
        rad = np.array([0.000000,  0.000000,  0.3669,
                        0.000000,  0.000000,  -0.3669])
        molecule.energy = -1.1336286746

    elif name == 'HeH+':

        charge = [2., 1.]
        rad = np.array([0.000000,  0.000000,  0.2567,
                        0.000000,  0.000000,  -0.5134])
        molecule.energy = -2.932879

    elif name == 'Be':

        charge = [4.]
        rad = np.array([0.000000,  0.000000,  0.000000])
        molecule.energy = -14.57302317

    elif name == 'Ne':

        charge = [10.]
        rad = np.array([0.000000,  0.000000,  0.000000])
        molecule.energy = -128.5470981

    elif name == 'Mg':

        charge = [12.]
        rad = np.array([0.000000,  0.000000,  0.000000])
        molecule.energy = -199.6146364

    elif name == 'Ar':

        charge = [18.]
        rad = np.array([0.000000,  0.000000,  0.000000])
        molecule.energy = -526.8175128

    elif name == 'H2O':

        charge = [8., 1., 1.]
        #rad = np.array([0.000000,  0.000000,  0.000000,
        #                0.758602,  0.000000,  0.504284,
        #                0.758602,  0.000000,  -0.504284])
        rad = np.array([0.000, 0.000, 0.113,
                        0.000,	0.752,	-0.451,
                        0.000,	-0.752,	-0.451])
        molecule.energy = -76.066676

    elif name == 'CH4':

        charge = [6., 1., 1., 1., 1.]
        rad = np.array([0.0000, 0.0000, 0.0000,
                        0.6248,	0.6248,	0.6248,
                        -0.6248, -0.6248, 0.6248,
                        -0.6248, 0.6248, -0.6248,
                         0.6248, -0.6248, -0.6248])
        molecule.energy = -40.216345

    elif name == 'C2H6':

        charge = [6., 6., 1., 1., 1., 1., 1., 1.]
        rad = np.array([0.0000,	0.0000,	0.7623,
                        0.0000,	0.0000,	-0.7623,
                         0.0000, 1.0108, 1.1544,
                        -0.8754, -0.5054, 1.1544,
                         0.8754, -0.5054, 1.1544,
                         0.0000, -1.0108, -1.1544,
                         -0.8754, 0.5054, -1.1544,
                        0.8754,	0.5054,	-1.1544])
        molecule.energy = -79.265431

    elif name == 'C6H6':
        
        charge = [6., 6., 6., 6., 6., 6., 1., 1., 1., 1., 1., 1.]
        rad = np.array([ 0.0, 2.61474609375, 0.0,
                        0.0, -2.61474609375, 0.0,
                        2.26318359375, 1.318359375, 0.0,
                        2.26318359375, -1.318359375, 0.0,
                        -2.26318359375, 1.318359375, 0.0,
                        -2.26318359375, -1.318359375, 0.0,
                         0.0, 4.658203125, 0.0,
                        0.0, -4.658203125, 0.0,
                        4.02099609375, 2.3291015625, 0.,
                        4.02099609375, -2.3291015625, 0.,
                        -4.02099609375, 2.3291015625, 0.,
                        -4.02099609375, -2.3291015625, 0.,
                        ])
        rad = rad /const
        molecule.energy = -230.2017048

    elif name == 'C2H5OH':

        charge = [6., 6., 8., 1., 1., 1., 1., 1., 1]
        rad = np.array([1.1657,	-0.4170, 0.0000,
                        0.0000,	0.5466,	0.0000,
                         -1.1908, -0.1944, 0.0000,
                        -1.9328, 0.3795, 0.0000,
                         2.1038, 0.1251, 0.0000,
                         1.1335, -1.0501, 0.8773,
                         1.1335, -1.0501, -0.8773,
                        0.0471,	1.1868,	0.8767,
                         0.0471, 1.1868, -0.8767])
        molecule.energy = -154.155574


    else:   # manual setting of molecule
        molecule.name = 'manual setting'


    ####################################################################

    if 3*len(charge) <> len(rad):
        raise Exception('charge and coordinate arrays do not coincide each other')

    
    num_atoms = len(charge)
    molecule.num_atoms = num_atoms
    molecule.name = name
    molecule.atoms = [proto() for i in range(num_atoms)]

    for i in xrange(num_atoms):
        molecule.atoms[i].charge = charge[i]
        
    molecule.Ne = 0
    for i in xrange(num_atoms):
        molecule.Ne += molecule.atoms[i].charge            
    molecule.orbitals = int(molecule.Ne/2) # closed-shell

    rad = np.reshape(rad, (num_atoms, 3))
    for i in xrange(num_atoms):
        molecule.atoms[i].rad = const * rad[i, :]

    return molecule
