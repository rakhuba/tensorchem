import numpy as np
import tucker3d as tuck
import os
import copy

def write_psi(molecule_name,folder,psi):
    
    norb = len(psi)
    #g = [None]*norb
    
    for i in xrange(norb):
        g = psi[i].core
        u1 = psi[i].u[0]
        u2 = psi[i].u[1]
        u3 = psi[i].u[2]
        
        np.save(folder+'/' + molecule_name + '_' +str(i)+ '_' + 'u1',u1)
        np.save(folder+'/' + molecule_name + '_' +str(i)+ '_' + 'u2',u2)
        np.save(folder+'/' + molecule_name + '_' +str(i)+ '_' + 'u3',u3)
        np.save(folder+'/' + molecule_name + '_' +str(i)+ '_' + 'g', g)
    
    return

def read_psi(molecule_name, folder):
    
    #g = np.load(folder+'/'  + molecule_name + '_' + 'g.npy')
    #norb = len(g)
    
    psi = []
    i = 0
    while True:
        try:
            psi.append(tuck.tensor())
            psi[i].core = np.load(folder+'/' + molecule_name + '_' +str(i)+ '_'  + 'g.npy')
        
            psi[i].u[0] = np.load(folder+'/' + molecule_name + '_' +str(i)+ '_'  + 'u1.npy')
            psi[i].u[1] = np.load(folder+'/' + molecule_name + '_' +str(i)+ '_'  + 'u2.npy')
            psi[i].u[2] = np.load(folder+'/' + molecule_name + '_' +str(i)+ '_'  + 'u3.npy')
        
            psi[i].n = (len(psi[i].u[0]), len(psi[i].u[1]), len(psi[i].u[2]))
            psi[i].r = copy.copy(psi[i].core.shape)
            i += 1
        except:
            break

    return psi[:-1]


def write_psi_old(molecule_name,folder,psi):

    norb = len(psi)
    g = [None]*norb
    
    for i in xrange(norb):
        g[i] = psi[i].core
        u1 = psi[i].u[0]
        u2 = psi[i].u[1]
        u3 = psi[i].u[2]
        
        np.save(folder+'/' + molecule_name + '_' +str(i)+ '_' + 'u1',u1)
        np.save(folder+'/' + molecule_name + '_' +str(i)+ '_' + 'u2',u2)
        np.save(folder+'/' + molecule_name + '_' +str(i)+ '_' + 'u3',u3)
    
    np.save(folder+'/' + molecule_name + '_' + 'g',g)
    
    return

def read_psi_old(molecule_name, folder):

    g = np.load(folder+'/'  + molecule_name + '_' + 'g.npy')
    norb = len(g)
    
    psi = [None]*norb
    for i in xrange(norb):
        psi[i] = tuck.tensor()
        psi[i].core = g[i]

        psi[i].u[0] = np.load(folder+'/' + molecule_name + '_' +str(i)+ '_'  + 'u1.npy')
        psi[i].u[1] = np.load(folder+'/' + molecule_name + '_' +str(i)+ '_'  + 'u2.npy')
        psi[i].u[2] = np.load(folder+'/' + molecule_name + '_' +str(i)+ '_'  + 'u3.npy')
    
        psi[i].n = (len(psi[i].u[0]), len(psi[i].u[1]), len(psi[i].u[2]))
        psi[i].r = copy.copy(psi[i].core.shape)

    return psi
       
    
