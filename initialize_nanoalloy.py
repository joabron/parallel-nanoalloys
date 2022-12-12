# This script initializes a nanoalloys based on the users input

# imports
from ase.cluster.cubic import FaceCenteredCubic
from ase.cluster.cubic import BodyCenteredCubic
from ase.visualize import view
import coreshell as cs
import random as rd

'''
This script contains functions for initializing nanoalloys in different ways. 

'''
#Make a function to do num_A = ... and num_B = ... in rectangle
#Make a function to build layered box
#Make a function to build sub-clustered
#Check if metals are fcc or bcc

def build_random_box(atom_A, atom_B, ratio, x, y, z):
    #The input must be odd numbers
    #Sometimes a even number of rows is returned?
    surfaces = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    x = int((x - 1) / 2)
    y = int((y - 1) / 2)
    z = int((z - 1) / 2)
    layers = [x, y, z]
    cluster = FaceCenteredCubic(atom_A, surfaces, layers)
    
    #swap atom_A for atom_B at random
    num_atoms = len(cluster)
    swap_atoms = int(ratio * num_atoms)
    i = 0
    while i < swap_atoms:
        index = rd.randint(0, num_atoms - 1)
        if cluster[index].symbol != atom_B:
            cluster[index].symbol = atom_B
            i += 1
    num_A = num_atoms - swap_atoms
    num_B = swap_atoms
    return cluster, num_A, num_B

def build_random_sphere(atom_A, atom_B, ratio, size):
    surfaces = [(1, 1, 1), (1, 1, 0), (1, 0, 0)]
    layers = [size-1, size, size-1]
    cluster = FaceCenteredCubic(atom_A, surfaces, layers)

    #swap atom_A for atom_B at random
    num_atoms = len(cluster)
    swap_atoms = int(ratio * num_atoms)
    i = 0
    while i < swap_atoms:
        index = rd.randint(0, num_atoms - 1)
        if cluster[index].symbol != atom_B:
            cluster[index].symbol = atom_B
            i += 1
    num_A = num_atoms - swap_atoms
    num_B = swap_atoms
    return cluster, num_A, num_B

def build_coreshell(atom_A, atom_B, ratio, size, lc, n=-1):
    surfaces = [(1, 1, 1), (1, 1, 0), (1, 0, 0)]
    layers = [size-1, size, size-1]
    cluster = FaceCenteredCubic(atom_A, surfaces, layers)
    cluster = cs.CoreShellFCC(cluster, atom_B, atom_A, ratio, lc, n_depth=n)
    return cluster

atoms = build_coreshell('Al', 'Pt', 0.6, 4, 4.09, n=1)
view(atoms)