# This script initializes a nanoalloys based on the users input

# imports
from ase.cluster.cubic import FaceCenteredCubic
from ase.cluster.cubic import BodyCenteredCubic
from ase.visualize import view
import random as rd

'''
This script contains functions for initializing nanoalloys in different ways. 

'''
#Make a function to do num_A = ... and num_B = ... in rectangle
#Make a function to build core-shell
#Make a function to build layered cube
#Make a function to build sub-clustered
#Check if metals are fcc or bcc

def build_random_cube(atom_A, atom_B, ratio, size):
    surfaces = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    size = int((size - 1)/2)
    layers = [size, size, size]
    cluster = FaceCenteredCubic(atom_A, surfaces, layers)
    num_atoms = len(cluster)
    swap_atoms = int(ratio * num_atoms)
    i = 0
    while i < swap_atoms:
        index = rd.randint(0, num_atoms - 1)
        if cluster[index].symbol != atom_B:
            cluster[index].symbol = atom_B
            i += 1
    return cluster

def build_random_sphere(atom_A, atom_B, ratio, size):
    symbols = atom_A
    surfaces = [(1, 1, 1), (1, 1, 0), (1, 0, 0)]
    layers = [size-1, size, size-1]
    cluster = FaceCenteredCubic(atom_A, surfaces, layers)
    num_atoms = len(cluster)
    swap_atoms = int(ratio * num_atoms)
    i = 0
    while i < swap_atoms:
        index = rd.randint(0, num_atoms - 1)
        if cluster[index].symbol != atom_B:
            cluster[index].symbol = atom_B
            i += 1
    return cluster