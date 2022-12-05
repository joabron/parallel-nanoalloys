# This script initializes a nanoalloys based on the users input

# imports
from ase.cluster.cubic import FaceCenteredCubic
from ase.visualize import view
import random as rd

A = 5
B = 5
atom_A = "Ni"
atom_B = "Fe"

surfaces = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
layers = [1, 1, 1]
lc = 3.61 
cluster = FaceCenteredCubic(atom_A, surfaces, layers, lc)

for i in range(B):
        index = rd.randint(0, A+B)
        cluster[index].symbol = atom_B

#print(cluster[0].symbol)
view(cluster[0:4])
view(cluster)