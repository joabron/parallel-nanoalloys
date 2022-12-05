# This script initializes a nanoalloys based on the users input

# imports
from ase.cluster.cubic import FaceCenteredCubic
from ase.visualize import view

atom1 = "Ni"

surfaces = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
layers = [1, 1, 1]
lc = 3.61 
cluster = FaceCenteredCubic(atom1, surfaces, layers, lc)

view(cluster)