#!/usr/bin/env python 

import initialize_nanoalloy as init
from replica_exchange_serial import ReplicaExchange
from asap3 import EMT
import pandas as pd

print("got inside run_replica")
input_file = pd.read_csv("inputs.txt", delimiter=":", skiprows = [0,1,2,3,5,6,12,13,16,17,22,23,29,30], header=None, skipinitialspace=True)
print("File has been read")
choice = int(input_file.iloc[0, 1])
if choice == 0:
    atom_A, atom_B, ratio, size, lc = input_file.iloc[1:6, 1]
    ratio = float(ratio)
    size = int(size)
    lc = float(lc)
    atoms = init.build_coreshell(atom_A, atom_B, ratio, size, lc)

elif choice == 1:
    atom, size = input_file.iloc[6:8, 1]
    size = int(size)
    atoms = init.build_pure_sphere(atom, size)

elif choice == 2:
    atom_A, atom_B, num_A, num_B = input_file.iloc[8:12, 1]
    num_A = int(num_A)
    num_B = int(num_B)
    atoms = init.build_num_atoms(atom_A, atom_B, num_A, num_B)

atoms.cell = [5, 5, 5, 90, 90, 90]
print("Starting EMT")
atoms.calc = EMT()

num_rep = int(input_file.iloc[17, 1])
start_temp = int(input_file.iloc[18, 1])
end_temp = int(input_file.iloc[19, 1])

print("starting RE")
my_RE = ReplicaExchange(num_rep, start_temp, end_temp, atoms)

num_swap = int(input_file.iloc[20, 1])
num_MDrun = int(input_file.iloc[21, 1])

print("Starting serial run")
my_RE.run_serial_RE(num_swap, num_MDrun)
