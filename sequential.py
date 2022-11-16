#!/usr/bin/env python 

# imports
from ase.cluster.cubic import FaceCenteredCubic
from ase.visualize import view
from ase.optimize import BFGS
# from ase.calculators.emt import EMT
from asap3 import EMT
from ase.io.trajectory import Trajectory
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units
from ase.md.verlet import VelocityVerlet

# generate intitial strucuture
surfaces = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
layers = [1, 1, 1]
lc = 3.61  # WTF IS THIS?
atoms = FaceCenteredCubic("Cu", surfaces, layers, lc)

# create vacuum environment
atoms.center(vacuum=10.0)
# print(atoms)
pos = atoms.get_positions()
pos[4] = [9, 9, 9]
atoms.set_positions(pos)

# show atoms
# view(atoms)

# set atom calculator
atoms.calc = EMT()

e = atoms.get_potential_energy()
print("Potential energy =", e)

dyn = BFGS(atoms, trajectory='BFGS.traj')
dyn.run(fmax=0.05)

# trying to access trajectory
traj = Trajectory("BFGS.traj")
view(traj)

# set momenta of the atoms corresponding to temperature_K
MaxwellBoltzmannDistribution(atoms, temperature_K=300)

# We want to run MD with constant energy using the VelocityVerlet algorithm.
dyn = VelocityVerlet(atoms=atoms, timestep=5 * units.fs, trajectory="velocityVerlet.traj")  # 5 fs time step.

# Just a function that nicely print the energy
def printenergy(a):
    """Function to print the potential, kinetic and total energy"""
    epot = a.get_potential_energy() / len(a)
    ekin = a.get_kinetic_energy() / len(a)
    print('Energy per atom: Epot = %.3feV  Ekin = %.3feV (T=%3.0fK)  '
          'Etot = %.3feV' % (epot, ekin, ekin / (1.5 * units.kB), epot + ekin))


# Now run the dynamics
printenergy(atoms)
for i in range(20):
    dyn.run(10)  # take this many steps
    printenergy(atoms)

# IF YOU WANT tO PLAY THE MOVIE, type the following into terminal
# ase gui <>.traj