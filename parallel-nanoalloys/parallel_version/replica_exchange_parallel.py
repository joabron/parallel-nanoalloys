#!/usr/bin/env python3

# [Add description of code here]

#%%
# =============================================================================
# Imports and constants
# =============================================================================
import math 
import numpy as np
from asap3.md.langevin import Langevin              # MD
from ase.io.trajectory import Trajectory            # write to trajectory
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units                               # for timesteps
import pandas as pd
# from ase.calculators.emt import EMT
# import initialize_nanoalloy as nanoalloy
from mpi4py import MPI

#%% 
# =============================================================================
# Functions
# =============================================================================

def swap_positions(list, pos1, pos2):
    ''' Swap two positions in list 

        INPUTS:
            list       :    python list
            pos1, pos2 :    indices of elements to swapped in list
        
        OUTPUTS:
            list       :    returns the same list
    '''
    list[pos1], list[pos2] = list[pos2], list[pos1]
    return list

# =============================================================================
# Replica Exchange
# =============================================================================
class Replica:
    ''' Replica class for individual replicas during Replica Exchange
        
        The Replica class logs data from the MD simulations in a dataframe,
        and writes it to a csv of name: 'replica_idx.csv', for replica idx.
        
        Example of the format of the csv log file for a replica with idx = 1:
        1, 1, 1, 1
        # time,  energy,     beta,   target_beta
        10.0, -0.68626290, 301, 300,
        20.0, -0.66438440, 299, 300,
        30.0, -0.71729520, 402, 400,

        The Replica class also saves a 'replica_idx.traj' file with a
        trajectory of the MD simulation.
        
        Class functions: 
            MD_run(md_num_steps) : run Molecular Dynamics using Langevin
                                   1 md_num_steps = 3 fs of MD simulation
    '''
    
    def __init__(self, rank, target_T, atom_config):
        ''' Initialize instance of the Replica Class '''
        
        ############# BASIC ATTRIBUTES
        print("proc ",rank," created a replica")
        self.rank = rank                    # replica index
        self.configuration = atom_config    # replica atom configuration
        self.target_T = target_T            # target temperature in [K]
        self.T = target_T                   # actual temperature in [K]
        self.own_temperature_idx = rank        # index of the temperature, initially rank, but subject to swap
        
        ############# LANGEVIN
        MaxwellBoltzmannDistribution(self.configuration,temperature_K=self.target_T) #trying to set initial temperature
        self.MD = Langevin( atoms           = self.configuration,                  
                            timestep        = 3*units.fs, 
                            temperature_K   = target_T, 
                            friction        = 1/2
                            #trajectory      = "replica_"+str(self.rank)+".traj"
                            )

        ############# TRAJECTORY OUTPUT
        traj = Trajectory("replica_"+str(self.rank)+".traj", 'a', self.configuration, master=True)
        self.MD.attach(traj.write, interval=2)
        
        ############# LOGGER
        self._i = 1
        self.frame = pd.DataFrame(columns=["time[ps]", "epot/N[eV]", "beta[1/eV]", "target_beta[1/eV]"])
        
        self._times = [self.rank]            # list of times for csv files
        self._epot = [self.rank]             # list of energies for csv files
        self._betas = [self.rank]            # list of betas for csv files
        self._target_betas = [self.rank]     # list of target betas for csv files
        
        # logger print function
        def _save_log(a=self):
            if self._i >= 1:   # TODO try with and without this
                self._times.append(a.MD.get_time() / (1000 * units.fs))
                self._epot.append(a.configuration.get_potential_energy() / len(a.configuration))
                self._betas.append(1/(a.configuration.get_temperature()*units.kB))                     
                #self._betas.append(1/(a.target_T*units.kB)) # note that configuration beta is now simply set to target beta
                self._target_betas.append(1/(a.target_T*units.kB))

        def _increase_i():
            self._i += 1
        
        self.MD.attach(_save_log, interval=1)
        self.MD.attach(_increase_i, interval=1)

        # See if Maxwell function correctly sets the temp
        print(f"temp of replica {self.rank} = {self.configuration.get_temperature()}")   

    @property
    def Beta(self):
        ''' Compute the inverse temperature, Beta = 1/(T*k_B) [1/eV] '''
        return 1/(self.T * units.kB)
    
    @property
    def U(self):
        ''' Compute the replica's potential energy, U [eV] '''
        return self.configuration.get_potential_energy()
    
    def _reset_i(self):
        self._i = 1

    def _produce_csv(self):
        self.frame["time[ps]"] = self._times
        self.frame["epot/N[eV]"] = self._epot  # is now in eV
        self.frame["beta[1/eV]"] = self._betas
        self.frame["target_beta[1/eV]"] = self._target_betas
        self.frame = self.frame.drop([i for i in range(1,51)],axis=0)

        output_filename = str('replica_'+str(self.rank)+'.csv')
        self.frame.to_csv(output_filename, header=False, index=False)
    
    def run_MD(self, md_num_steps): # [atm keeping print statements for tests]
        ''' Perform MD simulation and log the new configuration data
            
            INPUTS:
            md_num_steps :    number of MD steps to run 
        '''
        
        self.MD.run(md_num_steps)   # the MD simulation
        
        
        # [for now we print these things to discsuss them with Palagin]
        print(" epot[eV]            = ", str(self.configuration.get_potential_energy() / len(self.configuration)))
        print(" ekin[eV]            = ", str(self.configuration.get_kinetic_energy() / len(self.configuration)))
        
        print(" T[K]                = ", 2*self.configuration.get_kinetic_energy() 
                                        / (3 * units.kB*len(self.configuration)))
        
        print(" mean abs velocity   = ", np.sqrt(np.mean((self.configuration.get_velocities()**2))),"\n")
        

class ReplicaExchange: # [clean up]
    ''' TODO Replica Exchange class for running the replica exchange scheme.
        
        The Replica Exchange class 
        
        Example of the format of the csv log file for a replica with idx = 1:
        1, 1, 1, 1
        # time,  energy,     beta,   target_beta
        10.0, -0.68626290, 301, 300,
        20.0, -0.66438440, 299, 300,
        30.0, -0.71729520, 402, 400,

        The Replica class also saves a 'replica_idx.traj' file with a
        trajectory of the MD simulation.
        
        Class functions: 
            MD_run(md_num_steps) : run Molecular Dynamics using Langevin
                                   1 md_num_steps = 3 fs of MD simulation
    '''
    def __init__(self, num_replicas, min_temperature, max_temperature, 
                         initial_config, type_of_swapping='one_at_a_time', 
                         order_of_swapping = 'odd_even'):
        ''' TODO Initialize a replica exchange (RE) instance with its replicas from 
        class Replica. 
        
        min_temperature :       lowest temperature in the RE
        max_temperature :       highest temperature in the RE
        initial_configs :       a list specifying the intial configurations 
                                of the replicas
        type_of_swapping :      'one_at_a_time' (default, sequential) or 
                                'simultaneous' (parallel, not currently implemented)
        ordering_of_swapping :  'odd_even' (default) or 'random pairs' 
                                (currently not implemented)
        
        '''
        
        # Initialize replicas
        # Check if initial configs is a list or not. If only one initial_config 
        # is passed, initialize all the replicas with the same initial_config. 
        # If a list is passed, check if list has the correct length (same number 
        # as replicas)

        # Initializing MPI
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        print("rank = ",self.rank)
        if self.rank == 0:
            print("size",self.size)

        self.num_replicas = num_replicas

        # set number of replicas to size
        if self.size != self.num_replicas:
            print("For now, the number of replicas has to be equal to the comm size")
            self.num_replicas = self.size
        
        self.type_of_swapping   = type_of_swapping
        
        # List with the temperatures that are implemented
        self.temperatures       = [min_temperature + (max_temperature-min_temperature)
                                   /(num_replicas-1)*i for i in range(num_replicas)]
        
        # Temperature idx list to keep track of the order of the replicas and at
        # what temperatures they currenlty are. The numbers correspond to the 
        # replica idx and the position in the list correspond to the temperature. 
        # So if temperature_idx is [2, 0, 1], replica 2 is at temperature 0, 
        # replica 0 is at temperature 1, etc. 
        self.temperature_idx    = [i for i in range(self.num_replicas)]
            
        self.replica = Replica(rank=self.rank,
                                 target_T=self.temperatures[self.rank],
                                 atom_config=initial_config) 

        # self.replicas = self.comm.gather(self.replica, root=0)
        
        # if self.replica.rank != 0:
        #     self.replicas = [0 for _ in range(self.num_replicas)]
        
        # self.comm.Bcast(self.replicas,root=0)


        # Initialize accepted/rejected statistics for computing RE efficiency 
        # swap_log tracks which pairs were attempted to swap. It is a 2D list, 
        # where each inner entry is a swap attempt. The inner swap attempt 
        # structure is as follows: [temperature_of_replica_before_swap_1, 
        # temperature_of_replica_before_swap_2, replica_1, replica_2, bool 0 
        # if rejected and 1 if accepeted]
        # TODO Didn't implement this
        self.accepted = 0
        self.rejected = 0
        self.swap_log = []   
        
    @property
    def total_swap_attempts(self):
        """ Return the total amount of swap attempts """
        return self.accepted + self.rejected
    
    @property
    def swapping_rate(self):
        pass #implement this at a later time
    
    @property
    def efficiency(self):
        """ Calculate the efficicency (ratio of accepted:rejected) of the RE """
        try:
            return self.accepted/self.rejected
        except ZeroDivisionError:
            print("There were no swaps rejected (efficiency undefined because \
                  of division by zero!)") 
    
    def calc_swap_prob(self, beta_a, beta_b, U_a, U_b): 
        """ Calculate swapping probability p of replica pair A and B, where beta 
        is the inverse temperature 1/(T*kB) and U is the potential energy"""
        p = math.exp((beta_a-beta_b)*(U_a - U_b))   # probability of accepting swap 
        p_condition = np.random.uniform()           # realization from Uniform(0,1)
        accept_swap = (p_condition < p)             # Swap condition criterion
        return accept_swap, p

    def pairs_to_swap(self, number_of_swaps,  type_of_swapping='one_at_a_time', order_of_swapping = 'odd_even'):
        pass #implmenet later if time allows. implement individual swaps instead of swap phases

    def generate_replica_temperatures(self):
        """ Generate a list of the target tempeartures of each replica. The positions
        in the list correspond to the respective replica target temperature, 
        e.g. position 0 is the target temperature of replica 0, etc. Used for testing 
        purposes (mostly)"""
        return [self.replicas[i].target_T for i in range(self.num_replicas)]
    
    def attempt_swap(self, swap_partner_idx):
        """ Attempt to swap replica A and B by choosing the temperatures indices 
        of temperature_idx that will be swapped. So if the inputs are 0 and 1, 
        we swap the replica that currenlty has tempeareture 0 with the one that 
        currenlty has temperature 1"""

        rank_partner = self.temperature_idx[swap_partner_idx]
        print("my rank and rank_partner", self.rank, rank_partner)
        
        if self.rank < rank_partner:                                
            print("Rank ", self.rank, " is receiving U")
            U_partner = self.comm.recv(source=rank_partner,tag=0) # Receive U from partner
            print("Rank ", self.rank, " received U")
            # compute probability
            accept_swap, p = self.calc_swap_prob(self.replica.Beta, 1/(self.temperatures[swap_partner_idx]*units.kB), self.replica.U, U_partner)
            print("Rank ", self.rank, "is sending accept")
            self.comm.send(accept_swap, dest=rank_partner,tag=0)
            print("Rank ", self.rank, "sent accept")
        else:
            print("Rank ", self.rank, " is sending U")
            self.comm.send(self.replica.U,dest=rank_partner,tag=0) # send U to partner
            print("Rank ", self.rank, " sent U")

            print("Rank ", self.rank, "is receiving accept")
            accept_swap = self.comm.recv(source=rank_partner,tag=0)
            print("Rank ", self.rank, "received accept")

        # If accepted swap
        if accept_swap:
            print("Rank", self.rank, "is swapping with ", rank_partner)
            # only one of the ranks should update
            if self.rank < rank_partner:
                print(f"Replica {self.replica.rank}: {self.replica.target_T:>6.1f}  –—    ––>  {self.temperatures[swap_partner_idx]:5.1f}")
                print(f"                      \/ ")
                print(f"                      /\ ")
                print(f"Replica {rank_partner}: {self.temperatures[swap_partner_idx]:>6.1f}  –—    ––>  {self.replica.target_T:5.1f}")
            
            # set new target temperatures
            self.replica.target_T = self.temperatures[swap_partner_idx] # swap/set replica temperature

            # set temperatures for MD
            self.replica.MD.set_temperature(self.replica.target_T*units.kB)         # set the temperature of the replica MD
            
            # scale velocities
            self.replica.configuration.set_velocities(self.replica.configuration.get_velocities()*np.sqrt(self.replica.Beta/(1/(self.temperatures[swap_partner_idx]*units.kB))))
            
            self.replica.own_temperature_idx = swap_partner_idx

            # Add one to 'accepted' list
            self.accepted += 1

        else:
            
            # printing swap visually
            print("Rank ", self.rank, "rejected with ", rank_partner)
            if self.rank < rank_partner:
                print(f"Replica {self.replica.rank}: {self.replica.target_T:>6.1f}  ––––––––>  {self.replica.target_T:5.1f}")
                print(" ")
                print(" ")
                print(f"Replica {rank_partner}: {self.temperatures[swap_partner_idx]:>6.1f}  ––––––––>  {self.temperatures[swap_partner_idx]:5.1f}")
            
            self.rejected += 1
        
    def run_parallel_RE(self, number_of_swap_phases, md_num_steps = 10):
        """ Perform RE for each replica given the number of swap phases """
        #[add some more dcomunetatio about how this ufnction works and what it is doing]
        if self.rank == 0:
            print(f"Starting a replica exchange with {self.num_replicas} replicas.")
        
        # [Maybe add this in a function later]
        # Create lists with indices of replica swaps to tell swap_attempt function 
        # which replicas to attempt to swap
        even_swap_idx = [[2*i,2*i+1] for i in range(self.num_replicas//2)]
        odd_swap_idx = [[2*i+1,2*(i+1)] for i in range((self.num_replicas-1)//2)]
        
        for swap_phase in range(number_of_swap_phases):
            
            #Even swap phase
            if swap_phase % 2 == 0: 
                idx_to_use = even_swap_idx
            # Odd swap phase
            else: 
                idx_to_use = odd_swap_idx

            if self.rank == 0:
                print(f"Start of MD stage {swap_phase}.\n")
            
            print(f"replica {self.rank} : Running MD\n ")
            self.replica.run_MD(md_num_steps)
            self.replica._reset_i()
            print(f"replica {self.rank} : MD done\n ")

            self.comm.Barrier() # synchronize procs

            if self.rank == 0:
                print(f"End of MD stage {swap_phase}.\n")
                print(f"\nStart of swap phase {swap_phase}.\n")
            
            flat_idx_to_use = [item for sublist in idx_to_use for item in sublist]
            if self.rank == 0:
                print(flat_idx_to_use)

            if self.rank in [self.temperature_idx[i] for i in flat_idx_to_use]:     # now we know which replicas should attempt a swap, but NOT their swapping partners
                # how do we get their swapping partners?
                if swap_phase %2 == 0:
                    swap_partner_idx = idx_to_use[(self.temperature_idx).index(self.replica.rank)//2][((self.temperature_idx).index(self.replica.rank)+1)%2]
                else:
                    swap_partner_idx = idx_to_use[((self.temperature_idx).index(self.replica.rank)-1)//2][((self.temperature_idx).index(self.replica.rank))%2]
                self.attempt_swap(swap_partner_idx)

            # I think its the proc with the highest rank. Idx being one too large. This is true, dont have a fix idea yet
            # I'm printing the ranks, 2 and 1
            # The list is [[1,2]], so length is 1, which is the last print. Rank 2 // 2 = 1, so idx_to_use[1] is out of bounds. It is only a problem for the largest rank
            # update temperature_idx list
            #self.temperature_idx = [0 for _ in range(self.num_replicas)]

            self.idx_to_use_unsorted = self.comm.gather(self.replica.own_temperature_idx, root = 0)
            self.ranks = self.comm.gather(self.replica.rank, root = 0)
            
            if self.rank == 0:
                print("rank 0: gathered idx_to_use_unsorted", self.idx_to_use_unsorted)
                print("rank 0: gathered ranks", self.ranks)
            
            # Order it - we assume that its ordered by rank before
            # Sort by population
            # cities = sorted(cities, key=lambda city: city['population'])
            if self.replica.rank == 0:
                list_to_sort = [self.idx_to_use_unsorted,[i for i in range(self.num_replicas)]]
                self.temperature_idx = [x for _, x in sorted(zip(list_to_sort[0],list_to_sort[1]), key = lambda x:x[0])]
                print("rank 0: new sorted temperature_idx", self.temperature_idx)

            self.temperature_idx = self.comm.bcast(self.temperature_idx, root = 0)

        self.replica._produce_csv()

# =============================================================================
# Main
# =============================================================================

# def main(num_replicas, temperature_start, temperature_end, config, num_swap_phases, num_md_steps): #TODO inputs of main for replica exchange
    
#     RE = ReplicaExchange(num_replicas,temperature_start,temperature_end,config)

#     RE.run_parallel_RE(number_of_swap_phases=num_swap_phases,md_num_steps=num_md_steps)   
    
# config1 = nanoalloy.build_coreshell('Al', 'Pt', 0.6, 4, 4.09, n=1)
# config1.calc = EMT()
# main(num_replicas=8,temperature_start=500,temperature_end=1200,config=config1,num_swap_phases=5,num_md_steps=100)

#%% 
# =============================================================================
# Testing
# =============================================================================

# # TEST CONFIGURATIONS
# surfaces = [(1, 0, 0), (1, 1, 0), (1, 1, 1)]
# layers = [1, 2, 1]
# lc = 3.61000
# config1 = FaceCenteredCubic('Cu', surfaces, layers, latticeconstant=lc); config1.calc=EMT()
# config2 = FaceCenteredCubic('Ni', surfaces, layers, latticeconstant=lc); config2.calc=EMT()
# config3 = FaceCenteredCubic('Al', surfaces, layers, latticeconstant=lc); config3.calc=EMT()
# config4 = FaceCenteredCubic('Ag', surfaces, layers, latticeconstant=lc); config4.calc=EMT()

# # group configs to use as input in RE
# initial_configs = [config1,config2,config3,config4]
# num_replicas = 4

# a = ReplicaExchange(num_replicas, 500,800, initial_configs) # RE instance


# #%% 
# #a.run_serial_RE(2,10)
# a.replicas[0].run_MD(1000)

#print(a.generate_replica_temperatures())

#%% 
# =============================================================================
# TODO
# ============================================================================= 
# implment parallelization (later on)
# when to implement the swapping/checking for swapping? probably fixed time intervals
# which ones to swap? probably determinstic odd-even
# add log of what replica at what T, time, etc.

# put in replica exchnage class: 
# initialize X many cores
# each core produces its own log file locally with Temperature, time stamp and potential enery nd replica number, target temperature. (think about what happens when temperature is flucatuating)
# at the end root file gathers everything and retur one file with all the logs of each replica (next to each other)
# choose spacing between temperatures. Besides linear, it could be exponential or inverse exponential.
# we could call the number of replicas K to conform with wham. Also so we don't have num_replicas = num_replicas in init.
# MD re-init every stage vs. MD.set_temperature()
# MaxwellBoltzmannDist. temperature or not when init. MD
# what type of printing do we prefer?
# we need to achieve less oscillation, Palagin said +-150 K is fine
#
