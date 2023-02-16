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
import pandas as pd                                 # for output csv files
import time

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
    
    def __init__(self, index, target_T, atom_config):
        ''' Initialize instance of the Replica Class '''
        
        ############# BASIC ATTRIBUTES
        self.idx = index                    # replica index
        self.configuration = atom_config    # replica atom configuration
        self.target_T = target_T            # target temperature in [K]
        self.timestep = 3                   # harcoded to three [fs]
        
        ############# LANGEVIN
        MaxwellBoltzmannDistribution(self.configuration,temperature_K=self.target_T) #trying to set initial temperature
        self.MD = Langevin( atoms           = self.configuration,                  
                            timestep        = self.timestep*units.fs, 
                            temperature_K   = target_T, 
                            friction        = 1/3)
        
        ############# TRAJECTORY
        traj = Trajectory("replica_"+str(self.idx)+".traj", 'a', self.configuration)
        self.MD.attach(traj.write, interval=2) # append to traj file at interval 2
        
        ############# LOGGER
        self._i = 1
        self.frame = pd.DataFrame(columns=["time[ps]", "epot/N[eV]", "beta[1/eV]", "target_beta[1/eV]"])
        
        self._times = [self.idx]            # list of times for csv files
        self._epot = [self.idx]             # list of energies for csv files
        self._betas = [self.idx]            # list of betas for csv files
        self._target_betas = [self.idx]     # list of target betas for csv files

        # logger print function
        def _save_log(a=self):
            if self._i >= 100:
                self._times.append(a.MD.get_time() / (1000 * units.fs))
                self._epot.append(a.configuration.get_potential_energy() / len(a.configuration))
                self._betas.append(1/(a.configuration.get_temperature()*units.kB))                     
                self._target_betas.append(1/(a.target_T*units.kB))

        def _increase_i():
            self._i += 1
        
        self.MD.attach(_save_log, interval=1)
        self.MD.attach(_increase_i, interval=1)
    
        ######### INITIAL TEMPS SET BY MAXWELL DISTB.
        print(f"temp of replica {self.idx} = {self.configuration.get_temperature()}")   

    @property
    def Beta(self):
        ''' Compute the inverse temperature, Beta = 1/(T*k_B) [1/eV] '''
        return 1/(self.target_T * units.kB)
    
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

        output_filename = str('replica_'+str(self.idx)+'.csv')
        self.frame.to_csv(output_filename, header=False, index=False)

    def run_MD(self, md_num_steps): # [atm keeping print statements for tests]
        ''' Perform MD simulation and log the new configuration data
            
            INPUTS:
            md_num_steps :    number of MD steps to run 
        '''
        
        self.MD.run(md_num_steps)   # the MD simulation
        
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
                         initial_configs, type_of_swapping='one_at_a_time', 
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
        if type(initial_configs) != list:
            initial_configs = [initial_configs for _ in range(num_replicas)]
        if len(initial_configs) != num_replicas:
            self.num_replicas = len(initial_configs)
            print("Number of initial configurations and number of replicas are not the equal.\
                   Using the number of initial configurations as number of replicas.\n")            
        else:
            self.num_replicas = num_replicas    
        
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
            
        self.replicas = [Replica(index=i,
                                 target_T=self.temperatures[i],
                                 atom_config=initial_configs[i]) 
                                 for i in range(self.num_replicas)]

        # Initialize accepted/rejected statistics for computing RE efficiency 
        # swap_log tracks which pairs were attempted to swap. It is a 2D list, 
        # where each inner entry is a swap attempt. The inner swap attempt 
        # structure is as follows: [temperature_of_replica_before_swap_1, 
        # temperature_of_replica_before_swap_2, replica_1, replica_2, bool 0 
        # if rejected and 1 if accepeted]
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
    
    def attempt_swap(self, idx_temperature_A, idx_temperature_B):
        """ Attempt to swap replica A and B by choosing the temperatures indices 
        of temperature_idx that will be swapped. So if the inputs are 0 and 1, 
        we swap the replica that currenlty has tempeareture 0 with the one that 
        currenlty has temperature 1"""
        
        # Check if not swapping with its own
        if idx_temperature_A == idx_temperature_B:
            raise Exception("Idx of swap attempt cannot be equal. Trying to swap with itself!")
            
        # Get replica idx to know which replicas are at the temperatures that will be swapped
        A_idx = self.temperature_idx[idx_temperature_A] 
        B_idx = self.temperature_idx[idx_temperature_B]
        A = self.replicas[A_idx]
        B = self.replicas[B_idx]
        
        #Log the the swap attempt
        local_swap_log = []
        local_swap_log += [min(idx_temperature_A, idx_temperature_B),
                       max(idx_temperature_A, idx_temperature_B)]   # temperatures
        local_swap_log += [min(A_idx, B_idx),max(A_idx, B_idx)]         # replicas
        
        # Accept or reject swap based on computed probability
        accept_swap, p = self.calc_swap_prob(A.Beta, B.Beta, A.U, B.U)
        
        #print(f"Attempting to swap replica {A.idx} and replica {B.idx} with T = {A.target_T} K and T = {B.target_T} K, respectively.")
        if accept_swap:
            #print("Swap accepted, initializing swap ...")
            # printing swap visually
            print(f"Replica {A.idx}: {A.target_T:>6.1f}  –—    ––>  {B.target_T:5.1f}")
            print(f"                      \/ ")
            print(f"                      /\ ")
            print(f"Replica {B.idx}: {B.target_T:>6.1f}  –—    ––>  {A.target_T:5.1f}")
            
            # Scale velocities
            A.configuration.set_velocities(A.configuration.get_velocities()*np.sqrt(A.Beta/B.Beta))
            B.configuration.set_velocities(B.configuration.get_velocities()*np.sqrt(B.Beta/A.Beta))
            
            # set new target temperatures
            A.target_T = self.temperatures[idx_temperature_B] # swap/set replica temperature
            B.target_T = self.temperatures[idx_temperature_A] # swap/set replica temperature

            # set temperatures for MD
            A.MD.set_temperature(A.target_T*units.kB)         # set the temperature of the replica MD
            B.MD.set_temperature(B.target_T*units.kB)         # set the temperature of the replica MD
                        
            self.temperature_idx = swap_positions(self.temperature_idx, idx_temperature_A, idx_temperature_B)
            #print("   ... temperatures swapped and new MD tempreatures set. Swap completed.")
            # Add one to 'accepted' list

            self.accepted += 1
            
            # Log the swap attempt acceptance criterion as 1 (accepted)
            local_swap_log += [1]     

        else:
            #print("Swap rejected.")
            # printing swap visually
            print(f"Replica {A.idx}: {A.target_T:>6.1f}  ––––––––>  {A.target_T:5.1f}")
            print(" ")
            print(" ")
            print(f"Replica {B.idx}: {B.target_T:>6.1f}  ––––––––>  {B.target_T:5.1f}")
            # [Add one to 'rejected' list ]
            # [Possibl race conditions? ]
            # [Later on add which replicas were rejected and when? ]
            self.rejected += 1
            
            # Log the swap attempt acceptance criterion as 0 (rejected)
            local_swap_log += [0]

        # Log the local_swap_log to the main_swap_log
        self.swap_log.append(local_swap_log)
        
        #return print("End of swap attempt. \n")
        
    def run_serial_RE(self, number_of_swap_phases, md_num_steps = 10):
        """ Perform RE for each replica given the number of swap phases """
        #[add some more dcomunetatio about how this ufnction works and what it is doing]
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

            print(f"Start of MD stage {swap_phase}.\n")
            for replica_id, replica in enumerate(self.replicas): #[possibly add time]
                print(f"Running MD for replica {replica_id} ...\n ")
                replica.run_MD(md_num_steps)
                print("... MD run done.\n ")
                replica._reset_i()
            
            print(f"End of MD stage {swap_phase}.\n")
            
            print(f"Start of swap phase {swap_phase}.\n")
            for swap_num, swap_temperature_idx in enumerate(idx_to_use):
                print(f"-- Swap attempt: {swap_num}")
                self.attempt_swap(swap_temperature_idx[0], swap_temperature_idx[1])
                print(f"-- End of swap attempt {swap_num}.\n")

            print(f"End of swap phase {swap_phase}. New temperatures:")
            print(f"{self.generate_replica_temperatures()}\n")

            # 
            print(f"reset 'i' check: replica 0: {self.replicas[0]._i}, replica -1: {self.replicas[-1]._i}\n")
        
        for replica in self.replicas:
            replica._produce_csv()

