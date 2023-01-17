#!/usr/bin/env python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ase.units import kB
import py_bigfloat as bf
import bigfloat
from mpi4py import MPI

class WHAM:
	"""
	The Weighted Histogram Analysis Method (WHAM) on data provided by a Replica Exchange (RE).

	INPUT:
	It needs one CSV input file containing all replica instantaneous potential energies [eV] and instantaneous 
	inverse temperatures - betas [1 / (k_B * K); k_B = [eV / K]] and target betas and time of measurement [-] for each replica. 
	Empty lines and lines starting with a hashmark ('#') are ignored.
	All replicas must start at unique temperature!

	INPUT FORMAT:

	# rep_id,rep_id,      rep_id, rep_id,      rep_id,  rep_id,      rep_id, rep_id,      rep_id,  rep_id,      rep_id, rep_id
	1,       1,           1,      1,           2,       2,           2,      2,           3,       3,           3,      3       
	# time,  energy,      beta,   target_beta, time,    energy,      beta,   target_beta, time,    energy,      beta,   target_beta
	100.000, -0.68626290, 0.5500, 0.5501,      100.000, -0.78626290, 0.7700, 0.6601,      100.000, -0.78626290, 0.7700, 0.7701
	200.000, -0.66438440, 0.5500, 0.5501,      200.000, -0.76438440, 0.7700, 0.7701,      200.000, -0.96438440, 0.7700, 0.6601
	300.000, -0.71729520, 0.6600, 0.6601,      300.000, -0.81729520, 0.7700, 0.7701,      300.000, -0.76438440, 0.7700, 0.5501

	OUTPUT:
	U = [eV]
	cV = [eV / K]

	OUTPUT FORMAT:
	TODO:

	"""

	def __initialize(self, filename):
	
		self.file = pd.read_csv(filename, delimiter=",", skip_blank_lines=True, comment="#", header=None)
		print(self.file)

		self.A_kn = self.file.iloc[1:, 1::4].transpose()
		self.A_kn_min = self.A_kn.min().min()
		self.A_kn_max = self.A_kn.max().max()

		T_min = 100
		factor = 10
		self.M = int((self.A_kn_max - self.A_kn_min) * factor / (kB * T_min))
		self.bin_width = (self.A_kn_max - self.A_kn_min) / self.M
		print("bin_width", self.bin_width)


		self.K = int(self.file.iloc[0].max())
		self.L = self.K  #TODO: one nice day maybe this will be what it should be, independent of K
		self.N_k = len(self.file.index) - 1
		self.G_mk = np.ones((self.M, self.K))  #TODO: one nice day maybe this will be what it should be
		# self.G_mk = np.random.normal(1, 0.1, (self.M, self.K))
		# self.G_mk[2, 2] = 1.2
		# TODO: try out different g later != 1

		self.Psi_m_kn = ((self.A_kn - self.A_kn_min) / self.bin_width).astype(int)  #indicates which bin the energy falls into
		self.Psi_m_kn[self.Psi_m_kn == self.M] = self.M - 1

		self.H_km = np.zeros((self.K, self.M))
		for k in range(self.K):
			for n in range(self.N_k):
				self.H_km[k, self.Psi_m_kn.iloc[k, n]] += 1

		self.H_m = np.sum(self.H_km, axis=0)
		self.H_m_eff = np.sum(self.H_km.transpose() / self.G_mk, axis=1)

		self.N_kl = pd.DataFrame(data=np.zeros((self.K, self.L)), columns=self.file.iloc[1, 3::4].astype(str))

		self.beta = self.file.iloc[1:, 3::4]
		for n in range(self.N_k):
			for k in range(self.K):
				self.N_kl.loc[k, str(self.beta.iloc[n, k])] += 1

		self.N_ml_eff = np.empty((self.M, self.L))
		for m in range(self.M):
			for l in range(self.L):
				self.N_ml_eff[m, l] = np.sum(self.N_kl.iloc[:, l] / self.G_mk[m, :])
		
		# FROM HERE ON I TRY TO IMPLEMENT BIGFLOAT WRAPPER TO SEE IF PRECISION IS THE PROBLEM
		self.f_l = bf.bf_array(np.ones(self.L))
		self.U_m = bf.bf_array(np.linspace(self.A_kn_min + self.bin_width/2, self.A_kn_max - self.bin_width/2, self.M))
		self.density_of_states_estimate_m = bf.bf_array(np.zeros(self.M))

	def __get_W_kn(self, beta):
		"""
		INPUTS:
		beta = [1/(k_B * K)]

		INTERMEDIATES:


		OUTPUTS:
		numpy matrix with entries W_kn[k, n] = w_kn(β) 
		"""

		W_kn = bf.bf_array(np.empty((self.K, self.N_k)))
		for k in range(self.K):
			for n in range(self.N_k):
				m = self.Psi_m_kn.iloc[k, n]
				temp1 = bf.bf_array(self.density_of_states_estimate_m.array[m])
				temp2 = bf.bf_array(self.U_m.array[m])
				temp3 = bf.bf_array(bigfloat.BigFloat(-beta, context=W_kn.context))
				temp4 = bf.bf_array(bigfloat.BigFloat(self.H_m[m], context=W_kn.context))
				W_kn.array[k, n] = (temp1 * bf.exp(temp2 * temp3) / temp4).array[0]

		return W_kn # bf_array


	def __A_estimate(self, A_kn, beta):
		"""
		INPUTS: 
		beta = [1/(k_B * K)]

		INTERMEDIATES:
		W_kn = numpy matrix with entries W_kn[k, n] = w_kn(β)
		A_kn = numpy matrix with entries A_kn[k, n] = <observable>_kn
		numerator = numerator of the following formula
		denominator = denominator of the following formula

		OUTPUTS:
		estimate of <observable> at inverse temperature β

						Σ_{k=1}^{K} Σ_{n=1}^{N_k} w_kn(β)*A_kn
		A_estimate(β) = --------------------------------------
						   Σ_{k=1}^{K} Σ_{n=1}^{N_k} w_kn(β)
		"""

		W_kn = self.__get_W_kn(beta)  # bf_array

		numerator = bf.sum(W_kn * bf.bf_array(A_kn))
		denominator = bf.sum(W_kn)

		return numerator / denominator # bf_array

	def __iterate(self, epsilon, max_iter):

		self.__update_density_of_states_estimate_m()
		old_f_l = self.f_l # bf_array
		self.__update_free_energy_l()
		iter = 0
		
		while (bigfloat.greater(bf.norm(self.f_l - old_f_l).array[0], epsilon) and (iter < max_iter)):
			print(iter)
			print("f_l")
			# self.f_l.print()
			print("sigma")
			# self.density_of_states_estimate_m.print()

			self.__update_density_of_states_estimate_m()
			old_f_l = self.f_l
			self.__update_free_energy_l()
			iter += 1

		print("escaped while loop at iteration =", iter, "out of max iterations =", max_iter)
		
	def __update_density_of_states_estimate_m(self):

		denominator = bf.bf_array(np.zeros(self.M))
		for l in range(self.L):
			denominator = denominator + bf.bf_array(self.N_ml_eff[:, l] * self.bin_width) * bf.exp(self.U_m*(-1)*self.beta.iloc[0, l] + bf.bf_array(self.f_l.array[l]))

		self.density_of_states_estimate_m = bf.bf_array(self.H_m_eff) / denominator

	def __update_free_energy_l(self):
		dense = self.density_of_states_estimate_m # bf_array
		sum = bf.bf_array(np.zeros(self.L))
		for m in range(self.M):
			temp1 = bf.bf_array(dense.array[m]) * self.bin_width
			temp2 = bf.bf_array(-self.beta.iloc[0, :].to_numpy())
			temp3 = bf.bf_array(self.U_m.array[m])
			sum = sum + bf.exp(temp2*temp3) * temp1
			# print("sum")
			# sum.print()
		self.f_l = bf.log(sum) * (-1)

	def save_data(self, input_file_name, output_filename, temp_start, temp_end, num_data_points=100, convergence_epsilon=1e-3, convergence_max_iter=100):

		#only load the file and initialize necessary variables for calculations
		self.__initialize(input_file_name)

		#now do the iteration until convergence
		print("now going to iterate until forever")
		self.__iterate(epsilon=convergence_epsilon, max_iter=convergence_max_iter)
		
		k_B = kB  # [eV / K]
		U = self.A_kn.to_numpy()
		U_2 = U*U
		T_range = np.linspace(temp_start, temp_end, num_data_points)
		beta_range = 1 / (k_B * T_range)

		N = 1 #TODO: need to convert between specific heat capacity and heat capacity
		U_2_estimate = np.zeros(num_data_points, dtype=float)
		U_estimate = np.zeros(num_data_points, dtype=float)
		res_U_2_estimate = np.zeros(num_data_points, dtype=float)
		res_U_estimate = np.zeros(num_data_points, dtype=float)


		comm = MPI.COMM_WORLD
		my_rank = comm.Get_rank()
		size = comm.Get_size()

		print("rank, size =", my_rank, size)

		leftover = int(len(beta_range) % size)
		# print("lefover", leftover)
		range_size = int((len(beta_range) - leftover) / size)
		# print("range_size", range_size)

		# for i, log_beta in enumerate(beta_range):


		for i, log_beta in enumerate(beta_range[range_size * my_rank: range_size * (my_rank + 1)]):
			print("for loop index", i, my_rank)
			index = range_size * my_rank + i
			print("index", index)
			U_estimate[index] = float(self.__A_estimate(U, log_beta).array[0])
			U_2_estimate[index] = float(self.__A_estimate(U_2, log_beta).array[0])

		print(U_estimate)
		
		if my_rank == 0 and leftover != 0:
			for i, log_beta in enumerate(beta_range[-leftover:]):
				print("for loop index", i, my_rank)
				index = len(beta_range) - leftover + i
				print("index leftover", index)
				U_estimate[index] = float(self.__A_estimate(U, log_beta).array[0])
				U_2_estimate[index] = float(self.__A_estimate(U_2, log_beta).array[0])

			print(U_estimate)

		comm.Reduce([U_estimate, MPI.DOUBLE], [res_U_estimate, MPI.DOUBLE], op=MPI.SUM, root=0) 
		comm.Reduce([U_2_estimate, MPI.DOUBLE], [res_U_2_estimate, MPI.DOUBLE], op=MPI.SUM, root=0) 

		if my_rank == 0:
			print("res", res_U_estimate)
			c_v = (res_U_2_estimate - res_U_estimate**2) / (N * k_B * T_range**2)  # now just in [eV / K]
			c_v = (c_v / k_B)  # nondimensionalized c_v

			frame = pd.DataFrame(columns=["temp", "c_v", "U_estimate", "U^2_estimate"])
			frame["temp"] = T_range
			frame["c_v"] = c_v
			frame["U_estimate"] = res_U_estimate
			frame["U^2_estimate"] = res_U_2_estimate

			print("filename =", output_filename)
			frame.to_csv(output_filename, header=True, index=False)

	def show_plot(self, input_file_name):

		comm = MPI.COMM_WORLD
		my_rank = comm.Get_rank()

		if my_rank == 0:
			print("showing plot only from rank 0")
			data = pd.read_csv(input_file_name, header=0) # expects header
			T = data["temp"].to_numpy()
			c_v = data["c_v"].to_numpy()
			plt.plot(T, c_v)
			plt.show()

filename = "wham_data.csv"

input_file = pd.read_csv("inputs.txt", delimiter=":", skiprows = [0,1,2,3,5,6,12,13,16,17,22,23,29,30], header=None, skipinitialspace=True)

wham = WHAM()
wham.save_data(input_file_name="collapse.csv",
			   output_filename=filename,
			   convergence_epsilon=float(input_file.iloc[12, 1]),
			   convergence_max_iter=int(input_file.iloc[13, 1]),
			   temp_start=float(input_file.iloc[14, 1]), 
			   temp_end=float(input_file.iloc[15, 1]),
			   num_data_points=int(input_file.iloc[16, 1]))

wham.show_plot(filename)

