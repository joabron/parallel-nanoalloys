#!/usr/bin/env python
"""
This program will calculate energy average, fluctuation and specific heat
curves for a range of temperatures using the Weighted Histogram Analysis Method
(WHAM).

It is normally used to process output from Replica Exchange Molecular Dynamics
(REMD) runs.

It needs a set of input files containing replica energies vs time for each
replica, as well as the instantaneous temperatures of each replica in each time
step.

An <energies*> file should be a text file with 6 fields in each row, separated
by whitespace. The fields should be organized as follows:

<time> <aveTemperature> <avePressure> <avePotential> <instPotential>
<instKineticE>

Only the <time> and <instPotential> are used, which denote the time and
instantaneous potential energy of the frame.  Energy should be in kcal/mol
units, otherwise the value of the Boltzmann-constant (K_B) should be modified
in the source code.  Empty lines and lines starting with a hashmark ('#') are
ignored.  Files having the ".bz2" extension will be treated as bzip2 compressed
files.

For example:
#ave = average; inst = instant
#Time         aveTemperature          avePressure       avePotential   instPotential    instKineticE
     100.000         0.83037         0.0000012982         -411.09977      -686.26290      2307.91329
     200.000         0.55341        -0.0000001098         -686.24821      -664.38440      2251.34883
     300.000         0.55628         0.0000003669         -702.13495      -717.29520      2273.45426
     400.000         0.56393         0.0000002264         -713.19302      -716.46560      2314.19416
...

The <replica_temps> file should be a text file which contains the instantaneous
temperatures (in 1/K) for each replica at each time step.  Each row should contain N+1 
fields if N replicas were used, organized as follows:

<time> <temp1> <temp2> ... <tempN>

Empty lines and lines starting with a hashmark ('#') are ignored.

For example:
#time T(P0) T(P1) T(P2) .....
      0.0000     0.5500     0.5588     0.5678     0.5768     0.5861     0.5955     0.6050     0.6147     0.6245     0.6345     0.6447     0.6550     0.6655     0.6762     0.6870     0.6980     0.7092     0.7205     0.7321     0.7438     0.7557     0.7678     0.7801
     0.7926     0.8053     0.8182     0.8313     0.8446     0.8581     0.8719     0.8858     0.9000
    100.0000     0.5500     0.5588     0.5678     0.5768     0.5861     0.6050     0.5955     0.6245     0.6147     0.6447     0.6345     0.6655     0.6550     0.6870     0.6762     0.7092     0.6980     0.7321     0.7205     0.7438     0.7557     0.7678     0.7801
     0.8053     0.7926     0.8182     0.8313     0.8581     0.8446     0.8858     0.8719     0.9000
    200.0000     0.5588     0.5500     0.5678     0.5768     0.5955     0.6147     0.5861     0.6345     0.6050     0.6550     0.6245     0.6762     0.6447     0.6870     0.6655     0.7205     0.6980     0.7321     0.7092     0.7438     0.7678     0.7557     0.7801
     0.8182     0.7926     0.8053     0.8446     0.8719     0.8313     0.9000     0.8581     0.8858
...

The script is built according to the guidelines described in [Chodera2007]:

John D. Chodera, William C. Swope, Jed W. Pitera, Chaok Seok, and Ken A. Dill:
Use of the Weighted Histogram Analysis Method for the Analysis of Simulated and
Parallel Tempering Simulations
J. Chem. Theory Comput., 2007, 3 (1), pp 26-41.
DOI: 10.1021/ct0502864

Several output files are generated.

* histogram.dat:
Contains a histogram of energy values using the specified binning.

<energy> <count_total> <count1> <count2> ... <countN>

<energy>: centre of the energy histogram bin
<count_total>: total histogram counts for all replicas.
<countn>: histogram counts for replica n.

* psi_avg.dat:
Average histogram bin occupancy.

<occupancy_1_1> <occupancy_1_2> ... <occupancy_1_K>
:
<occupancy_N_1> <occupancy_N_2> ... <occupancy_N_K>

<occupancy_n_k>: average occupancy of bin n in trajectory k.

* g.dat:
Statistical inefficiency values (as defined in [Chodera2007]).

<g_1_1> <g_1_2> ... <g_1_K>
:
<g_N_1> <g_N_2> ... <g_N_K>

<g_n_k>: statistical inefficiency value for bin n in trajectory k.

* wham_out.dat:
WHAM calculation output, average energy, average squared energy and specific
heat calculated for various temperatures.

<T> <E_avg> <E2_avg> <Cv> <sqrt_d2Cv>

<T>: temperature in Kelvins.
<E_avg>: average energy at temperature T, in kcal/mol.
<E2_avg>: average squared energy at temperature T, in (kcal/mol)^2.
<Cv>: molar heat capacity at constant volume at temperature T, in kcal/mol/K.
<sqrt_d2Cv>: square root of the variance of Cv, calculated according to [Chodera2007].


"""

import glob
import numpy
from numpy import *

import sys
import bz2

# usage: ./pywham.py <energies1> <energies2> <energies3> ... <replica_temps>

if (len(sys.argv) < 2):
	print("usage: %s <energies1> [<energies2> ...] <replica_temps>" % sys.argv[0])
	sys.exit(-1)

files = sys.argv[1:-1]
replica_temps_fn = sys.argv[-1]

# time of the first time frame
FIRST_T = 1000
# number of histogram bins
N_BINS = 200
# conversion from internal temperature units to Kelvins
# T[K] = T_TO_K * T[internal]
T_TO_K = 300.0/0.6
# Boltzmann constant in kcal/mol/K
K_B = 0.001987
#K_B = 8.617343e-5

# convergence threshold for free energy iterations
EPSILON = 1e-4
# maximum number of free energy iterations
MAX_ITER = 1000000

# min and max temperatures in Kelvins for WHAM calculation
MIN_T = 300.0
MAX_T = 1400.0
# number of points for WHAM calculation
NUM_T = 100


# here it is assumed that replicas swap temperatures
energies_d = []
for fn in files:
	# it's crazy, some time steps are missing!
	# need to be very careful while matching replica energies and temperatures.
	if (fn.endswith(".bz2")): f = bz2.BZ2File(fn)
	else: f = open(fn)
	ener = dict()
	for s in f:
		if (s.startswith("#")): continue
		try:
			l = s.split()
			t = int(round(float(l[0])))
			e = float(l[4])
			ener[t] = e
		except (ValueError, IndexError):
			pass
	energies_d.append(ener)
	f.close()

f = open(replica_temps_fn)
times = []
temps_d = dict()
for s in f:
	# there can be restarts and overlaps in time!
	if (s.startswith("#")): continue
	try:
		l = s.split()
		t = float(l[0])
		if (t not in temps_d): times.append(t)
		temps_d[t] = [float(x) for x in l[1:]]
	except (ValueError, IndexError):
		pass
f.close()

energies = []
temps = []
for t in times:
	if (t < FIRST_T): continue
	ok = True
	e = []
	for (i, d) in enumerate(energies_d):
		if (t not in d):
			ok = False
			break
		e.append(d[t])
	if (ok):
		energies.append(e)
		temps.append(temps_d[t])

energies = array(energies)

N = energies.shape[0]
print("N =", N)

temps = array(temps)
print("temps.shape =", temps.shape)
print("energies.shape =", energies.shape)

temperatures = list(temps[0, :])
temperatures.sort()

all_min = amin(energies)
all_max = amax(energies)

print("all_min =", all_min)
print("all_max =", all_max)
dU = (all_max - all_min) / N_BINS
print("dU = ", dU)

# number of replicas
K = temps.shape[1]
bins = zeros((K, N), dtype=int)
H = zeros((N_BINS, K+1), dtype=int)

E = zeros((K, N))
for k in xrange(K):
	h = [0] * N_BINS
	for (i, e) in enumerate(energies[:, k]):
		#ii = int(N_BINS * (e - all_min) / (all_max - all_min))
		ii = int(((e - all_min) / dU) + 0.5)
		if (ii >= N_BINS): ii = N_BINS-1
		bins[k, i] = ii
		H[ii, k] += 1
		E[k, i] = e

print("sum(H, axis=0) =", sum(H, axis=0))
print("sum(H, axis=1) =", sum(H, axis=1))

# output energy histogram
f = open("histogram.dat", "w")
for m in xrange(N_BINS):
	e = (m+0.5) * dU + all_min
	l = ["%.6f" % e]
	n = 0
	for k in xrange(K):
		l.append("%d" % H[m, k])
		n += H[m, k]
	l.insert(1, "%d" % n)
	H[m, K] = n
#	print " ".join(l)
	f.write(" ".join(l) + "\n")
f.close()

# beta count

# N_beta[k, l] = number of times replica k was in temperature (index) l
N_beta = zeros((K, K), dtype=int)
tt = dict()
for k in xrange(K):
	tt[temperatures[k]] = k
for k in xrange(K):
	for i in xrange(N):
		N_beta[k, tt[temps[i, k]]] += 1
print(N_beta)

betas = 1/(K_B*temps*T_TO_K)

# WHAM

# correlations

t = 1
psi_avg = 1.0 * copy(H[:, :-1]) / N

# output average bin occupancy
f = open("psi_avg.dat", "w")
for m in xrange(N_BINS):
	l = []
	for k in xrange(K):
		l.append("%lg" % psi_avg[m, k])
	f.write(" ".join(l))
	f.write("\n")
f.close()

g = zeros((N_BINS, K))
crossed_zero = zeros((N_BINS, K), dtype=bool)
while (t < N):
	C = zeros((N_BINS, K))
	idx = (bins[:, :-t] == bins[:, t:])
	for k in xrange(K):
		for (i, n) in enumerate(bincount(bins[k, idx[k, :]])):
			C[i, k] += n
	idx = psi_avg > 0
	C[idx] = (C[idx] - (psi_avg[idx]**2)*(N-t)) / (psi_avg[idx] - psi_avg[idx]**2)
	# truncate the sum for the autocorrelation functions at the first term that crosses zero
	# this is the method described in [Chodera2007]
	crossed_zero[C < 0] = True
	print("t = %6d, #idx = %d, #crossed_zero = %d" % (t, sum(idx), sum(crossed_zero)))
	# exit if all correlation functions have crossed zero
	if (sum(idx) == sum(crossed_zero)): break
	# add all terms that have not crossed zero yet
	g[-crossed_zero] += 2 * C[-crossed_zero] / N
	t += 1
g += 1.0

print(g)

# output sampling efficiencies
f = open("g.dat", "w")
for m in xrange(N_BINS):
	l = []
	for k in xrange(K):
		l.append("%lg" % g[m, k])
	f.write(" ".join(l))
	f.write("\n")
f.close()

# f = unitless free energy
f = zeros(K, dtype=float128)

beta = zeros(K)
for (k, T) in enumerate(temperatures): beta[k] = 1/(K_B*T*T_TO_K)
U = zeros(N_BINS)
for m in xrange(N_BINS): U[m] = (m+0.5) * dU
print(U)


print("f =", f)
print("beta =", beta)

# g[m, k]
# H[m, k]
# N_beta[k, l]

# g^-1
g_rec = power(array(g, dtype=float128), -1)

print("g_rec =", g_rec.shape)
print("H =", H.shape)
# effective histogram counts
# H_eff[m] array
H_eff = sum(g_rec * H[:, :-1], axis=1)
print("H_eff =", H_eff.shape)
# N_beta_eff[m, l] matrix
print("N_beta =", N_beta.shape)
# effective number of samples for a temperature
N_beta_eff = matrix(g_rec) * matrix(N_beta)
print("N_beta_eff =", N_beta_eff.shape)

for ii in xrange(MAX_ITER):
	print("ii = %6d, f =" % (ii, ), f)

	new_ef = zeros(K, dtype=float128)

	# exponent[l, m] matrix
	exponent = matrix(f, dtype=float128).T - matrix(beta, dtype=float128).T*matrix(U)

	for l in xrange(K):
		# D[m] array
		D = sum(N_beta_eff.A * exp(exponent + beta[l] * matrix(U)).T.A, axis=1) # eq. 66
		new_ef[l] = sum(H_eff / D)
	idx = (new_ef <= 0)
	new_f = -log(new_ef)
	new_f[idx] = 0.0

	# normalize to avoid numeric drift
	new_f -= min(new_f)
	print("f =", f)
	print("new_f =", new_f)
	diff = max(abs(new_f - f))
	f = new_f
	if (diff < EPSILON):
		print("stopping, difference = %lg < %lg" % (diff, EPSILON))
		break
	else:
		print("difference = %lg" % diff)

# save state

#savez("state.npz", g=g, f=f)
#data = load("state.npz")
#f = data["f"]
#g = data["g"]

print("final f =", f)


# STATISTICAL INEFFICIENCY

# calculates cross correlation for x and y datasets
def cross_corr(x, y):
	x_avg = mean(x, axis=1)
	y_avg = mean(y, axis=1)
	(K, N) = x.shape

	sigma2_xy = sum(x * y) / (N-1) - x_avg*y_avg
	
	t = 1
	g = zeros(K, dtype=float128)
	crossed_zero = zeros(K, dtype=bool)
	while (t < N):
		if ((t % 1000) == 1): print("cross_corr: t =", t)
		C = sum((x[:, :-t] * y[:, t:] + x[:, t:] * y[:, :-t]), axis=1) / (2 * (N - t))
		C -= x_avg*y_avg
		C /= sigma2_xy

		idx = (C < 0)
		crossed_zero[idx] = True
		if (crossed_zero.all()): break
		g[-crossed_zero] += 2 * C[-crossed_zero] * (1 - (float(t)/N))
		#print "t = %6d, #crossed_zero = %d, gg = %s" % (t, sum(crossed_zero), str(gg))
		t += 1
	g += 1.0
	return (x_avg, y_avg, sigma2_xy, g)

# output average energies, fluctuations and specific heat at all temperatures

outf = open("wham_out.dat", "w")
min_T = MIN_T / T_TO_K
max_T = MAX_T / T_TO_K

# A*w
Aw = zeros((K, N), dtype=float128)

# A^2*w
A2w = zeros((K, N), dtype=float128)
for single_T in linspace(min_T, max_T, NUM_T):
	single_beta = 1/(K_B*single_T*T_TO_K)
	w = zeros((K, N), dtype=float128)

	# exponent[l, m] matrix
	exponent = matrix(f, dtype=float128).T - matrix(beta, dtype=float128).T*matrix(U)

	# D[m] array
	D = sum(N_beta_eff.A * exp(exponent + single_beta * matrix(U)).T.A, axis=1)
	weights = H_eff / D
	print("weights =", weights)

	for k in xrange(K):
		for (i, m) in enumerate(bins[k, :]):
			w[k, i] = weights[m] / sum(H[m, :])
			Aw[k, i] = w[k, i] * U[m]

	E_avg_1 = sum(Aw)/sum(w)
	print("E_avg_1 =", E_avg_1)

	# move energy zero point to E_avg to minimize d2Cv (see later)
	for k in xrange(K):
		for (i, m) in enumerate(bins[k, :]):
			Aw[k, i] = w[k, i] * (U[m] - E_avg_1)
			A2w[k, i] = w[k, i] * (U[m] - E_avg_1)**2

	E_avg = sum(Aw)/sum(w)
	E2_avg = sum(A2w)/sum(w)
	print("E_avg =", E_avg)
	print("E2_avg =", E2_avg)

	(Aw_avg, Aw_avg, sigma2_Aw_Aw, g_Aw_Aw) = cross_corr(Aw, Aw)
	(w_avg, w_avg, sigma2_w_w, g_w_w) = cross_corr(w, w)
	(Aw_avg, w_avg, sigma2_Aw_w, g_Aw_w) = cross_corr(Aw, w)

	##### d2E
	# this correction is needed to avoid negative d2E values (see [Chodera2007])
	max_g_Aw_w = sqrt(g_Aw_Aw * g_w_w) * abs(sqrt(sigma2_Aw_Aw*sigma2_w_w) / sigma2_Aw_w)
	idx = (g_Aw_w > max_g_Aw_w)
	g_Aw_w[idx] = max_g_Aw_w[idx]

	print("g_Aw_Aw =", g_Aw_Aw)
	print("g_Aw_w =", g_Aw_w)
	print("g_w_w =", g_w_w)

	X = N * sum(Aw_avg)
	Y = N * sum(w_avg)
	A = X/Y
	d2X = sigma2_Aw_Aw / (N / g_Aw_Aw)
	d2Y = sigma2_w_w / (N / g_w_w)
	dXdY = sigma2_Aw_w / (N / g_Aw_w)
	delta2X = sum(d2X) * N**2
	delta2Y = sum(d2Y) * N**2
	deltaXdeltaY = sum(dXdY) * N**2

	d2E = A**2 * ((delta2X / X**2) + (delta2Y / Y**2) - (2*deltaXdeltaY / (X*Y)))
	print("d2E =", d2E)

	##### d2E2

	(A2w_avg, A2w_avg, sigma2_A2w_A2w, g_A2w_A2w) = cross_corr(A2w, A2w)
	(A2w_avg, w_avg, sigma2_A2w_w, g_A2w_w) = cross_corr(A2w, w)

	max_g_A2w_w = sqrt(g_A2w_w * g_w_w) * abs(sqrt(sigma2_A2w_w*sigma2_w_w) / sigma2_A2w_w)
	idx = (g_A2w_w > max_g_A2w_w)
	g_A2w_w[idx] = max_g_A2w_w[idx]

	print("g_A2w_A2w =", g_A2w_A2w)
	print("g_A2w_w =", g_A2w_w)
	print("g_w_w =", g_w_w)

	X = N * sum(A2w_avg)
	Y = N * sum(w_avg)
	A2 = X/Y
	d2X = sigma2_A2w_A2w / (N / g_A2w_A2w)
	d2Y = sigma2_w_w / (N / g_w_w)
	dXdY = sigma2_A2w_w / (N / g_A2w_w)
	delta2X = sum(d2X) * N**2
	delta2Y = sum(d2Y) * N**2
	deltaXdeltaY = sum(dXdY) * N**2

	d2E2 = A2**2 * ((delta2X / X**2) + (delta2Y / Y**2) - (2*deltaXdeltaY / (X*Y)))
	print("d2E2 =", d2E2)

	d2Cv = (d2E2 + 2*E_avg*d2E) * (single_beta / (single_T*T_TO_K))**2
	print("dCv =", sqrt(d2Cv))

	# Cv

	Cv = (E2_avg - E_avg*E_avg) * single_beta / (single_T*T_TO_K)

	print("%10.3f %10.3f %10.3f %10.3f %10.3lg" % (single_T*T_TO_K, E_avg_1+all_min, E2_avg, Cv, sqrt(d2Cv)))
	outf.write("%.3f %.3f %.3f %.3f %.3g\n" % (single_T*T_TO_K, E_avg_1+all_min, E2_avg, Cv, sqrt(d2Cv)))

outf.close()
