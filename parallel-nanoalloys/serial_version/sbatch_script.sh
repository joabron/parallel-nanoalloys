#!/bin/bash

#SBATCH --job-name="Py_pi"
#SBATCH --time=00:30:00
#SBATCH --ntasks=20
#SBATCH --cpus-per-task=1
#SBATCH --partition=compute
#SBATCH --mem-per-cpu=1GB
#SBATCH --account=education-EEMCS-courses-TW3725TU

export CPATH=/apps/arch/2022r2/software/linux-rhel8-skylake_avx512/gcc-8.5.0/openmpi-4.1.1-fezcq73heq4rzzsbcumuq5xx4v5asv45/include:$CPATH

export CPATH=/apps/arch/2022r2/software/linux-rhel8-skylake_avx512/gcc-8.5.0/mpfr-4.1.0-ok2rrvi5iz3oeokaopoa4ntoqj4aja2t/include:$CPATH

export LIBRARY_PATH=/apps/arch/2022r2/software/linux-rhel8-skylake_avx512/gcc-8.5.0/mpfr-4.1.0-ok2rrvi5iz3oeokaopoa4ntoqj4aja2t/lib:$LIBRARY_PATH

export LD_LIBRARY_PATH=/apps/arch/2022r2/software/linux-rhel8-skylake_avx512/gcc-8.5.0/mpfr-4.1.0-ok2rrvi5iz3oeokaopoa4ntoqj4aja2t/lib:$LD_LIBRARY_PATH

export CPATH=/apps/arch/2022r2/software/linux-rhel8-skylake_avx512/gcc-8.5.0/gmp-6.2.1-yzrwjlbp4peag6mbwuffwdhhxkly7ktd/include:$CPATH

export LIBRARY_PATH=/apps/arch/2022r2/software/linux-rhel8-skylake_avx512/gcc-8.5.0/gmp-6.2.1-yzrwjlbp4peag6mbwuffwdhhxkly7ktd/lib:$LIBRARY_PATH

export LD_LIBRARY_PATH=/apps/arch/2022r2/software/linux-rhel8-skylake_avx512/gcc-8.5.0/gmp-6.2.1-yzrwjlbp4peag6mbwuffwdhhxkly7ktd/lib:$LD_LIBRARY_PATH

module load 2022r2
module load gcc
module load gmp/6.2.1
module load mpfr/4.1.0
#module load openmpi
module load python
#module load py-mpi4py

rm *.csv
rm *.traj

# python sequential.py
srun python run_replica.py

python collapse.py

srun python serial_wham.py
