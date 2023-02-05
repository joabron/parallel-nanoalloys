# parallel-nanoalloys
Parallel algorithms for studying the thermal stability of nanoalloys. A repository for all the code used in the TW3725TU Final Project of the Computational Science and Engineering Minor 

# Usage
There are two versions of the program. The serial and parallel. Directory serial_version includes all the python files which run sequentially. It is a good starting point for any new contributions regarding parallelization. Directory parallel_version includes combination of files - some can be run in parallel and others not. This is in detail described in section below - structure of the code. Both directories include all the necessary files for running the whole program framework: from nanoalloy initialization to reading of melting temperature from graph.

NOTE: to run the parallel version on DelftBlue supercomputer, it suffices to submit a job as described in 

# structure of the code

collapse.py - 
coreshell.py -
initialize_nanoalloy.py - 
inputs.txt - 
parallel_wham.py - 
py_bigfloat.py -
replica_exchange_parallel.py -
run_replica.py -
sbatch_script.sh - 

## Installation
To run the complete version of the whole program (serial or parallel version), it is only necessary to download all the files from the corresponding directory: serial_version or parallel_version.

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

## Authors and acknowledgment

The following authors all contributed equally to the project: Adam Axelsen, Adrian Beňo, Joachim Bron, Philip Vos

## License

[MIT](https://choosealicense.com/licenses/mit/)
