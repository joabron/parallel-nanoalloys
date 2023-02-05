## parallel-nanoalloys
A parallel molecular dynamics simulation algorithm for studying the thermal stability of nanoalloys. A repository for all the code used in the TW3725TU Final Project of the Computational Science and Engineering (CSE) Minor. This code implements a combination of various sub-components for the investigation of the thermal stability of nanoalloys: a nanoalloy initialization, followed by molecular dynamics (MD) runs in a Replica Exchange (RE) framework, followed by the Weigthed Histogram Analysis Method (WHAM), a statistical analysis tool for generating the graph of heat capcity vs temperature, from the temperature and potential energy data output of RE.

A literature review, containing the theoretical background of the project, and a report of the project, serving as the extended user guide of the code and containing a detailed descritpion of the code structure, can be requested with the authors. 

**Abstract**: Bimetallic nanoalloys are nano-scale clusters of atoms that can greatly vary in composition and shape. They have shown promising results in catalysis, biomedical and optical applications. Exploration of their thermodynamic stability through physical experiments is both difficult and expensive, thus justifying the need for molecular dynamics (MD) simulations to quicken the process of finding the most stable structures with specified properties of interest. A novel streamlined Replica Exchange and Weighted Histogram Analysis Method software is developed as a tool for studying thermal stability of custom configurations and compositions of bimetallic nanoalloys. The software is parallelized and tested on the DelftBlue supercomputer, for which the run-times are analyzed and demonstrate a sub-linear speedup.

**Keywords**: molecular dynamics simulation, nanoal-
loys, thermal stability, inter-atomic potentials, Python
programming, MPI-parallelization, high-performance
computing

## Usage
There are two versions of the program: the serial version and the parallel version. The [serial_version](./parallel-nanoalloys/serial_version) includes all the python files which run sequentially. It is a good starting point for any new contributions regarding parallelization. On the other hand, [parallel_version](./parallel-nanoalloys/parallel_version) includes a combination of files - some can be run in parallel and others not. This is in detail described in the section *structure of the code*. Both directories include all the necessary files for running the whole program framework: from nanoalloy initialization to the reading of melting temperature from the heat capcity vs temperature graph.

NOTE: to run the parallel version on DelftBlue supercomputer, it suffices to submit a job as described in `DelftBlue_user_guide.txt`. This file contains all the necessary steps, such as library downloads, path exports and DelftBlue supercomputer specifications, such as number of cores to be used. 

## Code structure
`filename.py` - <must be run sequentially/parallel, or is supporting file (not to be executed on its own)><description of the functionality>
  
  `collapse.py` - sequential, "collapses" all the csv log files generated by different replicas into one csv file - to be read by parallel_wham.py for statistical data analysis
  
  `coreshell.py` - supporting, helps initialize coreshell nanoalloy.
  
  `initialize_nanoalloy.py` - supporting, allows to initialize nanoalloy based on inputs.txt file
  
  `inputs.txt` - supporting, contains all the user specifications regarding the research, such as element type, what type of nanoalloy, etc.
  
  `parallel_wham.py` - parallel/supporting, contains parallel implementation of the Weighted Histogram Analysis Method
  
  `py_bigfloat.py` - supporting, wrapper class for python's `BigFloat` library, so that it functions in `numpy` fashion
  
  `replica_exchange_parallel.py` - parallel/supporting, contains parallel implementation of Replica Exchange algorithm
  
`run_replica.py` - parallel, reads nanoalloy initialization inputs from inputs.txt and calls to execute replica_exchange_parallel.py
  
`sbatch_script.sh` - script to be submitted to scheduler on DelftBlue supercomputer, user can modify number of cores, runtime, etc.

## Installation
To run the complete version of the whole program (serial or parallel version), download all the files from the corresponding directory: [serial_version](./parallel-nanoalloys/serial_version) or [parallel_version](./parallel-nanoalloys/parallel_version). Then run the files in the order specified by the `sbatch_script.sh`

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

## Authors and acknowledgment

The following authors all contributed equally to the project: Adam Axelsen, Adrian Beňo, Joachim Bron, Philip Vos. 
  
The authors would like to thank Dr. Dennis Plagin for his help and guidance during the project as well as Dr. Neil Budko for the opportunity to work on this project in the scope of the CSE minor.

## License

[MIT](./LICENSE.md)
