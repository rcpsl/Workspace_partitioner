# NN-Verification
Towards Formal Verification of Neural Network Controlled Autonomous Systems

-----------------
1. Introduction

This Python package contains the implementation of the algorithms described in the paper "Towards Formal Verification of Neural Network Controlled Autonomous Systems", HSCC 2019. This file describes the contents of the package, and provides instructions regarding its use. This package is used to generate all the tables documented in Section 6 (entitled “RESULTS”) in the paper, i.e., Table 1, Table 2, Table 3, and Table 4. HSCC19Paper50.pdf is a copy of the paper.

-----------------
2. Installation

The tool was written for Python 2.7.10. Earlier versions may be sufficient, but not tested. In addition to Python 2.7.10, the solver requires the following:

- CPLEX Optimizer (https://www.ibm.com/analytics/cplex-optimizer), an optimization package for LP, QP, QCP, MIP, provided for free for academic use via the IBM Academic Initiative.

- CPLEX Python Interface. To install the Python API for CPLEX, please follow the instructions at the following URL:

https://www.ibm.com/support/knowledgecenter/SSSA5P_12.7.0/ilog.odms.cplex.help/CPLEX/GettingStarted/topics/set_up/Python_setup.html

- pycddlib, a Python wrapper for cddlib that is used for transformation between H-representation and V-representation of a general convex polyhedron. All experiments use pycddlib 2.0.0 at the following URL. The newest version pycddlib 2.0.1 is not tested and may not work. 

https://pypi.org/project/pycddlib/2.0.0/


---------------------------------------------------
3. Running the experiments described in the paper

The package includes four experiments, each of which generates results reported in a table of the paper. Run scripts for each experiment is located in the subfolder named by the corresponding table index. Specifically, script `run.sh` in each subfolder generates results of a whole table, and the output is saved in file `results.txt` in the same subfolder. In case only want to generate results for certain lines of a table, one can only run tests with same indices as line numbers in the table by commenting out other tests in `run.sh`.

Run scripts should be run from current directory, i.e. the top-most directory of this package. In case permission is denied, please make sure that run scripts are in executable mode. Details about each experiment and the expected results can be found below. Results in the paper are generated using a MacBook Pro laptop with the following specifications:
- Processor: 2.5 GHz Intel Core i7
- Memory: 16 GB 1600 MHz DDR3
Different runs for the same test case can lead to slightly different execution times.


---------------------------------------------------
4. Description of the Experiments

4.1 Experiment 1

This experiment summarizes the scalability of the workspace partitioning algorithm and results are reported in Table 1 in the paper. To generate the results, run the following commands
```
cd NN-Verification/Code/v1
./table1/run.sh
```
The file `table1/results.txt` has the results of Table 1.

4.2 Experiment 2

In this experiment, the pre-processing step is for the same region and using different neural network architectures. The results of this experiment correspond to Table 2 in the paper.
To generate the results, run the following commands
```
cd NN-Verification/Code/v1
./table2/run.sh
```
The file `table2/results.txt` has the results of Table 2.

4.3 Experiment 3

In this experiment the pre-processing step is done starting from different same region and using the same neural network architectures. The results of this experiment correspond to Table 3 in the paper.
To generate the results, run the following commands
```
cd NN-Verification/Code/v1
./table3/run.sh
```
The file `table3/results.txt` has the results of Table 3.

4.4 Experiment 4

Experiment 4 shows performance comparison between our proposed strategy that uses counterexamples obtained from pre-processing and SMC encoding without preprocessing. To generate the results, run the following commands
```
cd NN-Verification/Code/v1
./table4/run.sh
```
The file `table4/results.txt` has the results of Table 4.

if permission is denied when running any experiment, make sure that run.sh is in executable mode.

4. Description of the Experiments
---------------------------------------------------
