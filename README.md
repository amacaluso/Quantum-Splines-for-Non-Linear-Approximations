# Quantum Spline

This repository contains the code to reproduce the results presented in the paper [Quantum Splines for Non-Linear Approximation](https://dl.acm.org/doi/pdf/10.1145/3387902.3394032),
published in the *Proceedings* of the [17th ACM International Conference on Computing Frontiers (CF'20)](http://www.computingfrontiers.org/2020/) and freely available on the [ACM Digital Library](https://dl.acm.org/doi/abs/10.1145/3387902.3394032).

# Description

The code is organised with one script per activation function. In particular, *Sigmoid.py, Tanh.py, Relu.py, Elu.py* compute a B-spline approximation of the corrispondent function using three approaches: 
- *Full quantum*: uses the HHL quantum algorithm to estimates B-spline coeffients and a second quantum circuit for function evaluation
- *Quantum Hybrid*: uses the HHL algorithm to estimates B-Spline coefficients and classical approach for function evaluation
- *Classical*: coefficients estimates and function evaluation are performed with classical device

The script *Utils.py* contains the import of the needed packages and all the custom routines for function evaluation.

The script *Utils_Spline.py* contains the custom routines for B-Spline coefficients estimates.

The script *Viz_complexity.py* plots the comparison in terms of computational complexity between quantum HHL and classical algorithm for matrix inversion.

The script *All_viz.py* plots in one plot all the activation functions.

# Installation

In order to run the code and reproduce the results of the paper, it is recommended to re-create the same testing environment following the procedure below.

*Note: tested on linux OS only; it assumes Anaconda is installed*

 - First, clone the repository:
 
 `git clone https://github.com/amacaluso/Quantum_Spline.git`
 
  - Second, create a conda environment fron scratch using the *environment.yml* specs:
  
  ```
# enter the repository
cd Quantum_Spline

# create an environment named QSplines with desired dependencies
conda env create -f environment.yml 
```
 - Third, install *qiskit* package:
 
 ```angular2
# activate the environmet
conda activate QSplines

# install qiskit 0.13
pip install qiskit==0.13
```

