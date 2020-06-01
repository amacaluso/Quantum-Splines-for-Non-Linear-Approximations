# Quantum Spline

This repository contains the code to reproduce the results presented in the paper 
[Quantum Splines for Non-Linear Approximation](https://dl.acm.org/doi/pdf/10.1145/3387902.3394032),
published in the *Proceedings* of the 
[17th ACM International Conference on Computing Frontiers (CF'20)](http://www.computingfrontiers.org/2020/) 
and freely available on the [ACM Digital Library](https://dl.acm.org/doi/abs/10.1145/3387902.3394032).

# Description

The idea underpinning the work is to leverage *spline* functions to overcome the constraint to unitary transformations
and being able to approximate non-linear functions on a quantum device. To demonstrate this, we formulated the problem
according to a B-Spline parametrisation and we investigated 4 popular non-linear activation functions commonly used for Neural Networks, 
namely *sigmoid*, *hyperbolic tangent (tanh)*, *rectified linear unit (relu)* and *exponential linear unit (elu)*.

# Usage

The code is organised in four different scripts, one per activation function. 
Specifically, *Sigmoid.py, Tanh.py, Relu.py, Elu.py* compute a B-spline approximation of the corrispondent function using three approaches: 
- *Full quantum*: uses the HHL quantum algorithm to estimates B-spline coeffients and a second quantum circuit for function evaluation
- *Quantum Hybrid*: uses the HHL algorithm to estimates B-Spline coefficients and classical approach for function evaluation
- *Classical*: coefficients estimates and function evaluation are performed under the classical computing paradigm

The script *Utils.py* is used to import the needed packages and all of the custom routines for function evaluation.

The script *Utils_Spline.py* contains the custom routines for B-Spline coefficients estimation.

The script *Viz_complexity.py* plots the comparison between quantum HHL and classical algorithm for matrix inversion in terms of computational complexity.

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

