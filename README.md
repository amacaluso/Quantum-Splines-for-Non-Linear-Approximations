# Quantum Spline

This repository contains the code to reproduce the results in the paper [Quantum Splines for Non-Linear Approximation](https://dl.acm.org/doi/pdf/10.1145/3387902.3394032),
under publication on [ACM Digital Library](https://dl.acm.org/) for the [ACM International Conference on Computing Frontiers 2020](http://www.computingfrontiers.org/2020/).

# Description

The code is organised with one script per activation function. In particular, *Sigmoid.py, Tanh.py, Relu.py, Elu.py* compute a B-spline approximation of the corrispondent function using three approaches: 
- *Full quantum*: uses the HHL quantum algorithm to estimates B-spline coeffients and a second quantum circuit for function evaluation
- *Quantum Hybrid*: uses the HHL algorithm to estimates B-Spline coefficients and classical approach for function evaluation
- *Classical*: coefficients estimates and function evaluation are performed with classical device

The script *Utils.py* contains the import of the needed packages and all the custom routines for function evaluation.

The script *Utils_Spline.py* contains the custom routines for B-Spline coefficients estimates.

The script *Viz_complexity.py* plots the comparison in terms of computational complexity between quantum HHL and classical algorithm for matrix inversion.

The script *All_viz.py* plots in one plot all the activation functions.

