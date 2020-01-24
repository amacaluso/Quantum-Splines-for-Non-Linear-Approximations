import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
import seaborn as sns
import random
from pylab import *
from scipy.interpolate import CubicSpline, splev, splrep


from qiskit.aqua import run_algorithm
from qiskit.aqua.input import LinearSystemInput
from qiskit.quantum_info import state_fidelity
from qiskit.aqua.algorithms.classical import ExactLSsolver
from qiskit.quantum_info import state_fidelity
from qiskit import BasicAer
from qiskit.aqua import QuantumInstance
from qiskit.aqua.algorithms.single_sample import HHL
from qiskit.aqua.utils import random_hermitian

import math

def sigmoid(x):
  return 1 / (1 + math.exp(-4*x))

def fidelity(hhl, ref):
    solution_hhl_normed = hhl / np.linalg.norm(hhl)
    solution_ref_normed = ref / np.linalg.norm(ref)
    fidelity = state_fidelity(solution_hhl_normed, solution_ref_normed)
    print("fidelity %f" % fidelity)
    return fidelity




params = {
    'problem': {'name': 'linear_system'},
    'algorithm': {'name': 'HHL'},
    'eigs': {'expansion_mode': 'suzuki',
             'expansion_order': 2,
             'name': 'EigsQPE',
             'num_ancillae': 3,
             'num_time_slices': 50 },
    'reciprocal': { 'name': 'Lookup'},
    'backend': { 'provider': 'qiskit.BasicAer',
                 'name': 'statevector_simulator'}
}

def poly_data(x):
    return pd.Series([1, x, x**2, x**3])


def lin_data(x):
    return pd.Series([1, x])