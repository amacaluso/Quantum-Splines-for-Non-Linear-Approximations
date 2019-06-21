from qiskit.aqua import run_algorithm
from qiskit.aqua.input import LinearSystemInput
from qiskit.quantum_info import state_fidelity
from qiskit.aqua.algorithms.classical import ExactLSsolver
import numpy as np

params = {
    'problem': {
        'name': 'linear_system'
    },
    'algorithm': {
        'name': 'HHL'
    },
    'eigs': {
        'expansion_mode': 'suzuki',
        'expansion_order': 2,
        'name': 'EigsQPE',
        'num_ancillae': 3,
        'num_time_slices': 50
    },
    'reciprocal': {
        'name': 'Lookup'
    },
    'backend': {
        'provider': 'qiskit.BasicAer',
        'name': 'statevector_simulator'
    }
}

def fidelity(hhl, ref):
    solution_hhl_normed = hhl / np.linalg.norm(hhl)
    solution_ref_normed = ref / np.linalg.norm(ref)
    fidelity = state_fidelity(solution_hhl_normed, solution_ref_normed)
    print("fidelity %f" % fidelity)


matrix = [[10, 0], [0, 0.1]]
vector = [1, 4]
params['input'] = {
    'name': 'LinearSystemInput',
    'matrix': matrix,
    'vector': vector
}


result = run_algorithm(params)
print("solution ", np.round(result['solution'], 5))

result_ref = ExactLSsolver(matrix, vector).run()
print("classical solution ", np.round(result_ref['solution'], 5))

print("probability %f" % result['probability_result'])
fidelity(result['solution'], result_ref['solution'])

params2 = params
params2['reciprocal'] = {
    'scale': 0.5
}

result = run_algorithm(params2)
print("solution ", np.round(result['solution'], 5))

result_ref = ExactLSsolver(matrix, vector).run()
print("classical solution ", np.round(result_ref['solution'], 5))

print("probability %f" % result['probability_result'])
fidelity(result['solution'], result_ref['solution'])


from qiskit import BasicAer
from qiskit.aqua import QuantumInstance
from qiskit.aqua.algorithms.single_sample import HHL
from qiskit.aqua.utils import random_hermitian


params5 = params
params5['algorithm'] = {
    'truncate_powerdim': False,
    'truncate_hermitian': False
}
params5['reciprocal'] = {
    'name': 'Lookup',
    'negative_evals': True
}
params5['eigs'] = {
    'expansion_mode': 'suzuki',
    'expansion_order': 2,
    'name': 'EigsQPE',
    'negative_evals': True,
    'num_ancillae': 6,
    'num_time_slices': 70
}
params5['initial_state'] = {
    'name': 'CUSTOM'
}
params5['iqft'] = {
    'name': 'STANDARD'
}
params5['qft'] = {
    'name': 'STANDARD'
}


# set the random seed to get the same pseudo-random matrix for every run
np.random.seed(1)
matrix = XX + np.diag(np.repeat(100, 4)) #random_hermitian(4)
max(np.linalg.svd(matrix)[1])/min(np.linalg.svd(matrix)[1])

vector = [1, 2, 3, 1]

print("random matrix:")
m = np.array(matrix)
print(np.round(m, 3))

algo_input = LinearSystemInput(matrix=matrix, vector=vector)
hhl = HHL.init_params(params5, algo_input)
backend = BasicAer.get_backend('statevector_simulator')
quantum_instance = QuantumInstance(backend=backend)
result = hhl.run(quantum_instance)
print("solution ", np.round(result['solution'], 5))

result_ref = ExactLSsolver(matrix, vector).run()
print("classical solution ", np.round(result_ref['solution'], 5))

print("probability %f" % result['probability_result'])
fidelity(result['solution'], result_ref['solution'])


print("circuit_width", result['circuit_info']['width'])
print("circuit_depth", result['circuit_info']['depth'])


XX_inv = np.linalg.inv( X.values.transpose().dot(X.values) )
XY = X.values.transpose().dot(y)

XY = vector


XX_inv.dot(vector)












