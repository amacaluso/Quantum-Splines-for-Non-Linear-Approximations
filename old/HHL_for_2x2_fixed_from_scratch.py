# Import packages
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import execute
from qiskit import BasicAer,IBMQ
from math import pi
from qiskit.tools.visualization import plot_histogram, plot_state_city

from qiskit.tools import qcvv as tomo

# Define Circuit
backend = BasicAer.get_backend('qasm_simulator')
qr = QuantumRegister(4, name="qr")
cr = ClassicalRegister(4, name="cr")
circuit = QuantumCircuit(qr, cr, name="HLL_2x2")

# This is how I want keep my qubit
# |ancilla >
# |C >
# |C >
# |b >
# |b> is by default 0

# Phase Estimation
circuit.h(qr[3])
circuit.cx(qr[3], qr[2])
circuit.cx(qr[2], qr[1])
circuit.x(qr[2])
circuit.swap(qr[1], qr[2])
circuit.barrier()

# Controlled Rotation
circuit.cu3(pi,0, 0, qr[2], qr[0])
circuit.cu3(pi/3, 0, 0, qr[1], qr[0])
circuit.barrier()

# Inverse Phase Estimation
circuit.swap(qr[1], qr[2])
circuit.x(qr[2])
circuit.cx(qr[2], qr[1])
circuit.cx(qr[3], qr[2])
circuit.h(qr[3])
circuit.barrier()

circuit.measure(qr, cr)
print(circuit)

backend_qasm = BasicAer.get_backend('qasm_simulator')
backend_state = BasicAer.get_backend('statevector_simulator')

# Execute the circuit on the qasm simulator.
job = execute(circuit, backend_qasm, shots=8192)
result = job.result()
counts = result.get_counts(circuit)
print(counts)
plt = plot_histogram(counts)
plt.show()
# # Construct state tomography set for measurement of qubits [0, 1] in the Pauli basis
tomo_set = tomo.process_tomography_set([3], meas_basis='Pauli')
# Add the state tomography measurement circuits to the Quantum Program
tomo_circuits = tomo.create_tomography_circuits(circuit, tomo_set)
tomo_job = execute(tomo_circuits, backend=backend_qasm, shots=1024)
tomo_results = tomo_job.result()
#
tomo_data = tomo.tomography_data(tomo_results, 'HHL_circuit', tomo_set)
rho_fit = tomo.fit_tomography_data(tomo_data)
# plot the state
plt = plot_state_city(rho_fit)
plt.show()







