from Utils import *


# +++++ Random Seed +++++ #
# set the random seed to get
# the same pseudo-random matrix for every run
np.random.seed(1)


# Fidelity is useful to check whether two states are same or not.
# For quantum (pure) states, the fidelity is the squared overlap between them
# %%
params = {
    'problem': {'name': 'linear_system'},
    'algorithm': {'name': 'HHL'},
    'eigs': {'expansion_mode': 'suzuki',
             'expansion_order': 2,
             'name': 'EigsQPE',
             'num_ancillae': 3,
             'num_time_slices': 50},
    'reciprocal': {'name': 'Lookup'},
    'backend': {'provider': 'qiskit.BasicAer',
                'name': 'statevector_simulator'}
}


''' Ridge Regression with 4 variables'''

# Parameters for distributions

N = 100 # n° training points
dw = 0
up = 10

mu_eps = 0
std_eps = 1

x1 = np.random.uniform(dw, up, N)
x2 = np.random.uniform(dw, up, N)
x3 = np.random.uniform(dw, up, N)
x4 = np.random.uniform(dw, up, N)

beta_1 = 1;
beta_2 = 2;
beta_3 = 2;
beta_4 = 1;

y = beta_1 * x1 + beta_2 * x2 + + beta_3*x3 + beta_4*x4 + np.random.normal(mu_eps, std_eps, N)
X = pd.concat([pd.Series(x1), pd.Series(x2), pd.Series(x3), pd.Series(x4)], axis=1)

XX = X.values.transpose().dot(X.values)

print('\n Matrix correlation \n', np.corrcoef([x1, x2, x3, x4]), '\n')

# Condition number
k = max(np.linalg.svd(XX)[1]) / min(np.linalg.svd(XX)[1])
print('condition number = ', np.round(k, 2))

'''Fitting Linear Regression using skl'''

reg = LinearRegression().fit(X, y)

print('R squared =', np.round(reg.score(X, y), 2), '\n\n')
print('Coeffs (skl) = ', reg.coef_)

'''Fitting Linear Regression by hand'''

XX_inv = np.linalg.inv(X.values.transpose().dot(X.values))
XY = X.values.transpose().dot(y)
print('Coeffs (by hand) = ', XX_inv.dot(XY))

# %%
# Scatter plot matrix
# sns.set(style="ticks")
# df = pd.concat( [pd.Series(x1), pd.Series(x2), pd.Series(x3), pd.Series(x4), pd.Series(y)], axis = 1 )
# sns.pairplot(df, diag_kind="kde") #, hue="species")
# %%
'''Quantum algorithm for linear regression (design matrix 2x2)'''

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

matrix = XX.tolist()
vector = XY.tolist()

# Condition number
k = max(np.linalg.svd(XX)[1]) / min(np.linalg.svd(XX)[1])
print('condition number = ', np.round(k, 2))

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
# %%
# set the random seed to get the same pseudo-random matrix for every run
# np.random.seed(1)

# matrix = XX.tolist()
# vector = XY.tolist()

# # Condition number
# k = max(np.linalg.svd(XX)[1])/min(np.linalg.svd(XX)[1])
# print( 'condition number = ', np.round( k, 2) )
# %%
''' Ill-conditioned ridge regression'''

alpha_1 = 1
alpha_2 = 3
x3 = alpha_1 * x2 + alpha_2 * x1 + np.random.normal(m_e, s_e, N)

gamma_1 = 1
gamma_2 = 2
x4 = gamma_1 * x2 + gamma_2 * x1 + np.random.normal(m_e, s_e, N)

# %%
# matrix = XX.tolist() ; print( 'Matrix XX \n', np.round( matrix ) )# [[2, 0], [0, 1]]
# vector = XY.astype(int).tolist() ; print( '\n Vector XY \n', np.round( vector ) ) # [1, 4]

# np.linalg.inv( matrix ).dot( vector )
# print('The Pearson coefficient between x1 and x2 is \n: ', np.round( np.corrcoef(x1, x2)[0,1], 3 ) )

X = pd.concat([pd.Series(x1), pd.Series(x2), pd.Series(x3), pd.Series(x4)], axis=1)
y = beta_1 * x1 + beta_2 * x2 + beta_3 * x3 + beta_4 * x4 + np.random.normal(m_e, s_e, N)

XX = X.values.transpose().dot(X.values)

print('\n Matrix correlation \n', np.corrcoef([x1, x2, x3, x4]), '\n')

# Condition number
k = max(np.linalg.svd(XX)[1]) / min(np.linalg.svd(XX)[1])
print('condition number = ', np.round(k, 2))

# matrice = XX # + np.diag(np.repeat(100, 4, axis=0))
# print( np.round(XX) )

# k = max(np.linalg.svd(matrice)[1])/min(np.linalg.svd(matrice)[1])
# print( 'condition number = ', np.round( k, 2) )

'''Fitting Linear Regression using skl'''

reg = LinearRegression().fit(X, y)

print('R squared =', np.round(reg.score(X, y), 2), '\n\n')
print('Coeffs (skl) = ', reg.coef_)

'''Fitting Linear Regression by hand'''

XX_inv = np.linalg.inv(X.values.transpose().dot(X.values))
XY = X.values.transpose().dot(y)
print('Coeffs (by hand) = ', XX_inv.dot(XY))

# %%

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

matrix = XX.tolist()
vector = XY.tolist()

# Condition number
k = max(np.linalg.svd(XX)[1]) / min(np.linalg.svd(XX)[1])
print('condition number = ', np.round(k, 2))

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
# %%
'''Introduction of the penalty for ill-conditioned case'''

lambda_penalty = 10 ** 7
matrix = (XX + np.diag(np.repeat(lambda_penalty, 4, axis=0))).tolist()
print(np.round(matrix))

k = max(np.linalg.svd(matrix)[1]) / min(np.linalg.svd(matrix)[1])
print('(new) condition number = ', np.round(k, 2))

# %%
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
# %%
print("circuit_width", result['circuit_info']['width'])
print("circuit_depth", result['circuit_info']['depth'])
# %%
# 3D plot
# from mpl_toolkits import mplot3d
# %matplotlib inline
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D


# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax = plt.axes(projection='3d')

# # Data for a three-dimensional line
# # zline = np.linspace(0, 15, 1000)
# # xline = np.sin(zline)
# # yline = np.cos(zline)
# #ax.plot3D(xline, yline, zline, 'gray')

# # Data for a three-dimensional line
# zline = beta_0 + beta_1*x1 + beta_2*x2
# xline = x1
# yline = x2
# # ax.plot3D(xline, yline, zline, 'gray')


# # Data for three-dimensional scattered points
# zdata = beta_0 + beta_1*x1 + beta_2*x2
# xdata = x1
# ydata = x2
# ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens')

# plt3d = plt.figure().gca(projection='3d')
# plt3d.plot_surface(zline, yline, xline)
# %%
