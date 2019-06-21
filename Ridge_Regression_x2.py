from Utils import *



''' Ridge Regression with 2 variables'''

x1 = np.random.normal(0 , 0.1, 50); x2 = np.random.normal(0 , 0.2, 50)

beta_0 = 0; beta_1 = 1; beta_2 = 2

y = beta_0 + beta_1*x1 + beta_2*x2 + np.random.normal(0 , 0.1, 50)
X = pd.concat( [pd.Series(x1), pd.Series(x2)], axis = 1 )
reg = LinearRegression().fit(X, y)
reg.score(X, y)

XX = X.values.transpose().dot(X.values)
print('The condition number is: ', np.round( max(np.linalg.svd(XX)[1])/min(np.linalg.svd(XX)[1]), 2) )


XX_inv = np.linalg.inv( X.values.transpose().dot(X.values) )
XY = [1, 4 ] #X.values.transpose().dot(y)

XX_inv.dot(XY)

print('The regression coefficients are:', np.round( reg.coef_, 2))

matrix =  [[20, 0], [0, 11]]
vector = [1, 2]

np.linalg.inv( matrix ).dot( vector )

# params = {
#     'problem': {'name': 'linear_system'},
#     'algorithm': {'name': 'HHL'},
#     'eigs': {'expansion_mode': 'suzuki',
#              'expansion_order': 2,
#              'name': 'EigsQPE',
#              'num_ancillae': 3,
#              'num_time_slices': 50 },
#     'reciprocal': { 'name': 'Lookup'},
#     'backend': { 'provider': 'qiskit.BasicAer',
#                  'name': 'statevector_simulator'}
# }

print('The condition number is: ', np.round( max(np.linalg.svd(matrix)[1])/min(np.linalg.svd(matrix)[1]), 2) )



params['input'] = {
    'name': 'LinearSystemInput',
    'matrix': matrix,
    'vector': vector }


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
















'''Ridge Regression with 2 variables - ill conditioned matrix'''

x1 = np.random.normal(0 , 0.1, 50)
x2 = x1*2 + np.random.normal(0 , 0.1, 50)

print( 'Coefficiente di correlazione:', np.round( np.corrcoef( [x1, x2] )[0][1], 3) )

y = beta_0 + beta_1*x1 + beta_2*x2 + np.random.normal(0 , 0.1, 50)

X = pd.concat( [pd.Series(x1), pd.Series(x2)], axis = 1 )
reg = LinearRegression().fit(X, y)
reg.score(X, y)

XX = X.values.transpose().dot(X.values)

np.round( max(np.linalg.svd(XX)[1])/min(np.linalg.svd(XX)[1]), 2 )

XX_inv = np.linalg.inv( X.values.transpose().dot(X.values) )
XY = np.round( X.values.transpose().dot(y))

print( np.round( XX_inv.dot(XY), 2) )

reg = LinearRegression().fit(X, y)
np.round( reg.score(X, y), 2)

np.round( reg.coef_, 2 )






















