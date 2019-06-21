import numpy as np
# import matplotlib.pyplot as plt
import random
import pandas as pd
from sklearn.linear_model import LinearRegression


''' Ridge Regression with 2 variables'''

x1 = np.random.normal(0 , 0.1, 50)
x2 = np.random.normal(0 , 0.2, 50)

beta_0 = 0
beta_1 = 1
beta_2 = 2

y = beta_0 + beta_1*x1 + beta_2*x2 + np.random.normal(0 , 0.1, 50)


X = pd.concat( [pd.Series(x1), pd.Series(x2)], axis = 1 )
reg = LinearRegression().fit(X, y)
reg.score(X, y)


XX = X.values.transpose().dot(X.values)

np.round( max(np.linalg.svd(XX)[1])/min(np.linalg.svd(XX)[1]), 2)

XX_inv = np.linalg.inv( X.values.transpose().dot(X.values) )
XY = X.values.transpose().dot(y)

XX_inv.dot(XY)

reg = LinearRegression().fit(X, y)
reg.score(X, y)

reg.score(X, y)
reg.coef_



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
XY = X.values.transpose().dot(y)

print( np.round( XX_inv.dot(XY), 2) )

reg = LinearRegression().fit(X, y)
np.round( reg.score(X, y), 2)

np.round( reg.coef_, 2 )



''' Ridge Regression with 4 variables'''

x1 = np.random.normal(0 , 0.1, 50)
x2 = np.random.normal(0 , 0.2, 50)

beta_0 = 0
beta_1 = 0.1
beta_2 = 0.5

y = beta_0 + beta_1*x1 + beta_2*x2 + np.random.normal(0 , 0.1, 50)

alpha_0 = 0
alpha_1 = 1
alpha_2 = 0.5
x3 = alpha_0 + alpha_1*x2 + alpha_2*x1 + np.random.normal(0 , 0.1, 50)


gamma_0 = 0
gamma_1 = 0.3
gamma_2 = 0.6
x4 = gamma_0 + gamma_1*x2 + gamma_2*x1 + np.random.normal(0 , 0.01, 50)




X = pd.concat( [pd.Series(x1), pd.Series(x2)], axis = 1 )
reg = LinearRegression().fit(X, y)
reg.score(X, y)
np.corrcoef( [x1, x2, x3, x4])
print( np.corrcoef( [x1, x2, x3, x4]) )


X = pd.concat( [pd.Series(x1), pd.Series(x2), pd.Series(x3), pd.Series(x4)], axis = 1 )
# X = pd.concat( [pd.Series(x1), pd.Series(x2)], axis = 1 )
reg = LinearRegression().fit(X, y)
reg.score(X, y)
print( reg.coef_ )
np.corrcoef( [x1, x2, x3, x4])



XX = X.values.transpose().dot(X.values)

max(np.linalg.svd(XX)[1])/min(np.linalg.svd(XX)[1])

XX_inv = np.linalg.inv( X.values.transpose().dot(X.values) )
XY = X.values.transpose().dot(y)

XY = vector


XX_inv.dot(XY)

reg = LinearRegression().fit(X, y)
reg.score(X, y)

reg.score(X, y)

reg.coef_

reg.intercept_


XtX = XtX + np.diag(np.repeat(10, 4))

max(np.linalg.svd(XtX)[1])/min(np.linalg.svd(XtX)[1])






''' Ridge Regression with 2 variables'''

x1 = np.random.normal(0 , 0.1, 50)
x2 = np.random.normal(0 , 0.2, 50)

beta_0 = 0
beta_1 = 0.1
beta_2 = 0.5

y = beta_0 + beta_1*x1 + beta_2*x2 + np.random.normal(0 , 0.1, 50)

alpha_0 = 0
alpha_1 = 1
alpha_2 = 0.5
x3 = alpha_0 + alpha_1*x2 + alpha_2*x1 + np.random.normal(0 , 0.1, 50)


gamma_0 = 0
gamma_1 = 0.3
gamma_2 = 0.6
x4 = gamma_0 + gamma_1*x2 + gamma_2*x1 + np.random.normal(0 , 0.01, 50)




X = pd.concat( [pd.Series(x1), pd.Series(x2)], axis = 1 )
reg = LinearRegression().fit(X, y)
reg.score(X, y)
np.corrcoef( [x1, x2, x3, x4])
print( np.corrcoef( [x1, x2, x3, x4]) )


X = pd.concat( [pd.Series(x1), pd.Series(x2), pd.Series(x3), pd.Series(x4)], axis = 1 )
# X = pd.concat( [pd.Series(x1), pd.Series(x2)], axis = 1 )
reg = LinearRegression().fit(X, y)
reg.score(X, y)
print( reg.coef_ )
np.corrcoef( [x1, x2, x3, x4])



XX = X.values.transpose().dot(X.values)

max(np.linalg.svd(XX)[1])/min(np.linalg.svd(XX)[1])

XX_inv = np.linalg.inv( X.values.transpose().dot(X.values) )
XY = X.values.transpose().dot(y)



XX_inv.dot(XY)

reg = LinearRegression().fit(X, y)
reg.score(X, y)

reg.score(X, y)

reg.coef_

reg.intercept_


XtX = XtX + np.diag(np.repeat(10, 4))

max(np.linalg.svd(XtX)[1])/min(np.linalg.svd(XtX)[1])









matrix = [XX[0:].tolist()[0], XX[1:].tolist()[0] ]
vector = vector
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













