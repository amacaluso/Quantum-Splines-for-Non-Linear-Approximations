from Utils import *

n_obs = 1000

''' Ridge Regression'''
# Variables
x1 = np.random.normal(0 , 0.1, n_obs)
x2 = np.random.normal(0 , 0.2, n_obs)

# beta coefficients
beta_0 = 0
beta_1 = 1
beta_2 = 2

mu_eps = 0
std_eps = 0.1

std_x = 0.1





# Model definition
y = beta_0 + beta_1*x1 + beta_2*x2 + np.random.normal(mu_eps , std_eps, n_obs)


'''Regression using scikit-learn'''
# Create design matrix
X = pd.concat( [pd.Series(x1), pd.Series(x2)], axis = 1 )
reg = LinearRegression().fit(X, y)
reg.score(X, y)
reg = LinearRegression().fit(X, y)
reg.score(X, y)
reg.score(X, y)
reg.coef_


'''Regression from scratch'''
# Compute the matrix to invert
XX = X.values.transpose().dot(X.values)

# Condition number
k = max(np.linalg.svd(XX)[1])/min(np.linalg.svd(XX)[1])
print('k =',np.round( k, 2))
XX_inv = np.linalg.inv( X.values.transpose().dot(X.values) )
XY = X.values.transpose().dot(y)
XX_inv.dot(XY)

'''Ridge Regression with 2 variables - ill conditioned matrix'''
x1 = np.random.normal(mu_eps , std_eps, n_obs)
x2 = x1*2 + np.random.normal(mu_eps , std_x, n_obs)

print( 'Pearson coefficients:', np.round( np.corrcoef( [x1, x2] )[0][1], 3) )

y = beta_0 + beta_1*x1 + beta_2*x2 + np.random.normal(mu_eps , std_eps, n_obs)
X = pd.concat( [pd.Series(x1), pd.Series(x2)], axis = 1 )


'''Regression using scikit-learn - ill conditioned matrix'''
reg = LinearRegression().fit(X, y)
reg.score(X, y)
np.round( reg.score(X, y), 2)
np.round( reg.coef_, 2 )

'''Regressionfrom scratch - ill conditioned matrix'''
XX = X.values.transpose().dot(X.values)
np.round( max(np.linalg.svd(XX)[1])/min(np.linalg.svd(XX)[1]), 2 )
XX_inv = np.linalg.inv( X.values.transpose().dot(X.values) )
XY = X.values.transpose().dot(y)
print( np.round( XX_inv.dot(XY), 2) )


print('The condition number is: ', np.round( max(np.linalg.svd(XX)[1])/min(np.linalg.svd(XX)[1]), 2) )

matrix  = XX.tolist()
vector = XY.tolist()

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




fidelities = []
cond_nums = []
ro = []

std_x = 1

for i in range(10):
    print( i )
    '''Ridge Regression with 2 variables - ill conditioned matrix'''
    x1 = np.random.uniform(mu_eps, std_eps, n_obs)
    x2 = x1 + np.random.normal(mu_eps, std_x, n_obs)

    corr = np.corrcoef([x1, x2])[0][1]
    ro.append(corr)
    print('Pearson coefficients:', np.round(np.corrcoef([x1, x2])[0][1], 3))

    y = beta_0 + beta_1 * x1 + beta_2 * x2 + np.random.normal(mu_eps, std_eps, n_obs)
    X = pd.concat([pd.Series(x1), pd.Series(x2)], axis=1)

    XX = X.values.transpose().dot(X.values)
    k = max(np.linalg.svd(XX)[1]) / min(np.linalg.svd(XX)[1])
    XX_inv = np.linalg.inv(X.values.transpose().dot(X.values))
    XY = X.values.transpose().dot(y)

    print('The condition number is: ', np.round(k, 2))

    matrix = XX.tolist()
    vector = XY.tolist()

    params['input'] = {
        'name': 'LinearSystemInput',
        'matrix': matrix,
        'vector': vector}

    result = run_algorithm(params)
    print("solution ", np.round(result['solution'], 5))

    result_ref = ExactLSsolver(matrix, vector).run()
    print("classical solution ", np.round(result_ref['solution'], 5))

    print("probability %f" % result['probability_result'])
    f = fidelity(result['solution'], result_ref['solution'])

    fidelities.append(f)
    cond_nums.append(k)



A = fidelities.copy()
B = cond_nums.copy()


A = [a for _,a in sorted(zip(B,A))]

B.sort()

plt.scatter(B, A)
plt.show()
























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













