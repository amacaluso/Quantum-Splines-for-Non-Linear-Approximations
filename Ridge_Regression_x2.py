from Utils import *

n_obs = 1000
np.random.seed(1234)

''' Ridge Regression'''
# Variables
x1 = np.random.normal(5 , 2, n_obs)
x2 = np.random.normal(0 , 1, n_obs)

# beta coefficients
beta_0 = 0
beta_1 = 1
beta_2 = 2

mu_eps = 0
std_eps = 1

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
x2 = x1 + np.random.normal(mu_eps , std_x, n_obs)

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





lambda_vector = np.arange(.25, 30, .75)
# lambda_vector = np.arange(1, 200, 2.5)
len(lambda_vector)
fidelities_LR, fidelities_RR  = [], []
cond_num_LR, cond_num_RR = [], []
p_LR, p_RR = [], []
ro = []

std_x_vec = np.arange(.1, 5, .125)
len(std_x_vec)


for i in range(len(std_x_vec)):
    # i = 60
    '''Linear Regression'''
    x1 = np.random.uniform(mu_eps, std_eps, n_obs)
    x2 = 3*x1 + np.random.normal(mu_eps, .1, n_obs)

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

    # matrix = (XX + np.diag(np.repeat(lambda_penalty, 2, axis=0))).tolist()

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

    fidelities_LR.append(f)
    cond_num_LR.append(k)
    p_LR.append(result['probability_result'])

    '''Ridge Regression with 2 variables - ill conditioned matrix'''
    matrix = (XX + np.diag(np.repeat(lambda_vector[i], 2, axis=0))).tolist()
    print(np.round(matrix))

    k_ridge = max(np.linalg.svd(matrix)[1]) / min(np.linalg.svd(matrix)[1])
    print('The (ridge) condition number is: ', np.round(k_ridge, 2))

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

    fidelities_RR.append(f)
    cond_num_RR.append(k_ridge)
    p_RR.append(result['probability_result'])





fid_LR = fidelities_LR.copy() #; fid_LR = fid_LR[:-1]
fid_RR = fidelities_RR.copy()

k_LR = cond_num_LR.copy() #; k_LR = k_LR[:-1]
k_RR = cond_num_RR.copy()

prob_LR = p_LR.copy() #; prob_LR = prob_LR[:-1]
prob_RR = p_RR.copy()

corr = ro.copy() #; corr = corr[:-1]
l_values = lambda_vector.copy() #; l_values = l_values[0:58]


fid_LR = [f for _,f in sorted(zip(lambda_vector,fid_LR))]
fid_RR = [f for _,f in sorted(zip(lambda_vector,fid_RR))]

k_LR = [k for _,k in sorted(zip(lambda_vector, k_LR))]
k_RR = [k for _,k in sorted(zip(lambda_vector,k_RR))]

prob_LR = [p for _,p in sorted(zip(lambda_vector,prob_LR))]
prob_RR = [p for _,p in sorted(zip(lambda_vector,prob_RR))]

#corr = [r for _,r in sorted(zip(lambda_vector,corr))]

l_values.sort()

plt.plot(l_values, fid_LR, color = 'red', label = 'Fidelity (LR)')
plt.plot(l_values, fid_RR, color = 'lightsalmon', label = 'Fidelity (RR)')

plt.plot(lambda_vector, prob_LR, color = 'g')
plt.plot(lambda_vector, prob_RR, color = 'lime')
#plt.plot(l_values, corr, color = 'g', label = 'Correlation')
plt.legend()

plt2=plt.twinx()

plt2.plot(l_values, k_LR, color = 'blue', label = r'$\kappa$ (LR)')
plt2.plot(l_values, k_RR, color = 'skyblue', label = r'$\kappa$ (RR)')
plt.legend(loc = 'lower right')
plt.show()

plt.close()

np.corrcoef(fid_RR, k_RR)






data = pd.DataFrame()
data['lambda'] = l_values
data['fidelity_LR'] = fid_LR
data['fidelity_RR'] = fid_RR
data['k_LR'] = k_LR
data['k_RR'] = k_RR
data['prob_LR'] = prob_LR
data['prob_RR'] = prob_RR
data['correlation'] = corr

rand_n = np.random.randint(10**3)
data.to_csv(str(rand_n) + '_results.csv', index = False)






fid_LR2 = [p for _,p in sorted(zip(k_LR, fid_LR))]
k_LR2 = k_LR.sort()

plt2.plot(k_LR2, fid_LR2, color = 'skyblue', label = r'$\kappa$ (RR)')
plt.legend(loc = 'lower right')
plt.show()





#
#
# params2 = params
# params2['reciprocal'] = {
#     'scale': 0.5
# }
#
# result = run_algorithm(params2)
# print("solution ", np.round(result['solution'], 5))
#
# result_ref = ExactLSsolver(matrix, vector).run()
# print("classical solution ", np.round(result_ref['solution'], 5))
#
# print("probability %f" % result['probability_result'])
# fidelity(result['solution'], result_ref['solution'])
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# '''Ridge Regression with 2 variables - ill conditioned matrix'''
#
# x1 = np.random.normal(0 , 0.1, 50)
# x2 = x1*2 + np.random.normal(0 , 0.1, 50)
#
# print( 'Coefficiente di correlazione:', np.round( np.corrcoef( [x1, x2] )[0][1], 3) )
#
# y = beta_0 + beta_1*x1 + beta_2*x2 + np.random.normal(0 , 0.1, 50)
#
# X = pd.concat( [pd.Series(x1), pd.Series(x2)], axis = 1 )
# reg = LinearRegression().fit(X, y)
# reg.score(X, y)
#
# XX = X.values.transpose().dot(X.values)
#
# np.round( max(np.linalg.svd(XX)[1])/min(np.linalg.svd(XX)[1]), 2 )
#
# XX_inv = np.linalg.inv( X.values.transpose().dot(X.values) )
# XY = np.round( X.values.transpose().dot(y))
#
# print( np.round( XX_inv.dot(XY), 2) )
#
# reg = LinearRegression().fit(X, y)
# np.round( reg.score(X, y), 2)
#
# np.round( reg.coef_, 2 )
#
#
#
#
#
#
#
#
# ''' Ridge Regression with 2 variables'''
#
# x1 = np.random.normal(0 , 0.1, 50); x2 = np.random.normal(0 , 0.2, 50)
#
# beta_0 = 0; beta_1 = 1; beta_2 = 2
#
# y = beta_0 + beta_1*x1 + beta_2*x2 + np.random.normal(0 , 0.1, 50)
# X = pd.concat( [pd.Series(x1), pd.Series(x2)], axis = 1 )
# reg = LinearRegression().fit(X, y)
# reg.score(X, y)
#
# XX = X.values.transpose().dot(X.values)
# print('The condition number is: ', np.round( max(np.linalg.svd(XX)[1])/min(np.linalg.svd(XX)[1]), 2) )
#
#
# XX_inv = np.linalg.inv( X.values.transpose().dot(X.values) )
# XY = [1, 4 ] #X.values.transpose().dot(y)
#
# XX_inv.dot(XY)
#
# print('The regression coefficients are:', np.round( reg.coef_, 2))
#
# matrix =  [[20, 0], [0, 11]]
# vector = [1, 2]
#
# np.linalg.inv( matrix ).dot( vector )
#
#
#
#
#








