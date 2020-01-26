from Utils import *

'''Classical cubic spline'''
x = np.arange(-2,2, 0.25)
y = [sigmoid(i) for i in x]
cs = CubicSpline(x, y)
xs = np.arange(-2, 2, 0.1)

fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(x, y, 'o', label='data')
ax.plot(xs, cs(xs), label="S")
# ax.plot(xs, cs(xs, 1), label="S'")
#ax.plot(xs, cs(xs, 2), label="S''")
#ax.plot(xs, cs(xs, 3), label="S'''")
#ax.set_xlim(-0.5, 9.5)
ax.legend(loc='lower right')#), ncol=2)
plt.grid()
plt.show()


#### ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ ####
#### ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ ####
#### ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ ####
#### ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ ####

'''Interpolation using five quadratic functions and seven knots'''

# Generate dataset for test observation
x1 = np.arange(-2, -1, 0.0125)
x2 = np.arange(-1, 0, 0.0125)
x3 = np.arange( 0, 1, 0.0125)
x4 = np.arange( 1, 2, 0.0125)
X= [x1, x2, x3, x4]

qy =[]
cy = []

for i in range(len(X)):
    for x in X[i]:
        qy.append(q_beta[i][0]+x*q_beta[i][1])
        cy.append(c_beta[i][0] + x * c_beta[i][1])

x_new = np.arange(-2, 2, .0125)
y = [sigmoid(j) for j in x_new]
cs = CubicSpline(x_new, y)
xs = np.arange(-3, 3, 0.1)

fig, ax = plt.subplots(figsize=(6.5, 4))
ax.scatter(x_new, y, label='data')
ax.plot(xs, cs(xs), label = 'Cubic Spline')
ax.scatter(x_new, qy, color='red', marker='1', label = 'quantum linear spline')
ax.scatter(x_new, cy, color='tomato', marker='+', label = 'classical linear spline')
plt.grid()
plt.legend()
# plt.savefig('results/linear_spline.png')
plt.show()



'''Linear System'''
# one polynoamial
# Hyperbolic tangent between in (-2,2)
eq1  = pd.Series([1,-2, 0, 0, 0, 0, 0, 0])
eq2  = pd.Series([1,-1, 0, 0, 0, 0, 0, 0])
eq3  = pd.Series([0, 0, 1,-1, 0, 0, 0, 0])
eq4  = pd.Series([0, 0, 1, 0, 0, 0, 0, 0])
eq5  = pd.Series([0, 0, 0, 0, 1, 0, 0, 0])
eq6  = pd.Series([0, 0, 0, 0, 1, 1, 0, 0])
eq7  = pd.Series([0, 0, 0, 0, 0, 0, 1, 1])
eq7  = pd.Series([0, 0, 0, 0, 0, 0, 1, 1])
eq8  = pd.Series([0, 0, 0, 0, 0, 0, 1, 2])


M = pd.concat([eq1, eq2, eq3, eq4, eq5, eq6, eq7, eq8], axis = 1).transpose()

M_inv = np.linalg.inv(M)
print(M_inv)
print(y)

beta_classical = M_inv.dot(y)
print(beta_classical)
# Evaluation single observation
x_new = 1/2
y_new = beta_classical[4] + beta_classical[5]*x_new
print(y_new)

matrix = M.to_numpy().tolist()

# matrix  = M.to_numpy().tolist()
vector = y

params['input'] = {
    'name': 'LinearSystemInput',
    'matrix': matrix,
    'vector': vector }



result = run_algorithm(params)
print("solution ", result['solution'])

result_ref = ExactLSsolver(matrix, vector).run()
print("classical solution ", np.round(result_ref['solution'], 5))

print("probability %f" % result['probability_result'])
fidelity(result['solution'], result_ref['solution'])

beta_quantum = np.round(result['solution'], 5)
print(beta_quantum)


# Generate dataset for test observation
x_points = np.random.uniform(-2,2,100)
X = []

for x in x_points:
    x_poly = pd.Series([1, x, x**2, x**3])
    X.append(x_poly)
X = pd.DataFrame(X)
Y_new_c = X.dot(beta_classical)
Y_new_q = X.dot(beta_quantum.real)

plt.scatter(x_points, Y_new_c, color='darkcyan', marker='+')
plt.scatter(x_points, Y_new_q)
x_tanh = np.arange(-2,2, 0.5)
y_tanh = np.tanh(x_tanh)
plot(x_tanh, y_tanh, color = 'red')
plt.show()


np.arange(-2,2,4)
#%%

#%%
x_new = 1/2
y_new = beta[0] + beta[1]*x_new +  beta[2]*x_new**2 + beta[3]*x_new**3

print(y_new)



#### ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ ####
#### ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ ####
#### ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ ####


'''Interpolation using two quadratic functions and three knots'''
# sigmoid with c = 4 between in (-1, -.5)

eq11 = pd.Series([1,-.5, .25])
eq12 = pd.Series([1,-25, .0625])
eq13 = pd.Series([1, 0, 0])

y = [0.12, 0.27, .5]

M = pd.concat([eq11, eq12, eq13], axis = 1).transpose()
# M = M + np.diag(np.repeat(0.1, 3, axis=0))
q_beta = []
c_beta = []
fid = []

matrix = M.to_numpy().tolist()
vector = y
params['input'] = {
    'name': 'LinearSystemInput',
    'matrix': matrix,
    'vector': vector}

result = run_algorithm(params)
q = np.round(result['solution'].real, 5)
result_ref = ExactLSsolver(matrix, vector).run()
c = np.round(result_ref['solution'], 5)
f = fidelity(q, c)


x_new = np.arange(-1, -.5, .0125)
qy =[]
cy = []

for x in x_new:
    qy.append(q[0]+x*q[1] + q[2]*x**2)
    cy.append(c[0]+x*c[1] + c[2]*x**2)

asd = np.arange(-1, 1, .0125)
y = [sigmoid(j) for j in asd]
cs = CubicSpline(asd, y)
xs = np.arange(-1, -.5, 0.1)

fig, ax = plt.subplots(figsize=(6.5, 4))
ax.scatter(asd, y, label='data')
ax.plot(xs, cs(xs), label = 'Cubic Spline')
ax.scatter(x_new, qy, color='red', marker='1', label = 'quantum quadratic spline')
ax.scatter(x_new, cy, color='tomato', marker='+', label = 'classical quadratic spline')
plt.grid()
plt.legend()
# plt.savefig('results/linear_spline.png')
plt.show()







