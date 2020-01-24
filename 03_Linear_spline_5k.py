from Utils import *

x = np.arange(-5,5, 0.5)
y = np.tanh(x)
cs = CubicSpline(x, y)
xs = np.arange(-6, 6, 0.1)

fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(x, y, 'o', label='data')
ax.plot(xs, cs(xs), label="S")
ax.plot(xs, cs(xs, 1), label="S'")
ax.plot(xs, cs(xs, 2), label="S''")
ax.plot(xs, cs(xs, 3), label="S'''")
#ax.set_xlim(-0.5, 9.5)
ax.legend(loc='lower right', ncol=2)
plt.show()




'''Linear System'''
# one polynoamial
# Hyperbolic tangent between in (-2,2)
eq11 = pd.Series([1,-1])
eq12 = pd.Series([1,-0.75])
eq21 = pd.Series([1,-1])
eq22  = pd.Series([1,0])
eq31  = pd.Series([1,0])
eq32  = pd.Series([1,1])
eq41 = pd.Series([1,1])
eq42 = pd.Series([1,2])

y1 = [0.047, 0.26]
y2 = [0.26, 0.5]
y3 = [.5, .73]
y4 = [.73, .88]

M1 = pd.concat([eq11, eq12], axis = 1).transpose()
M2 = pd.concat([eq21, eq22], axis = 1).transpose()
M3 = pd.concat([eq31, eq32], axis = 1).transpose()
M4 = pd.concat([eq41, eq42], axis = 1).transpose()

M = [M1, M2, M3, M4]
Y = [y1, y2, y3, y4]

q_beta = []
c_beta = []
fid = []

for i in  range(len(M)):
    m = M[i]
    y = Y[i]
    matrix = m.to_numpy().tolist()
    vector = y

    params['input'] = {
        'name': 'LinearSystemInput',
        'matrix': matrix,
        'vector': vector}

    result = run_algorithm(params)
    #print("solution ", result['solution'])
    q = np.round(result['solution'].real, 5)

    result_ref = ExactLSsolver(matrix, vector).run()
    #print("classical solution ", np.round(result_ref['solution'], 5))
    c = np.round(result_ref['solution'], 5)

    # print("probability %f" % result['probability_result'])
    f = fidelity(q, c)

    fid.append(f)
    q_beta.append(q)
    c_beta.append(c)
    print(i)





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
plt.savefig('results/linear_spline.png')
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
