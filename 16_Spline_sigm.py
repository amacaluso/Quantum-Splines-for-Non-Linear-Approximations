from Utils import *

x = np.arange(-5, 5, 0.5)
y = [sigmoid(i) for i in x]
cs = CubicSpline(x, y)
xs = np.arange(-5, 5, 0.1)

fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(x, y, 'o', label='data')
ax.plot(xs, cs(xs), label="S")
ax.plot(xs, cs(xs, 1), label="S'")
ax.plot(xs, cs(xs, 2), label="S''")
ax.plot(xs, cs(xs, 3), label="S'''")
ax.set_xlim(-3, 3)
ax.set_ylim(-.1, 1.1)
ax.legend(loc='lower right', ncol=2)
ax.grid()
plt.show()


X = [-2, -1/2, 1/2, 2]

M = pd.concat([poly_data(x) for x in X], axis = 1).transpose()
y = [sigmoid(x) for x in X]

'''Linear System with classical inversion'''
# sigmoid between in (-2,2)
M_inv = np.linalg.inv(M)
print(M_inv)
beta_classical = M_inv.dot(y)
print(beta_classical)

# Evaluation single observation
x_new = 2
y_new = poly_data(x_new).to_numpy().dot(beta_classical)
print(y_new)


'''Quantum Linear inversion'''

matrix = M.to_numpy().tolist()
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
