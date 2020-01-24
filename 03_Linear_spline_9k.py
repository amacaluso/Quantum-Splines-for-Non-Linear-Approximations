from Utils import *

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




'''Linear System'''
# one polynoamial
# Hyperbolic tangent between in (-2,2)
eq11 = pd.Series([1, -1])
eq12 = pd.Series([1, -.75])

eq21 = pd.Series([1, -.75])
eq22  = pd.Series([1,-.5])

eq31  = pd.Series([1,-.5])
eq32  = pd.Series([1,-.25])

eq41 = pd.Series([1, -.25])
eq42 = pd.Series([1, 0])

eq51 = pd.Series([1, 0])
eq52 = pd.Series([1, .25])

eq61 = pd.Series([1, .25])
eq62  = pd.Series([1, .5])

eq71  = pd.Series([1, -.5])
eq72  = pd.Series([1, .75])

eq81 = pd.Series([1, .75])
eq82 = pd.Series([1,  1])



y1 = [0.017, 0.047]
y2 = [0.047, .12]
y3 = [.12, .27]
y4 = [.27, .5]

y5 = [.5, .73]
y6 = [.73, 0.88]
y7 = [.88, .95]
y8 = [.95, .98]

M1 = pd.concat([eq11, eq12], axis = 1).transpose()
M2 = pd.concat([eq21, eq22], axis = 1).transpose()
M3 = pd.concat([eq31, eq32], axis = 1).transpose()
M4 = pd.concat([eq41, eq42], axis = 1).transpose()
M5 = pd.concat([eq51, eq52], axis = 1).transpose()
M6 = pd.concat([eq61, eq62], axis = 1).transpose()
M7 = pd.concat([eq71, eq72], axis = 1).transpose()
M8 = pd.concat([eq81, eq82], axis = 1).transpose()

M = [M1, M2, M3, M4, M5, M6, M7, M8]
Y = [y1, y2, y3, y4, y5, y6, y7, y8]

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




p = .1
# Generate dataset for test observation
x1 = np.arange(-1, -.75, p)
x2 = np.arange(-.75, -.5, p)
x3 = np.arange( -.5, -.25, p)
x4 = np.arange( -.25, 0, p)
x5 = np.arange( 0, .25, p)
x6 = np.arange( .25, .5, p)
x7 = np.arange( .5, .75, p)
x8 = np.arange( .75, 1, p)


X= [x1, x2, x3, x4, x5, x6, x7, x8]

qy =[]
cy = []

for i in range(len(X)):
    for x in X[i]:
        qy.append(q_beta[i][0]+x*q_beta[i][1])
        cy.append(c_beta[i][0] + x * c_beta[i][1])

x_new = [item for sublist in X for item in sublist]
y = [sigmoid(j) for j in x_new]
cs = CubicSpline(x_new, y)
# xs = np.arange(-1, 1, p)

fig, ax = plt.subplots(figsize=(6.5, 4))
#ax.scatter(x_new, y, label='data')
ax.plot(xs, cs(xs), label = 'Cubic Spline')
ax.scatter(x_new, qy, color='red',  label = 'quantum linear spline', s = .8) # linestyle='dotted',
ax.scatter(x_new, cy, color='green', label = 'classical linear spline', s = .8) #  linestyle='dashdot',
plt.grid()
ax.set_xlim(-2,2)
plt.legend()
plt.savefig('results/linear_spline.png')
plt.show()


