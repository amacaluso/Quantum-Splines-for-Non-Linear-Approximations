lower = -1
upper = 1
step = .1

'''Linear System'''
interval = np.arange(lower,upper + .03, step)
M = []
Y = []
for i in range(1, len(interval)):
    eq1 = pd.Series([ 1, interval[i-1]])
    eq2 = pd.Series([1, interval[i]])
    M_c = pd.concat([eq1, eq2], axis=1).transpose()
    Y.append([function(interval[i-1]), function(interval[i])])
    M.append(M_c)

q_beta = []
c_beta = []
fid = []

for i in  range(len(M)):
    m = M[i]
    y = Y[i]
    if y == [0.0, 0.0]:
       y = [el +10**-4 for el in y]
    matrix = m.to_numpy().tolist()
    vector = y

    params['input'] = {
        'name': 'LinearSystemInput',
        'matrix': matrix,
        'vector': vector}

    result = run_algorithm(params)
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


X = []
for i in range(1, len(interval)):
    #i =1
    X.append(np.arange(interval[i-1], interval[i], 0.05).tolist())


qy =[]
cy = []

for i in range(len(X)):
    for x in X[i]:
        point = [1, x]
        coeffs = q_beta[i]
        qy.append(dot_product(point, coeffs))
        cy.append(c_beta[i][0] + x * c_beta[i][1])

qy = [0 if math.isnan(x) else x for x in qy]

x = [item for sublist in X for item in sublist]
y = [function(j) for j in x]

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(x, y, label=label_function, color = 'lightblue', linewidth=2)
#ax.plot(xs, cs(xs), label = 'Cubic Spline')
ax.plot(x, qy, color='forestgreen',  label = 'Quantum LS', dashes=(5, 5),  linestyle='dashed',linewidth=.9)
ax.plot(x, cy, color='firebrick', label = 'Classical LS', dashes=(50, 50),  linestyle='dashed') #, linewidth=.9)
x_fid = np.arange(lower + .05, upper, step).tolist()
ax.scatter(x_fid, fid, color = 'limegreen', label = 'Fidelity', s = 10)
ax.set_xlim(-1.1, 1.1)
#ax.set_ylim(-.1,1.1)
ax.grid(alpha = 0.3)
ax.set_xlabel(r'x')
ax.set_ylabel(r'$f(x)$')
plt.legend()
plt.savefig('results/' + label_function + '_full_linear_spline.png', dpi =1000)
plt.show()
plt.close()


data = pd.DataFrame()
data['x'] = x_new
data['y'] = y
data['quantum_beta'] = qy
data['classical_beta'] = cy
data.to_csv('results/' + label_function + '_full_data.csv', index=False)

F = pd.DataFrame(fid).fillna(0)
F['x'] = x_fid
F.columns = ['Fidelity', 'x']
F.to_csv('results/' + label_function + '_full_fidelity.csv', index=False)






# x = np.arange(-1,1, 0.1)
# y = [function(i) for i in x]
# cs = CubicSpline(x, y)
# xs = np.arange(-1, 1, 0.1)
#
# fig, ax = plt.subplots(figsize=(6.5, 4))
# ax.plot(x, y, 'o', label='data')
# ax.plot(xs, cs(xs), label="S")
# # ax.plot(xs, cs(xs, 1), label="S'")
# #ax.plot(xs, cs(xs, 2), label="S''")
# #ax.plot(xs, cs(xs, 3), label="S'''")
# #ax.set_xlim(-0.5, 9.5)
# ax.legend(loc='lower right')#), ncol=2)
# plt.grid()
# plt.show()
#