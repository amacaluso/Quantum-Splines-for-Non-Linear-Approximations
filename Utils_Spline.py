from Utils import *

def coeff_splines_estimation(x, y, label, saving = True):
    # lower = -1
    # upper = 1
    # step = .5
    # function = relu
    # label = 'relu'
    # c = 1

    ## Linear system for B-Spline - Problem definition
    M = []
    Y = []
    for i in range(1, len(x)):
        eq1 = pd.Series([ 1, x[i-1]])
        eq2 = pd.Series([1, x[i]])
        M_c = pd.concat([eq1, eq2], axis=1).transpose()
        Y.append([y[i-1], y[i]])
        M.append(M_c)


    ## Solving B-Spline using diagonal block matrix
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

    df = pd.DataFrame(columns = ['lower', 'upper', 'q_beta0', 'q_beta1', 'c_beta0', 'c_beta1'])

    for m, q, c in zip(M, q_beta, c_beta):
        row = [m[1][0], m[1][1], q[0], q[1], c[0], c[1]]
        row = pd.Series(row, index=df.columns)
        df = df.append(row, ignore_index=True)
    df['fidelity'] = fid
    if saving:
        df.to_csv('results/' + label + '_full.csv', index=False)
    return df

def estimate_function(data, function, label, c = 0, step = 0.05):
    # data = relu_coef
    # function = relu
    # label = 'relu'
    # step = .5
    # c=0

    interval = data.lower.tolist() + data.upper.tolist()
    interval = list(dict.fromkeys(interval))

    # Sampling points within intervals
    X = []
    for i in range(1, len(interval)):
        X.append(np.arange(interval[i-1], interval[i], step-0.1).tolist())

    # Function estimation - quantum and classical
    q_beta = [[b0, b1] for b0, b1 in zip(data.q_beta0, data.q_beta1)]
    c_beta = [[b0, b1] for b0, b1 in zip(data.c_beta0, data.c_beta1)]

    full_qy = []
    hybrid_qy =[]
    cy = []

    for i in range(len(X)):
        for x in X[i]:
            point = [1, x]
            coeffs = q_beta[i]
            full_qy.append(dot_product(point, coeffs))
            cy.append(c_beta[i][0] + x * c_beta[i][1])
            hybrid_qy.append(q_beta[i][0] + x * q_beta[i][1])

    full_qy = [0 if math.isnan(x) else x for x in full_qy]

    x = [item for sublist in X for item in sublist]
    y = [function(value, c) for value in x]


    data_est = pd.DataFrame()
    data_est['x'] = x
    data_est['y'] = y
    data_est['full_quantum'] = full_qy
    data_est['hybrid_quantum'] = hybrid_qy
    data_est['classical_spline'] = cy

    data_est.to_csv('results/' + label + '_estimates.csv', index=False)

    return data_est




def plot_activation(label, data, data_coef, full = True):

    # data = relu_est
    # data_coef = relu_coef

    x = data.x
    y = data.y
    cy = data.classical_spline

    if full:
        qy = data.full_quantum
        type = 'Full'
    else:
        qy = data.hybrid_quantum
        type = 'Hybrid'

    x_fid = (data_coef.lower + data_coef.upper)/2
    fid = data_coef.fidelity

    fig, ax = plt.subplots(figsize=(6, 5))
    # Full Qspline
    ax.plot(x, cy, color='orange', label='Classic spline',
            zorder=1)  # , dashes=(5, 7),  linestyle='dashed',linewidth=1.3)
    ax.plot(x, qy, color='steelblue',
            label=type + ' Qspline')  # , dashes=(5, 7),  linestyle='dashed',linewidth=1.3)
    ax.plot(x, y, label='Activation', color='sienna', linestyle='dotted', dashes=(1, 1.5), zorder=2,
            linewidth=3)
    ax.scatter(x_fid, fid, color='cornflowerblue', label='Fidelity', s=10)
    ax.set_xlim(-1.1, 1.1)
    ax.grid(alpha=0.3)
    ax.set_xticks(np.round(np.arange(-1, 1.1, .4), 1).tolist())
    ax.text(0.65, 0.1, label,
            transform=ax.transAxes, ha="left")
    plt.legend()
    plt.savefig('results/' + label + '_'+ type +'.png', dpi =300)
    plt.show()
    plt.close()




def single_plot( i, x, y, qy, cy, x_fid, fid, label, coord = [0.68, .1]):
    ax = plt.subplot(int(str(22)+ str(i)))
    ax.plot(x, y, color='orange', label = 'Classic spline', zorder=1)
    ax.plot(x, qy, color='steelblue',  label = 'QSpline')
    ax.plot(x, cy, label='Activation', color = 'sienna', linestyle='dotted', dashes=(1,1.5), zorder=2, linewidth=3)
    ax.scatter(x_fid, fid, color = 'cornflowerblue', label = 'Fidelity', s = 10)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-.2, 1.05)
    ax.grid(alpha = 0.3)
    ax.set_xticks(np.round(np.arange(-1, 1.1, .4),1).tolist())
    ax.set_yticks(np.round(np.arange(-.2, 1.05, .2),1).tolist())
    ax.text(coord[0], coord[1], label, transform=ax.transAxes, ha="left")



def load_data(label='sigmoid', approach='Hybrid'):
    data_fid = pd.read_csv('results/' + label + '_full.csv')
    data = pd.read_csv('results/' + label + '_estimates.csv')

    x = data.x
    y = data.y
    cy = data.classical_spline
    if approach == 'Hybrid':
        qy = data.hybrid_quantum
    else:
        qy = data.full_quantum
    x_fid = (data_fid.lower + data_fid.upper) / 2
    fid = data_fid.fidelity
    return x, y, qy, cy, x_fid, fid