### Experiments ###
from Utils_Spline import *

lower = -1.
upper = 1.
step = .1


## Definition of the interval of B-Spline
c=.5
label = 'relu'
def relu(x, c = 0):
    return c + max(0.0, x)

x = np.arange(lower, upper + .03, step).tolist()
y = [ relu(value, c) for value in x]

data_coef = coeff_splines_estimation(x, y, label) # data_coef = pd.read_csv('results/relu_full.csv')
data_est = estimate_function(data_coef, relu, label, c=0, step=step)

data_est.hybrid_quantum = data_est.hybrid_quantum - c
data_est.classical_spline = data_est.classical_spline - c
data_est.to_csv('results/' + label + '_estimates.csv', index=False)

plot_activation(label, data_est, data_coef, full = True)
plot_activation(label, data_est, data_coef, full = False)

