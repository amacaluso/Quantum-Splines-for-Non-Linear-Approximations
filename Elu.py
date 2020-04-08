### Experiments ###
from Utils_Spline import *

lower = -1.
upper = 1.
step = .1


## Definition of the interval of B-Spline
c = .3
label = 'elu'
def elu(z, c = 0, alpha = .3):
	return c + z if z >= 0 else c + alpha*(e**z -1)

x = np.arange(lower, upper + .03, step).tolist()
y = [ elu(value, c) for value in x]

# range = (np.max(y) - np.min(y))
# y_norm = (y-np.min(y))/range
# data_coef = coeff_splines_estimation(x, y_norm, label)

data_coef = coeff_splines_estimation(x, y, label)
data_est = estimate_function(data_coef, elu, label, c = 0, step=step)

# data_est.full_quantum = data_est.full_quantum * range + np.min(y)
# data_est.hybrid_quantum = data_est.hybrid_quantum * range + np.min(y)
# data_est.classical_spline = data_est.classical_spline * range + np.min(y)
# data_est.to_csv('results/elu_estimates.csv', index = False)

data_est.hybrid_quantum = data_est.hybrid_quantum - c
data_est.classical_spline = data_est.classical_spline - c
data_est.to_csv('results/' + label + '_estimates.csv', index=False)


plot_activation(label, data_est, data_coef, full = True)
plot_activation(label, data_est, data_coef, full = False)


