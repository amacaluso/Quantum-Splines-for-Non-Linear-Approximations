### Experiments ###
from Utils_Spline import *

lower = -1.
upper = 1.
step = .5


## Definition of the interval of B-Spline

label = 'sigmoid'

## Sigmoid
def sigmoid(x, c = 0):
  return c + 1 / (1 + math.exp(-4*x))


x = np.arange(lower, upper + .03, step).tolist()
y = [ sigmoid(value) for value in x]

data_coef = coeff_splines_estimation(x, y, label)
data_est = estimate_function(data_coef, sigmoid, label, c = 0, step=step)

plot_activation(label, data_est, data_coef, full = True)
plot_activation(label, data_est, data_coef, full = False)

