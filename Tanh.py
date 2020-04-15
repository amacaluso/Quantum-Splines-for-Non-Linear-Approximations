### Experiments ###
from Utils_Spline import *

lower = -1.
upper = 1.
step = .1


## Definition of the interval of B-Spline

label = 'tanh'

## Tanh
def tanh(x, c = 1):
  """
   Compute the value of the hyperbolic tangent function with parameter c, for a given point x.

   :param x: (float) input coordinate
   :param c: (float) shifting parameter
   :return: (float) the value of the hyperbolic tangent function
   """
  return (c + np.tanh(x))*c/2


x = np.arange(lower, upper + .03, step).tolist()
y = [tanh(value) for value in x]

data_coef = coeff_splines_estimation(x, y, label)
data_est = estimate_function(data_coef, tanh, label, c = 1, step=step)

plot_activation(label, data_est, data_coef, full = True)
plot_activation(label, data_est, data_coef, full = False)

