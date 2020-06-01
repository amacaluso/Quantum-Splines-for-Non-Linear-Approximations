# Copyright 2020 Antonio Macaluso
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

