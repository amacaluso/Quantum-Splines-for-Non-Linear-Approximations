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
c = .3
label = 'elu'
def elu(z, c = 0, alpha = .3):
	"""
	 Compute the value of the elu function with parameter c, for a given point x.

	 :param x: (float) input coordinate
	 :param c: (float) shifting parameter
	 :return: (float) the value of the elu function
	 """
	return c + z if z >= 0 else c + alpha*(e**z -1)

x = np.arange(lower, upper + .03, step).tolist()
y = [ elu(value, c) for value in x]


data_coef = coeff_splines_estimation(x, y, label)
data_est = estimate_function(data_coef, elu, label, c = 0, step=step)

data_est.hybrid_quantum = data_est.hybrid_quantum-c
data_est.classical_spline = data_est.classical_spline -c
data_est.to_csv('results/' + label + '_estimates.csv', index=False)


plot_activation(label, data_est, data_coef, full = True)
plot_activation(label, data_est, data_coef, full = False)


