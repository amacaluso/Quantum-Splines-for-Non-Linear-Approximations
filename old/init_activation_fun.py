'''Hybrid Quantum Splines'''
# --> Elu, Relu, Sigmoid, Tanh, Arctan

from Utils import *


lower = -1
upper = 1
step = .1



label_function = 'elu'

## Elu
def function(z,alpha = .3, c = .3):
	return c + z if z >= 0 else c + alpha*(e**z -1)

execfile('experiments_act_fun.py')

data.quantum_beta = data.quantum_beta - 1
data.classical_beta = data.classical_beta - 1
data['label'] = label_function

data_elu = data.copy()
data_fid_elu = F.copy()






## Tanh
def function(x, c = 1):
  return (c + np.tanh(x))*c/2
label_function = 'tanh'
execfile('04_experiments_act_fun.py')

data.quantum_beta = data.quantum_beta - 1
data.classical_beta = data.classical_beta - 1
data['label'] = label_function

data_tanh = data.copy()
data_fid_tanh = F.copy()



## Relu
def function(x, c = 1):
  return c + max(0.0, x)

label_function = 'relu'

execfile('04_experiments_act_fun.py')

data.quantum_beta = data.quantum_beta - 1
data.classical_beta = data.classical_beta - 1
data['label'] = label_function

data_relu = data.copy()
data_fid_relu = F.copy()




## Sigmoid
def function(x, c = 0):
  return c + 1 / (1 + math.exp(-4*x))
label_function = 'sigmoid'
execfile('04_experiments_act_fun.py')

data['label'] = label_function
data_sig = data.copy()
data_fid_sig = F.copy()


# ## Arctan
# def function(x, c = math.pi):
#   return (c + np.arctan(x))*c
# label_function = 'arctan'
# execfile('04_experiments_act_fun.py')
#
# data.quantum_beta = data.quantum_beta - 1
# data.classical_beta = data.classical_beta - 1
# data['label'] = label_function
#
# data_arct = data.copy()
# data_fid_arct = F.copy()

# d_arct = data_arct.loc[ :, ['y', 'quantum_beta', 'classical_beta']]
# d_arct.columns = ['y_arct', 'arct_quantum', 'arct_classical']


'''Collect results'''


d_tanh = data_tanh.loc[ :, ['y', 'quantum_beta', 'classical_beta']]
d_tanh.columns = ['y_tanh', 'tanh_quantum', 'tanh_classical']

d_sig = data_sig.loc[ :, ['y', 'quantum_beta', 'classical_beta']]
d_sig.columns = ['y_sig',  'sig_quantum', 'sig_classical']

d_relu = data_relu.loc[ :, ['y', 'quantum_beta', 'classical_beta']]
d_relu.columns = ['y_relu', 'relu_quantum', 'relu_classical']

d_elu = data_elu.loc[ :, ['y', 'quantum_beta', 'classical_beta']]
d_elu.columns = ['y_elu','elu_quantum', 'elu_classical']

x = data_tanh['x']
y = data_tanh['y']

data_full = pd.concat([x, d_sig, d_relu, d_elu, d_tanh], axis = 1)
fid_full = pd.concat([data_fid_sig['x'],
                      data_fid_sig['Fidelity'], data_fid_relu['Fidelity'],
                      data_fid_elu['Fidelity'], data_fid_tanh['Fidelity']],
                      axis = 1)
fid_full.columns = ['x', 'sig', 'relu', 'elu', 'tanh']
data_full.to_csv( 'data_full.csv', index = False)
fid_full.to_csv( 'fid_full.csv', index = False)
