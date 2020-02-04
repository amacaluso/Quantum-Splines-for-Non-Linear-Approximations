from Utils import *

## function
def function(x):
  return max(0.0, x)

label_function = 'Relu'

lower = -1
upper = 1
step = .1

'''Curve'''
data = pd.read_csv('results/'+label_function +'_data.csv')

y = data.y
#y = (y - np.min(y))/(np.max(y)-np.min(y))

x = data.x
y_qspline = data.quantum_beta
y_cspline = data.classical_beta
# y_cspline = (y_cspline - np.min(y_cspline))/(np.max(y_cspline)-np.min(y_cspline))

'''Fidelity'''

data_fid = pd.read_csv('results/'+ label_function + '_fidelity.csv')
x_fid = data_fid.x
fid = data_fid.Fidelity

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(x, y, label=label_function, color = 'lightblue', linewidth=2)
ax.plot(x, y_qspline, color='forestgreen',  label = 'Quantum LS', dashes=(5, 5),  linestyle='dashed',linewidth=.9)
ax.plot(x, y_cspline, color='firebrick', label = 'Classical LS', dashes=(20, 20),  linestyle='dashed') #, linewidth=.9)
ax.scatter(x_fid, fid, color = 'limegreen', label = 'Fidelity', s = 10)
ax.set_xlim(-1.1, 1.1)
#ax.set_ylim(-.1,1.1)
ax.grid(alpha = 0.3)
ax.set_xlabel(r'x')
ax.set_ylabel(r'$f(x)$')
plt.legend()
plt.savefig('results/' + label_function + '_linear_spline.png', dpi =1000)
plt.show()
plt.close()


