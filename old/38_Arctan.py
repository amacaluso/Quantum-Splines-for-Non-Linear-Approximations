from Utils import *


## function
def function(x):
  return np.arctan(x)
function(-100000)
label_function = 'arctan'

lower = -1
upper = 1
step = .1

'''Curve'''
data = pd.read_csv('results/arctan_data.csv')

y = (np.array( data.y )+math.pi/2) *1/2
#y = (y - np.min(y))/(np.max(y)-np.min(y))

x = data.x
y_qspline = data.quantum_beta
y_cspline = data.classical_beta *1/2
# y_cspline = (y_cspline - np.min(y_cspline))/(np.max(y_cspline)-np.min(y_cspline))

'''Fidelity'''

data_fid = pd.read_csv('results/arctan_fidelity.csv')
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



x = np.arange(-10,10, 0.25)
y = [(function(i)+math.pi/2)*1/math.pi for i in x]


cs = CubicSpline(x, y)
xs = np.arange(-2, 2, 0.1)

fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(x, y, 'o', label='data')
ax.plot(xs, cs(xs), label="S")
# ax.plot(xs, cs(xs, 1), label="S'")
#ax.plot(xs, cs(xs, 2), label="S''")
#ax.plot(xs, cs(xs, 3), label="S'''")
#ax.set_xlim(-0.5, 9.5)
ax.legend(loc='lower right')#), ncol=2)
plt.grid()
plt.show()

