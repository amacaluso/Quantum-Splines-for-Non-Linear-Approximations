from Utils import *
from sklearn.preprocessing import normalize

## function
## Tanh
def function(x, c = 1):
  return c + np.tanh(x)
label_function = 'tanh'

lower = -1
upper = 1
step = .1

'''Curve'''
data = pd.read_csv('results/'+label_function +'_data.csv')

y = (data.y +1)*1/2
# y = [y[i] for i in np.arange(0, 45, 5)]
#y = (y - np.min(y))/(np.max(y)-np.min(y))

x = data.x
xa = [x[i] for i in np.arange(0, 45, 5)]

y_qspline = data.quantum_beta
y_cspline = (data.classical_beta)*1/2
# y_cspline = (y_cspline - np.min(y_cspline))/(np.max(y_cspline)-np.min(y_cspline))

'''Fidelity'''

data_fid = pd.read_csv('results/'+ label_function + '_fidelity.csv')
x_fid = data_fid.x
fid = data_fid.Fidelity

# hybrid Qspline
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(x, y_cspline, color='orange', label = 'Classic spline', zorder=1) #, dashes=(5, 7),  linestyle='dashed',linewidth=1.3)
ax.plot(x, y_qspline, color='steelblue',  label = 'Qspline') #, dashes=(5, 7),  linestyle='dashed',linewidth=1.3)
ax.plot(x, y, label='Activation', color = 'sienna', linestyle='dotted', dashes=(1,1.5), zorder=2, linewidth=3)
ax.scatter(x_fid, fid, color = 'cornflowerblue', label = 'Fidelity', s = 10)
ax.set_xlim(-1.1, 1.1)
#ax.set_ylim(-.1,1.1)
ax.grid(alpha = 0.3)
# ax.set_xlabel(r'x')
ax.set_ylabel(r'$f(x)$', rotation = 0)
ax.set_xticks(np.round(np.arange(-1, 1.1, .2),1).tolist())
# Creating legend and title for the figure. Legend created with figlegend(), title with suptitle()
# handles, labels = ax.get_legend_handles_labels()
# fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05),
#            ncol = 4, borderaxespad=0)
ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2),
           ncol = 4, borderaxespad=0. )
plt.savefig('results/' + label_function + '_linear_spline.png', dpi =1000)
plt.show()
plt.close()


# full Qspline

markers_pos = np.arange(0,20, 2) #markevery=markers_pos
ticks_pos = np.round(np.arange(-1, 1.1, .2),1).tolist()
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(x, y_cspline, color='orange', label = 'Classic spline', zorder=1) #, dashes=(5, 7),  linestyle='dashed',linewidth=1.3)
ax.plot(x, y_qspline, color='steelblue',  label = 'full Qspline') #, dashes=(5, 7),  linestyle='dashed',linewidth=1.3)
ax.plot(x, y, label='Activation', color = 'sienna', linestyle='dotted',#marker='x', markersize=4,
        dashes=(1,2), zorder=2)
ax.scatter(x_fid, fid, color = 'cornflowerblue', label = 'Fidelity', s = 10)
ax.set_xlim(-1.1, 1.1)
#ax.set_ylim(-.1,1.1)
ax.grid(alpha = 0.3)
# ax.set_xlabel(r'x')
ax.set_ylabel(r'$f(x)$', rotation = 0)
ax.set_xticks(ticks_pos)
# Creating legend and title for the figure. Legend created with figlegend(), title with suptitle()
# handles, labels = ax.get_legend_handles_labels()
# fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05),
#            ncol = 4, borderaxespad=0)
ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2),
           ncol = 4, borderaxespad=0. )
plt.savefig('results/' + label_function + '_linear_spline.png', dpi =1000)
plt.show()
plt.close()

