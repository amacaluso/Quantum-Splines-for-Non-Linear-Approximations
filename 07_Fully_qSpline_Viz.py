from Utils import *

### Data import
label_function = 'sigmoid'

'''Curve'''
data = pd.read_csv('results/'+label_function +'_full_data.csv')

sig_y = data.y
x = data.x

y_qspline = data.quantum_beta
y_cspline = data.classical_beta


'''Fidelity'''
data_fid = pd.read_csv('results/'+ label_function + '_fidelity.csv')
x_fid = data_fid.x
fid = data_fid.Fidelity


fig = plt.figure(figsize=(6,3))

# Full Qspline
ax = plt.subplot(121)
ax.plot(x, y_cspline, color='orange', label = 'Classic spline', zorder=1) #, dashes=(5, 7),  linestyle='dashed',linewidth=1.3)
ax.plot(x, y_qspline, color='steelblue',  label = 'Full Qspline') #, dashes=(5, 7),  linestyle='dashed',linewidth=1.3)
ax.plot(x, sig_y, label='Activation', color = 'sienna', linestyle='dotted', dashes=(1,1.5), zorder=2, linewidth=3)
ax.scatter(x_fid, fid, color = 'cornflowerblue', label = 'Fidelity', s = 10)
ax.set_xlim(-1.1, 1.1)
ax.grid(alpha = 0.3)
ax.set_xticks(np.round(np.arange(-1, 1.1, .4),1).tolist())
ax.text(0.6, 0.1, 'Sigmoid',
        transform=ax.transAxes, ha="left")


label_function = 'tanh'

'''Curve'''
data = pd.read_csv('results/'+label_function +'_full_data.csv')

x = data.x
data.y = (data.y*2)-1

def function(x, c = 1):
  return (c + np.tanh(x))*c/2


data.quantum_beta = (data.quantum_beta)
data.classical_beta = (data.classical_beta)
data.y = [function(j)*2-1 for j in x]

y_qspline = (data.quantum_beta*2)-1
y_cspline = (data.classical_beta*2)-1

'''Fidelity'''
data_fid = pd.read_csv('results/'+ label_function + '_fidelity.csv')
x_fid = data_fid.x
fid = data_fid.Fidelity

ax1 = plt.subplot(122)
# hybrid Qspline
ax1.plot(x, y_cspline, color='orange', label = 'Classic spline', zorder=1) #, dashes=(5, 7),  linestyle='dashed',linewidth=1.3)
ax1.plot(x, y_qspline, color='steelblue',  label = 'Full Qspline') #, dashes=(5, 7),  linestyle='dashed',linewidth=1.3)
ax1.plot(x, data.y, label='Activation', color = 'sienna', linestyle='dotted', dashes=(1,1.5), zorder=2, linewidth=3)
ax1.scatter(x_fid, fid, color = 'cornflowerblue', label = 'Fidelity', s = 10)
ax1.set_xlim(-1.1, 1.1)
ax1.grid(alpha = 0.3)
ax1.set_xticks(np.round(np.arange(-1, 1.1, .4),1).tolist())
ax1.text(0.8, 0.1, 'Tanh',
        transform=ax1.transAxes, ha="left")
#fig.tight_layout(pad=.03)
# handles, labels = ax1.get_legend_handles_labels()
# fig.legend(handles, labels, loc='upper center',
#            ncol = 4, bbox_to_anchor=(0.47, 1.05), borderaxespad=0.)
plt.savefig('results/Full_Qspline.png', dpi = 700, bbox_inches='tight')
plt.show()
plt.close()


# fig = plt.figure(figsize=(6,1))
# handles, labels = ax.get_legend_handles_labels()
# fig.legend(handles, labels, loc='center', ncol = 4)
# plt.savefig('results/legend.png', dpi = 700, bbox_inches='tight')
# plt.show()
