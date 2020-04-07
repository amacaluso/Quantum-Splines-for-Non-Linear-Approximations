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


fig = plt.figure()

# Full Qspline
ax = plt.subplot(221)
ax.plot(x, y_cspline, color='orange', label = 'Classic spline', zorder=1) #, dashes=(5, 7),  linestyle='dashed',linewidth=1.3)
ax.plot(x, y_qspline, color='steelblue',  label = 'Full Qspline') #, dashes=(5, 7),  linestyle='dashed',linewidth=1.3)
ax.plot(x, sig_y, label='Activation', color = 'sienna', linestyle='dotted', dashes=(1,1.5), zorder=2, linewidth=3)
ax.scatter(x_fid, fid, color = 'cornflowerblue', label = 'Fidelity', s = 10)
ax.set_xlim(-1.1, 1.1)
ax.grid(alpha = 0.3)
ax.set_xticks(np.round(np.arange(-1, 1.1, .4),1).tolist())
ax.text(0.65, 0.1, 'Sigmoid',
        transform=ax.transAxes, ha="left")
#plt.show()

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

ax1 = plt.subplot(222)
# hybrid Qspline
ax1.plot(x, y_cspline, color='orange', label = 'Classic spline', zorder=1) #, dashes=(5, 7),  linestyle='dashed',linewidth=1.3)
ax1.plot(x, y_qspline, color='steelblue',  label = 'Full Qspline') #, dashes=(5, 7),  linestyle='dashed',linewidth=1.3)
ax1.plot(x, data.y, label='Activation', color = 'sienna', linestyle='dotted', dashes=(1,1.5), zorder=2, linewidth=3)
ax1.scatter(x_fid, fid, color = 'cornflowerblue', label = 'Fidelity', s = 10)
ax1.set_xlim(-1.1, 1.1)
ax1.grid(alpha = 0.3)
ax1.set_xticks(np.round(np.arange(-1, 1.1, .4),1).tolist())
ax1.text(0.75, 0.1, 'Tanh',
        transform=ax1.transAxes, ha="left")



'''Relu'''
label = 'relu'

data = pd.read_csv('results/relu_est_full.csv')
data_fid = pd.read_csv('results/relu_full.csv')

x_fid = (data_fid.upper + data_fid.lower) / 2
fid = data_fid.fidelity


# Creating the figure with four subplots, 2 per column/row
x = data.x
ax3 = plt.subplot(223)
ax3.plot(x, data.cspline, color='orange', label = 'Classic spline', zorder=1)
ax3.plot(x, data.qspline, color='steelblue',  label = 'Qspline')
ax3.plot(x, data.y, label='Activation', color = 'sienna', linestyle='dotted', dashes=(1,1.5), zorder=2, linewidth=3)
ax3.scatter(x_fid, fid, color = 'cornflowerblue', label = 'Fidelity', s = 10)
ax3.set_xlim(-1.1, 1.1)
#ax.set_ylim(-.1,1.1)
ax3.grid(alpha = 0.3)
# ax.set_xlabel(r'x')
# ax3.set_ylabel(r'$f(x)$', rotation = 0)
ax3.set_xticks(np.round(np.arange(-1, 1.1, .4),1).tolist())
ax3.set_yticks(np.round(np.arange(-.2, 1.1, .4),1).tolist())
ax3.text(0.78, 0.1, 'Relu',
        transform=ax3.transAxes, ha="left")

''' Elu '''
label = 'elu'

data = pd.read_csv('results/elu_est_full.csv')
data_fid = pd.read_csv('results/elu_full.csv')

x_fid = (data_fid.upper + data_fid.lower) / 2
fid = data_fid.fidelity


x = data.x
ax4 = plt.subplot(224)
ax4.plot(x, data.cspline, color='orange', label = 'Classic spline', zorder=1)
ax4.plot(x, data.qspline, color='steelblue',  label = 'Qspline')
ax4.plot(x, data.y, label='Activation', color = 'sienna', linestyle='dotted', dashes=(1,1.5), zorder=2, linewidth=3)
ax4.scatter(x_fid, fid, color = 'cornflowerblue', label = 'Fidelity', s = 10)
ax4.set_xlim(-1.1, 1.1)
#ax.set_ylim(-.1,1.1)
ax4.grid(alpha = 0.3)
# ax.set_xlabel(r'x')
# ax3.set_ylabel(r'$f(x)$', rotation = 0)
ax4.set_xticks(np.round(np.arange(-1, 1.1, .4),1).tolist())
ax4.set_yticks(np.round(np.arange(-.2, 1.1, .4),1).tolist())
ax4.text(0.83, 0.1, 'Elu',
        transform=ax4.transAxes, ha="left")
#fig.subplots_adjust(top=.9, left=0.1, right=0.9, bottom=.3)
# create some space below the plots by increasing the bottom-value
#fig.tight_layout(pad=.01)
# handles, labels = ax1.get_legend_handles_labels()
# fig.legend(handles, labels, loc='lower center',
#            ncol = 4, bbox_to_anchor=(0.45, -.02), borderaxespad=1.)

plt.savefig('results/Full_Qspline_x4.png', dpi = 400, bbox_inches='tight')
plt.show()
plt.close()


# fig = plt.figure(figsize=(6,1))
# handles, labels = ax.get_legend_handles_labels()
# fig.legend(handles, labels, loc='center', ncol = 4)
# plt.savefig('results/legend.png', dpi = 700, bbox_inches='tight')
# plt.show()





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


fig = plt.figure(figsize=(6,2.3))

# Full Qspline
ax = plt.subplot(121)
ax.plot(x, y_cspline, color='orange', label = 'Classic spline', zorder=1) #, dashes=(5, 7),  linestyle='dashed',linewidth=1.3)
ax.plot(x, y_qspline, color='steelblue',  label = 'Full Qspline') #, dashes=(5, 7),  linestyle='dashed',linewidth=1.3)
ax.plot(x, sig_y, label='Activation', color = 'sienna', linestyle='dotted', dashes=(1,1.5), zorder=2, linewidth=3)
ax.scatter(x_fid, fid, color = 'cornflowerblue', label = 'Fidelity', s = 10)
ax.set_xlim(-1.1, 1.1)
ax.grid(alpha = 0.3)
ax.set_xticks(np.round(np.arange(-1, 1.1, .4),1).tolist())
ax.text(0.65, 0.1, 'Sigmoid',
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
ax1.text(0.75, 0.1, 'Tanh',
        transform=ax1.transAxes, ha="left")
plt.savefig('results/Full_Qspline.png', dpi = 400, bbox_inches='tight')
plt.show()
plt.close()
