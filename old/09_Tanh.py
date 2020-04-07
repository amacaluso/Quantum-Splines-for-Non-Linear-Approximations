from Utils import *

label_function = 'tanh'

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

# hybrid Qspline
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(x, y_cspline, color='orange', label = 'Classic spline', zorder=1) #, dashes=(5, 7),  linestyle='dashed',linewidth=1.3)
ax.plot(x, y_qspline, color='steelblue',  label = 'Qspline') #, dashes=(5, 7),  linestyle='dashed',linewidth=1.3)
ax.plot(x, sig_y, label='Activation', color = 'sienna', linestyle='dotted', dashes=(1,1.5), zorder=2, linewidth=3)
ax.scatter(x_fid, fid, color = 'cornflowerblue', label = 'Fidelity', s = 10)
ax.set_xlim(-1.1, 1.1)
ax.grid(alpha = 0.3)
ax.set_ylabel(r'$f(x)$', rotation = 0)
ax.set_xticks(np.round(np.arange(-1, 1.1, .2),1).tolist())
ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2),
           ncol = 4, borderaxespad=0. )
plt.savefig('results/' + label_function + '_linear_spline.png', dpi =700)
plt.show()
plt.close()

