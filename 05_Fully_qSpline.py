from Utils import *

## Tanh
def function(x, c = 1):
  return (c + np.tanh(x))*c/2
# label_function = 'tanh'
# execfile('06_quantum_splines.py')



## Sigmoid
# def function(x, c = 0):
#   return c + 1 / (1 + math.exp(-4*x))
# label_function = 'sigmoid'
# execfile('06_quantum_splines.py')


d_sig = pd.read_csv('results/sigmoid_full_data.csv')
d_tanh = pd.read_csv('results/tanh_full_data.csv')


#######################################################
y = [(function(x)*2)-1 for x in d_tanh.x]
d_tanh.y = y

d_tanh.quantum_beta = (d_tanh.quantum_beta*2)-1
d_tanh.classical_beta = (d_tanh.classical_beta*2)-1
#######################################################


rss_quantum = [np.sum(np.square(d_sig.y - d_sig.quantum_beta)),
               np.sum(np.square(d_tanh.y - d_tanh.quantum_beta))]

rss_classic = [np.sum(np.square(d_sig.y - d_sig.classical_beta)),
               np.sum(np.square(d_tanh.y - d_tanh.classical_beta))]


fid_sig = pd.read_csv('results/sigmoid_full_fidelity.csv')
fid_tanh = pd.read_csv('results/tanh_full_fidelity.csv')


fid_avg = [ np.average(fid_sig.Fidelity), np.average(fid_tanh.Fidelity)]


tab = pd.DataFrame([pd.Series(['Sigmoid', 'Tanh']), pd.Series(rss_classic),
                   pd.Series(rss_quantum), pd.Series(fid_avg)])

tab = tab.transpose()
tab.columns = ['Function', 'RSS (classical)', 'RSS(Quantum)', 'AVG FIdelity']

tab.to_csv('results/table_full_qspline_results.csv', index = False)


# hybrid Qspline
fig = plt.figure()
ax = plt.subplot(121)


ax.plot(d_sig.x, d_sig.classical_beta, color='orange', label = 'Classic spline', zorder=1)
ax.plot(d_sig.x, d_sig.quantum_beta, color='steelblue',  label = 'Qspline')
ax.plot(d_sig.x, d_sig.y, label='Activation', color = 'sienna', linestyle='dotted', dashes=(1,1.5), zorder=2, linewidth=3)
ax.scatter(fid_sig.x, fid_sig.Fidelity, color = 'cornflowerblue', label = 'Fidelity', s = 10)
ax.set_xlim(-1.1, 1.1)
#ax.set_ylim(-.1,1.1)
ax.grid(alpha = 0.3)
# ax.set_xlabel(r'x')
ax.set_ylabel(r'$f(x)$', rotation = 0)
ax.set_xticks(np.round(np.arange(-1, 1.1, .4),1).tolist())
ax.text(0.6, 0.1, 'Sigmoid',
        transform=ax.transAxes, ha="left")


ax1 = plt.subplot(122)
ax1.plot(d_tanh.x, d_tanh.classical_beta, color='orange', label = 'Classic spline', zorder=1)
ax1.plot(d_tanh.x, d_tanh.quantum_beta, color='steelblue',  label = 'Qspline')
ax1.plot(d_tanh.x, d_tanh.y, label='Activation', color = 'sienna', linestyle='dotted', dashes=(1,1.5), zorder=2, linewidth=3)
ax1.scatter(fid_sig.x, fid_sig.Fidelity, color = 'cornflowerblue', label = 'Fidelity', s = 10)
ax1.set_xlim(-1.1, 1.1)
ax1.grid(alpha = 0.3)
ax1.set_xticks(np.round(np.arange(-1, 1.1, .4),1).tolist())
# handles, labels = ax.get_legend_handles_labels()
# fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -.05),
#            ncol = 4, borderaxespad=-0.5)
ax1.text(0.80, 0.1, 'Tanh',
        transform=ax1.transAxes, ha="left")
plt.savefig('results/full_qSpline.png', dpi =1000)
plt.show()
plt.close()


