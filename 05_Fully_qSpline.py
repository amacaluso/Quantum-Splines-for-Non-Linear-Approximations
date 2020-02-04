from Utils import *

## Tanh
def function(x, c = 1):
  return (c + np.tanh(x))*c/2
label_function = 'tanh'
execfile('06_quantum_splines.py')



## Sigmoid
def function(x, c = 0):
  return c + 1 / (1 + math.exp(-4*x))
label_function = 'sigmoid'
execfile('06_quantum_splines.py')


d_sig = pd.read_csv('results/sigmoid_full_data.csv')
d_tanh = pd.read_csv('results/tanh_full_data.csv')


interval = np.arange(lower,upper + .03, step)

X = []
for i in range(1, len(interval)):
    # i =1
    X.append(np.arange(interval[i - 1], interval[i], 0.05).tolist())

x = [item for sublist in X for item in sublist]
y = [(function(j)*2)-1 for j in x]
d_tanh.y = y

rss_quantum = [np.sum(np.square(d_sig.y - d_sig.quantum_beta)),
               np.sum(np.square(d_tanh.y - d_tanh.quantum_beta))]

rss_classic = [np.sum(np.square(d_sig.y - d_sig.classical_beta)),
               np.sum(np.square(d_tanh.y - d_tanh.classical_beta))]

fid_avg = [ np.average(fid_full.relu), np.average(fid_full.elu), np.average(fid_full.tanh),
            np.average(fid_full.sig), np.average(fid_full.arct)]


tab = pd.DataFrame([pd.Series(functions + ['arct']), pd.Series(rss_classical),
                   pd.Series(rss_quantum), pd.Series(fid_avg)])

tab = tab.transpose()
tab.columns = ['Function', 'RSS (classical)', 'RSS(Quantum)', 'AVG FIdelity']

tab.to_csv('results/table_results.csv', index = False)




# data['label'] = label_function
# data_sig = data.copy()
# data_fid_sig = F.copy()
#
#
# data.quantum_beta = data.quantum_beta - 1
# data.classical_beta = data.classical_beta - 1
# data['label'] = label_function
#
# data_tanh = data.copy()
# data_fid_tanh = F.copy()
#
# ## Relu
# def function(x, c = 1):
#   return c + max(0.0, x)
# label_function = 'relu'
# execfile('06_quantum_splines.py')
#
# data.quantum_beta = data.quantum_beta - 1
# data.classical_beta = data.classical_beta - 1
# data['label'] = label_function
#
# data_relu = data.copy()
# data_fid_relu = F.copy()
#
# ## Elu
# def function(z,alpha = .3, c = 2):
# 	return c + z if z >= 0 else c + alpha*(e**z -1)
# label_function = 'elu'
# execfile('06_quantum_splines.py')
#
# data.quantum_beta = data.quantum_beta - 1
# data.classical_beta = data.classical_beta - 1
# data['label'] = label_function
#
# data_elu = data.copy()
# data_fid_elu = F.copy()
#
#
# ## Arctan
# def function(x, c = 1):
#   return c + np.arctan(x)
# label_function = 'arctan'
# execfile('06_quantum_splines.py')
#
# data.quantum_beta = data.quantum_beta - 1
# data.classical_beta = data.classical_beta - 1
# data['label'] = label_function
#
# data_arct = data.copy()
# data_fid_arct = F.copy()
#
#
#
#

#
# d_arct = data_arct.loc[ :, ['y', 'quantum_beta', 'classical_beta']]
# d_arct.columns = ['y_arct', 'arct_quantum', 'arct_classical']
#
# d_sig = data_sig.loc[ :, ['y', 'quantum_beta', 'classical_beta']]
# d_sig.columns = ['y_sig',  'sig_quantum', 'sig_classical']
#
# d_relu = data_relu.loc[ :, ['y', 'quantum_beta', 'classical_beta']]
# d_relu.columns = ['y_relu', 'relu_quantum', 'relu_classical']
#
# d_elu = data_elu.loc[ :, ['y', 'quantum_beta', 'classical_beta']]
# d_elu.columns = ['y_elu','elu_quantum', 'elu_classical']
#
# x = data_tanh['x']
# y = data_tanh['y']
#
# data_full = pd.concat([x, d_sig, d_relu, d_elu, d_arct, d_tanh], axis = 1)
# fid_full = pd.concat([data_fid_sig['x'],
#                       data_fid_sig['Fidelity'], data_fid_relu['Fidelity'],
#                       data_fid_elu['Fidelity'], data_fid_arct['Fidelity'],
#                       data_fid_tanh['Fidelity']], axis = 1)
# fid_full.columns = ['x', 'sig', 'relu', 'elu', 'arct', 'tanh']
# data_full.to_csv( 'data_full.csv', index = False)
# fid_full.to_csv( 'fid_full.csv', index = False)
#
#
# x = data_full.x
# functions = ['relu', 'elu', 'tanh', 'sig']
#
# fig=plt.figure()
# ax1 = plt.subplot(221)
# ax2 = plt.subplot(222)
# ax3 = plt.subplot(223)
# ax4 = plt.subplot(224)
#
# f = functions[0]
# # y = data_full['y_'+f]
# # ax1.plot(x, y)
# # ax.plot(xs, cs(xs), label = 'Cubic Spline')
# qy = data_full[f+'_quantum']
# cy = data_full[f + '_classical']
# ax1.plot(x, qy, color='red', linestyle='dotted')
# ax1.plot(x, cy, color='green', linestyle='dashed')
# x_fid = fid_full.x
# fid = fid_full[f]
# ax1.scatter(x_fid, fid, color = 'limegreen', label = 'Fidelity', s = 7)
# ax1.set_xlim(-1.1, 1.1)
# ax1.set_ylim(-.1,1.1)
# ax1.grid(which = 'both', alpha = 0.3)
# # ax.set_xlabel(r'x')
# # ax1.set_ylabel(r'$f(x)$')
# ax1.text(0.80, 0.1, 'Relu',
#         transform=ax1.transAxes, ha="left")
#
#
#
# f = functions[1]
# # y = data_full['y_'+f]
# # ax2.plot(x, y)
# # ax.plot(xs, cs(xs), label = 'Cubic Spline')
# qy = data_full[f+'_quantum']
# cy = data_full[f + '_classical']
# ax2.plot(x, qy, color='red', linestyle='dotted')
# ax2.plot(x, cy, color='green', linestyle='dashed')
# x_fid = fid_full.x
# fid = fid_full[f]
# ax2.scatter(x_fid, fid, color = 'limegreen', label = 'Fidelity', s = 7)
# ax2.set_xlim(-1, 1.1)
# ax2.set_ylim(-.6,1.1)
# ax2.grid(alpha = 0.3)
# # ax.set_xlabel(r'x')
# # ax1.set_ylabel(r'$f(x)$')
# ax2.text(0.8, .1, 'Elu',
#         transform=ax2.transAxes, ha="left")
#
#
#
# f = functions[2]
# #y = data_full['y_'+f]
# #ax3.plot(x, y)
# # ax.plot(xs, cs(xs), label = 'Cubic Spline')
# qy = data_full[f+'_quantum']
# cy = data_full[f + '_classical']
# ax3.plot(x, qy, color='red', linestyle='dotted')
# ax3.plot(x, cy, color='green', linestyle='dashed')
# x_fid = fid_full.x
# fid = fid_full[f]
# ax3.scatter(x_fid, fid, color = 'limegreen', label = 'Fidelity', s = 7)
# ax3.set_xlim(-1.1, 1.1)
# #ax3.set_ylim(-.1,1.1)
# ax3.grid(alpha = 0.3)
# # ax.set_xlabel(r'x')
# # ax1.set_ylabel(r'$f(x)$')
# ax3.text(0.6, 0.1, 'Hyperbolic Tangent',
#         transform=ax3.transAxes, ha="center")
#
#
#
# f = functions[3]
# #y = data_full['y_'+f]
# #ax4.plot(x, y)
# # ax.plot(xs, cs(xs), label = 'Cubic Spline')
# qy = data_full[f+'_quantum']
# cy = data_full[f + '_classical']
# ax4.plot(x, qy, color='red', linestyle='dotted', label = 'Quantum Spline')
# ax4.plot(x, cy, color='green', linestyle='dashed', label = 'Classical Spline')
# x_fid = fid_full.x
# fid = fid_full[f]
# ax4.scatter(x_fid, fid, color = 'limegreen', label = 'Fidelity', s = 7)
# ax4.set_xlim(-1.1, 1.1)
# ax4.set_ylim(-.1,1.1)
# ax4.grid(which = 'major', alpha = 0.3)
# # ax.set_xlabel(r'x')
# # ax1.set_ylabel(r'$f(x)$')
# ax4.text(.65, .1, 'Sigmoid',
#         transform=ax4.transAxes, ha="left")
# # plt.legend()
# plt.savefig('results/full_spline.png', dpi =1000)
#
# # Creating legend and title for the figure. Legend created with figlegend(), title with suptitle()
# handles, labels = ax4.get_legend_handles_labels()
# fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.),
#            ncol = 3, borderaxespad=0)
#
# # plt.legend()
# plt.savefig('results/full_spline.png', dpi =800)
# plt.show()
# plt.close()
#
# rss_quantum = [np.sum(np.square(data_full.y_relu - data_full.relu_quantum)),
#                np.sum(np.square(data_full.y_elu - data_full.elu_quantum)),
#                np.sum(np.square(data_full.y_tanh - data_full.tanh_quantum)),
#                np.sum(np.square(data_full.y_sig - data_full.sig_quantum)),
#                np.sum(np.square(data_full.y_arct - data_full.arct_quantum))]
#
# rss_classical = [np.sum(np.square(data_full.y_relu - data_full.relu_classical)),
#                np.sum(np.square(data_full.y_elu - data_full.elu_classical)),
#                np.sum(np.square(data_full.y_tanh - data_full.tanh_classical)),
#                np.sum(np.square(data_full.y_sig - data_full.sig_classical)),
#                  np.sum(np.square(data_full.y_arct - data_full.arct_classical))]
#
# fid_avg = [ np.average(fid_full.relu), np.average(fid_full.elu), np.average(fid_full.tanh),
#             np.average(fid_full.sig), np.average(fid_full.arct)]
#
#
# tab = pd.DataFrame([pd.Series(functions + ['arct']), pd.Series(rss_classical),
#                    pd.Series(rss_quantum), pd.Series(fid_avg)])
#
# tab = tab.transpose()
# tab.columns = ['Function', 'RSS (classical)', 'RSS(Quantum)', 'AVG FIdelity']
#
# tab.to_csv('results/table_results.csv', index = False)
