from Utils import *
lower = -1
upper = 1
step = .1

data = pd.read_csv('data_full.csv')
data_fid = pd.read_csv('fid_full.csv')
x_fid = data_fid.x
x = data.x


fig = plt.figure()

''' Sigmoid '''
label = 'sigmoid'

# Creating the figure with four subplots, 2 per column/row
data.y_sig = data.y_sig

ax1 = plt.subplot(221)
ax1.plot(x, data.y_sig, color='orange', label = 'Classic spline', zorder=1)
ax1.plot(x, data.sig_quantum, color='steelblue',  label = 'Qspline')
ax1.plot(x, data.sig_classical, label='Activation', color = 'sienna', linestyle='dotted', dashes=(1,1.5), zorder=2, linewidth=3)
ax1.scatter(x_fid, data_fid.sig, color = 'cornflowerblue', label = 'Fidelity', s = 10)
ax1.set_xlim(-1.1, 1.1)
#ax.set_ylim(-.1,1.1)
ax1.grid(alpha = 0.3)
# ax.set_xlabel(r'x')
# ax3.set_ylabel(r'$f(x)$', rotation = 0)
ax1.set_xticks(np.round(np.arange(-1, 1.1, .4),1).tolist())
ax1.text(0.70, 0.1, 'Sigmoid',
        transform=ax1.transAxes, ha="left")



'''Tanh'''
label = 'tanh'
# Creating the figure with four subplots, 2 per column/row
data.y_tanh = (data.y_tanh*2)-1
x = data.x
data.tanh_quantum = (data.tanh_quantum*2)+1
data.tanh_classical = (data.tanh_classical*2)+1


ax2 = plt.subplot(222)
ax2.plot(x, data.y_tanh, color='orange', label = 'Classic spline', zorder=1)
ax2.plot(x, data.tanh_quantum, color='steelblue',  label = 'Qspline')
ax2.plot(x, data.tanh_classical, label='Activation', color = 'sienna', linestyle='dotted', dashes=(1,1.5), zorder=2, linewidth=3)
ax2.scatter(x_fid, data_fid.tanh, color = 'cornflowerblue', label = 'Fidelity', s = 10)
ax2.set_xlim(-1.1, 1.1)
ax2.grid(alpha = 0.3)
ax2.set_xticks(np.round(np.arange(-1, 1.1, .4),1).tolist())
ax2.text(0.80, 0.1, 'Tanh',
        transform=ax2.transAxes, ha="left")



'''Relu'''
label = 'relu'

# Creating the figure with four subplots, 2 per column/row
data.y_relu = data.y_relu-1

ax3 = plt.subplot(223)
ax3.plot(x, data.y_relu, color='orange', label = 'Classic spline', zorder=1)
ax3.plot(x, data.relu_quantum, color='steelblue',  label = 'Qspline')
ax3.plot(x, data.relu_classical, label='Activation', color = 'sienna', linestyle='dotted', dashes=(1,1.5), zorder=2, linewidth=3)
ax3.scatter(x_fid, data_fid.relu, color = 'cornflowerblue', label = 'Fidelity', s = 10)
ax3.set_xlim(-1.1, 1.1)
#ax.set_ylim(-.1,1.1)
ax3.grid(alpha = 0.3)
# ax.set_xlabel(r'x')
# ax3.set_ylabel(r'$f(x)$', rotation = 0)
ax3.set_xticks(np.round(np.arange(-1, 1.1, .4),1).tolist())
ax3.set_yticks(np.round(np.arange(-.2, 1.1, .4),1).tolist())
ax3.text(0.80, 0.1, 'Relu',
        transform=ax3.transAxes, ha="left")


''' Elu '''
label = 'elu'

# Creating the figure with four subplots, 2 per column/row
data.y_elu = data.y_elu-0.3
data.elu_quantum = data.elu_quantum + .7
data.elu_classical = data.elu_classical +.7

ax4 = plt.subplot(224)
ax4.plot(x, y, color='orange', label = 'Classic spline', zorder=1)
ax4.plot(x, data.elu_quantum, color='steelblue',  label = 'Qspline')
ax4.plot(x, data.elu_classical, label='Activation', color = 'sienna', linestyle='dotted', dashes=(1,1.5), zorder=2, linewidth=3)
ax4.scatter(x_fid, data_fid.elu, color = 'cornflowerblue', label = 'Fidelity', s = 10)
ax4.set_xlim(-1.1, 1.1)
ax4.grid(alpha = 0.3)
ax4.set_xticks(np.round(np.arange(-1, 1.1, .4),1).tolist())
ax4.set_yticks(np.round(np.arange(-.2, 1.1, .4),1).tolist())
ax4.text(0.80, 0.1, 'Elu',
        transform=ax4.transAxes, ha="left")
#fig.subplots_adjust(top=.9, left=0.1, right=0.9, bottom=.3)
# create some space below the plots by increasing the bottom-value
#fig.tight_layout(pad=.01)
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center',
           ncol = 4, bbox_to_anchor=(0.45, -.02), borderaxespad=1.)
fig.savefig('results/results_4x.png', bbox_inches='tight', dpi = 700)
plt.show()
plt.close()



rss_quantum = [
    np.sum(np.square(data.y_sig - data.sig_quantum)),
    np.sum(np.square(data.y_tanh - data.tanh_quantum)),
    np.sum(np.square(data.y_relu - data.relu_quantum)),
    np.sum(np.square(data.y_elu - data.elu_quantum))]

rss_classic = [
    np.sum(np.square(data.y_sig - data.sig_classical)),
    np.sum(np.square(data.y_tanh - data.tanh_classical)),
    np.sum(np.square(data.y_relu - data.relu_classical)),
    np.sum(np.square(data.y_elu - data.elu_classical))]

fid_avg = [ np.average(data_fid.sig), np.average(data_fid.tanh),
            np.average(data_fid.relu), np.average(data_fid.elu)]


tab = pd.DataFrame([pd.Series(['Sigmoid', 'Tanh', 'Relu', 'Elu']),
                   pd.Series(rss_classic),
                   pd.Series(rss_quantum),
                   pd.Series(fid_avg)])

tab = tab.transpose()
tab.columns = ['Function', 'RSS (classic)', 'RSS(quantum)', 'AVG Fidelity']

tab.to_csv('results/table_results.csv', index = False)
