from Utils_Spline import *

create_dir('results/single/')

def plot_activation(x, y, qy, cy, x_fid, fid, label, approach):
    # fig = plt.figure()
    ax = plt.subplot()
    ax.plot(x, y, color='orange', label='Classic spline', zorder=1)
    ax.plot(x, qy, color='steelblue', label='QSpline')
    ax.plot(x, cy, label='Activation', color='sienna', linestyle='dotted', dashes=(1, 1.5), zorder=2, linewidth=3)
    ax.scatter(x_fid, fid, color='cornflowerblue', label='Fidelity', s=10)
    ax.set_xlim(-1.1, 1.1)
    # ax.set_ylim(-.2, 1.05)
    ax.grid(alpha=0.3)
    ax.set_xticks(np.round(np.arange(-1, 1.1, .4), 1).tolist())
    ax.set_title(label, size=20)
    ax.tick_params(labelsize=18)
    plt.savefig('results/single/'+label+'_'+approach+'.png', dpi=300)
    plt.show()
    # ax.set_yticks(np.round(np.arange(-.2, 1.05, .2),1).tolist())
    #ax.text(coord[0], coord[1], label, transform=ax.transAxes, ha="left")

approach='Hybrid'

x, y, qy, cy, x_fid, fid = load_data('sigmoid', approach)
plot_activation(x, y, qy, cy, x_fid, fid, 'Sigmoid', approach)

x, y, qy, cy, x_fid, fid = load_data('tanh', approach)
plot_activation(x, y, qy, cy, x_fid, fid, 'Tanh', approach)

x, y, qy, cy, x_fid, fid = load_data('relu', approach)
plot_activation(x, y, qy, cy, x_fid, fid, 'Relu', approach)

x, y, qy, cy, x_fid, fid = load_data('elu', approach)
plot_activation(x, y, qy, cy, x_fid, fid, 'Elu', approach)



approach='Full'

x, y, qy, cy, x_fid, fid = load_data('sigmoid', approach)
plot_activation(x, y, qy, cy, x_fid, fid, 'Sigmoid', approach)

x, y, qy, cy, x_fid, fid = load_data('tanh', approach)
plot_activation(x, y, qy, cy, x_fid, fid, 'Tanh', approach)

x, y, qy, cy, x_fid, fid = load_data('relu', approach)
plot_activation(x, y, qy, cy, x_fid, fid, 'Relu', approach)

x, y, qy, cy, x_fid, fid = load_data('elu', approach)
plot_activation(x, y, qy, cy, x_fid, fid, 'Elu', approach)

# Plot Legend
# ax = plt.subplot()
# ax.plot(x, y, color='orange', label='Classic spline', zorder=1)
# ax.plot(x, qy, color='steelblue', label='Full QSpline')
# ax.plot(x, cy, label='Activation', color='sienna', linestyle='dotted', dashes=(1, 1.5), zorder=2, linewidth=3)
# ax.scatter(x_fid, fid, color='cornflowerblue', label='Fidelity', s=10)
# ax.set_xlim(-1.1, 1.1)
# # ax.set_ylim(-.2, 1.05)
# ax.grid(alpha=0.3)
# ax.set_xticks(np.round(np.arange(-1, 1.1, .4), 1).tolist())
# #ax.set_title(label, size=20)
# ax.tick_params(labelsize=18)
#
# fig = plt.figure(figsize=(6, 1))
# handles, labels = ax.get_legend_handles_labels()
# fig.legend(handles, labels, loc='center', ncol=4)
# plt.savefig('results/legend_Full.png', dpi=300, bbox_inches='tight')
# plt.show()
