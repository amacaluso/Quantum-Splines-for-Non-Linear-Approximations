from Utils import *
folder = 'results/'

lower = -1
upper = 1
step = .1


'''Arcotangent'''

label = 'arctan'
data = pd.read_csv(folder + label + '_data.csv')
def function(x, c = 1):
  return c + np.arctan(x)
interval = np.arange(lower, upper+.03, step)
X = []
for i in range(1, len(interval)):
    #i =1
    X.append(np.arange(interval[i-1], interval[i], 0.05).tolist())
data_fid = pd.read_csv(folder + label + '_fidelity.csv')

# Creating the figure with four subplots, 2 per column/row
fig = plt.figure()
x_new = [item for sublist in X for item in sublist]
y = [function(j) for j in x_new]

ax = plt.subplot(221)
x_function = np.arange(lower, upper, step/4)
y_function = [function(j) for j in x_function]
ax.plot(x_function, y_function, label=label)
ax.plot(data.x, data.quantum_beta, color='red', linestyle='dotted')
ax.plot(data.x, data.classical_beta, color='limegreen', linestyle='dashed')
x_fid = np.arange(lower + .05, upper, step).tolist()
ax.scatter(x_fid, data_fid, color = 'cyan', label = 'Fidelity', s = 10)
ax.set_xlim(-1.1, 1.1)
#ax.set_ylim(-.1,1.1)
ax.grid(alpha = 0.3)
#ax.set_xlabel(r'x')
ax.set_title(label)




'''Relu'''

label = 'relu'
data = pd.read_csv(folder + label + '_data.csv')
def function(x, c = 1):
  return c + max(0.0, x)
interval = np.arange(lower, upper+.03, step)
X = []
for i in range(1, len(interval)):
    #i =1
    X.append(np.arange(interval[i-1], interval[i], 0.05).tolist())
data_fid = pd.read_csv(folder + label + '_fidelity.csv')

# Creating the figure with four subplots, 2 per column/row
x_new = [item for sublist in X for item in sublist]
y = [function(j) for j in x_new]

ax1 = plt.subplot(222)
x_function = np.arange(lower, upper, step/4)
y_function = [function(j) for j in x_function]
ax1.plot(x_function, y_function, label=label)
ax1.plot(data.x, data.quantum_beta, color='red', linestyle='dotted')
ax1.plot(data.x, data.classical_beta, color='limegreen', linestyle='dashed')
x_fid = np.arange(lower + .05, upper, step).tolist()
ax1.scatter(x_fid, data_fid, color = 'cyan', label = 'Fidelity', s = 10)
ax1.set_xlim(-1.1, 1.1)
#ax.set_ylim(-.1,1.1)
ax1.grid(True, which='both',alpha = 0.3)
#ax1.set_xlabel(r'x')
ax1.set_title(label)



'''Sigmoid'''

label = 'sigmoid'
data = pd.read_csv(folder + label + '_data.csv')
def function(x, c = 0):
  return c + 1 / (1 + math.exp(-4*x))
interval = np.arange(lower, upper+.03, step)
X = []
for i in range(1, len(interval)):
    #i =1
    X.append(np.arange(interval[i-1], interval[i], 0.05).tolist())
data_fid = pd.read_csv(folder + label + '_fidelity.csv')

# Creating the figure with four subplots, 2 per column/row
x_new = [item for sublist in X for item in sublist]
y = [function(j) for j in x_new]

ax2 = plt.subplot(223)
x_function = np.arange(lower, upper, step/4)
y_function = [function(j) for j in x_function]
ax2.plot(x_function, y_function, label=label)
ax2.plot(data.x, data.quantum_beta, color='red', linestyle='dotted')
ax2.plot(data.x, data.classical_beta, color='limegreen', linestyle='dashed')
x_fid = np.arange(lower + .05, upper, step).tolist()
ax2.scatter(x_fid, data_fid, color = 'cyan', label = 'Fidelity', s = 10)
ax2.set_xlim(-1.1, 1.1)
#ax.set_ylim(-.1,1.1)
ax2.grid(True, which='both',alpha = 0.3)
#ax2.set_xlabel(r'x')
ax2.set_title(label)


'''Elu'''

label = 'elu'
data = pd.read_csv(folder + label + '_data.csv')
def function(z,alpha = .3):
	return z if z >= 0 else alpha*(e**z -1)
interval = np.arange(lower, upper+.03, step)
X = []
for i in range(1, len(interval)):
    #i =1
    X.append(np.arange(interval[i-1], interval[i], 0.05).tolist())
data_fid = pd.read_csv(folder + label + '_fidelity.csv')

# Creating the figure with four subplots, 2 per column/row
x_new = [item for sublist in X for item in sublist]
y = [function(j) for j in x_new]

ax3 = plt.subplot(224)
x_function = np.arange(lower, upper, step/4)
y_function = [function(j) for j in x_function]
ax3.plot(x_function, y_function, label=label)
ax3.plot(data.x, data.quantum_beta, color='red', linestyle='dotted')
ax3.plot(data.x, data.classical_beta, color='limegreen', linestyle='dashed')
x_fid = np.arange(lower + .05, upper, step).tolist()
ax3.scatter(x_fid, data_fid, color = 'cyan', label = 'Fidelity', s = 10)
ax3.set_xlim(-1.1, 1.1)
#ax.set_ylim(-.1,1.1)
ax3.grid(True, which='both',alpha = 0.3)
#ax3.set_xlabel(r'x')
ax3.set_title(label)


#plt.legend()
# fig.subplots_adjust(top=0.9, left=0.1, right=0.9, bottom=0.12)  # create some space below the plots by increasing the bottom-value
# axlist.flatten()[-2].legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3)
fig.savefig(folder + 'act.png', dpi = 1000)#, bbox_extra_artists=(leg, suptitle,), bbox_inches='tight')
plt.show()
