### Experiments ###
from Utils_Spline import *

label = 'relu'

## Relu
# def relu(vector, c = 1, normalisation = False):
#     y = [c + max(0.0, x) for x in vector]
#     if normalisation:
#         y_norm = (y - np.min(y))/(np.max(y)-np.min(y))
#         return y_norm
#     else:
#         return y

def relu(x, c = 1, normalisation = False):
    return c + max(0.0, x)


relu_coef = coeff_splines_estimation(relu, 'relu')
relu_est = estimate_function(relu_coef, relu, label, c = 1, plot = True)

relu_est.y = relu_est.y - 1
relu_est.qspline = relu_est.qspline - 1
relu_est.cspline = relu_est.cspline - 1

if plot:
    fig, ax = plt.subplots(figsize=(6, 5))
    # Full Qspline
    ax.plot(x, relu_est.cspline, color='orange', label='Classic spline',
            zorder=1)  # , dashes=(5, 7),  linestyle='dashed',linewidth=1.3)
    ax.plot(x, relu_est.qspline, color='steelblue',
            label='Full Qspline')  # , dashes=(5, 7),  linestyle='dashed',linewidth=1.3)
    ax.plot(x, relu_est.y, label='Activation', color='sienna', linestyle='dotted', dashes=(1, 1.5), zorder=2,
            linewidth=3)
    ax.scatter(x_fid, fid, color='cornflowerblue', label='Fidelity', s=10)
    ax.set_xlim(-1.1, 1.1)
    ax.grid(alpha=0.3)
    ax.set_xticks(np.round(np.arange(-1, 1.1, .4), 1).tolist())
    ax.text(0.65, 0.1, label,
            transform=ax.transAxes, ha="left")
    plt.legend()
    plt.savefig('results/' + label + '_full.png', dpi=300)
    plt.show()


for x in vector:
    print(x)
## Elu
# def elu(vector, alpha = .3, c = .3, normalisation = False):
#     y = [c + z if z >= 0 else c + alpha * (e ** z - 1) for z in vector]
#     if normalisation:
#         y_norm =  (y - np.min(y))/(np.max(y)-np.min(y))
#         return y_norm
#     else:
#         return y
#
#
# # elu_coef = coeff_splines_estimation(elu, 'elu')
# elu_est = estimate_function(elu_coef, elu, 'elu', plot = True)
#
#
#
# ## Elu
# def function(z,alpha = .3, c = .3):
# 	return c + z if z >= 0 else c + alpha*(e**z -1)
#
#
# execfile('experiments_act_fun.py')
#
#
#
# function = elu
# label = 'elu'













from Utils import *

label_function = 'relu'

'''Curve'''
data = pd.read_csv('results/'+label_function +'_data.csv')
x = data.x

## Sigmoid
def function(x, c = 0):
  return c + 1 / (1 + math.exp(-4*x))

sig_y =  [function(i) for i in x]

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

