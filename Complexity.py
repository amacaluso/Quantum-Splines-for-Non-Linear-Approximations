import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
from sklearn.linear_model import LinearRegression

# to do: mettere il grafico sul condition num in funzione di lambda
# verificare definizione esatta di k e s

############################################
#### +++++++++++ Sparsity +++++++++++++ ####
############################################
dim = 45
N = np.arange(1,dim)
s = N
s_up = np.arange(1,dim)
s_dw = np.repeat(2, len(s_up))
k = 10

pow_s = 2
pow_k = 2
cost_hhl = (np.log2(N))*(k**pow_k)*(s**pow_s)
cost_hhl_up = (np.log2(N))*(k**pow_k)*(N**pow_s)
cost_hhl_dw = (np.log2(N))*(k**pow_k)*(s_dw**pow_s)
cost_gauss = [n**3 for n in N]
cost_stressen = [n**2.8 for n in N]
cost_conju = N*s*k

#plt.tight_layout()
plt.scatter(N, cost_gauss, label = 'Gauss-Jordan', color='lightblue', marker='+')
plt.scatter(N, cost_stressen, label = 'Stressen', color='lightgreen', marker='+')
plt.fill_between(N, cost_hhl_dw, cost_hhl_up, color = 'lightblue', label = 'HHL (no assumptions)', alpha = 0.3  )
plt.ylim(-1, 10**4)
plt.grid(alpha=0.3)
# plt2.set_ylabel(r"Cost",color="lightblue")
# plt2.tick_params(axis='y', labelcolor='skyblue')
# plt2.legend(loc = 'lower right')
#plt.savefig('Performance.png')
plt.legend(loc = 'upper left', framealpha=1, fancybox=True,)
plt.xlabel(r'$N$')
plt.ylabel('Computational Cost')
plt2=plt.twinx()
plt2.plot(N, (np.log2(N))*(k**pow_k)*(4**pow_s), label = 'HHL',  color='seagreen', linestyle='--', linewidth=2)
plt2.scatter(N, cost_conju, label = 'Conjugate Gradient', color='seagreen', marker='.' )
#plt2.grid(alpha=0.3)
plt2.legend(loc = 'lower right', framealpha=1, title = 'B-Splines')
# plt2.set_ylim(-1,3000)
#plt.title('Performance of Quantum SLP classifier')
plt.savefig('splines.png')
plt.show()
plt.close()





############################################
#### +++++++ Condition Number +++++++++ ####
############################################
dim = 45
N = np.arange(2,dim)
c = 0.1
k=[]
for i in range(dim-2):
    c = c + 0.03
    k.append(c)
k = np.array(k)

s = 10
k_up = np.arange(1,dim-1)
k_dw = np.repeat(1, len(k_up))

cost_hhl = (np.log2(N))*(k**2)*(s**pow_s)
cost_hhl_up = (np.log2(N))*(k_up**pow_k)*(s**pow_s)
cost_hhl_dw = (np.log2(N))*(k_dw**pow_k)*(s**pow_s)
cost_gauss = [n**3 for n in N]
cost_stressen = [n**2.8 for n in N]
cost_conju = N*s*k

plt.tight_layout()
plt.scatter(N, cost_gauss, label = 'Gauss-Jordan', color='lightblue', marker='+')
plt.scatter(N, cost_stressen, label = 'Stressen', color='lightgreen', marker='+')
plt.fill_between(N, cost_hhl_dw, cost_hhl_up, color = 'lightblue',
                 label = 'HHL (no assumptions)', alpha = 0.3  )
plt.ylim(-1, 20**3)
# plt2.set_ylabel(r"Cost",color="lightblue")
# plt2.tick_params(axis='y', labelcolor='skyblue')
# plt2.legend(loc = 'lower right')
#plt.savefig('Performance.png')
plt.legend(loc = 'upper left', framealpha=1, fancybox=True,)
plt.xlabel(r'$N$')
plt.ylabel('Computational Cost')
plt2=plt.twinx()
plt2.plot(N, (np.log2(N))*(k**pow_k)*(s**pow_s), label = 'HHL',  color='seagreen', linestyle='--', linewidth=2)
plt2.scatter(N, cost_conju, label = 'Conjugate Gradient', color='seagreen', marker='.' )
plt2.grid(alpha=0.3)
plt2.legend(loc = 'lower right', framealpha=1, title = 'B-Splines')
#plt2.set_ylim(-1,3000)
#plt.title('Performance of Quantum SLP classifier')
plt.show()
plt.close()




