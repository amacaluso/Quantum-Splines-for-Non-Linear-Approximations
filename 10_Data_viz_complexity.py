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
# dim = 80
# N = np.arange(1,dim)
# s = N
# s_up = np.arange(1,dim)
# s_dw = np.repeat(2, len(s_up))
# k = 10
#
# pow_s = 2
# pow_k = 2
# cost_hhl = (np.log2(N))*(k**pow_k)*(s**pow_s)
# cost_hhl_up = (np.log2(N))*(k**pow_k)*(N**pow_s)
# cost_hhl_dw = (np.log2(N))*(k**pow_k)*(s_dw**pow_s)
# cost_gauss = [n**3 for n in N]
# cost_stressen = [n**2.8 for n in N]
# cost_conju = N*s*np.sqrt(k)
#
# #plt.tight_layout()
# plt.scatter(N, cost_gauss, label = 'Gauss-Jordan', color='deepskyblue', marker='+')
# plt.scatter(N, cost_stressen, label = 'Stressen', color='blue', marker='+')
# plt.fill_between(N, cost_hhl_dw, cost_hhl_up, color = 'lightblue',
#                  label = 'HHL (no assumptions)', alpha = 0.3  )
# plt.ylim(-1, 10**4)
# plt.grid(alpha=0.3)
# # plt2.legend(loc = 'lower right')
# #plt.savefig('Performance.png')
# plt.legend(loc = 'upper left', framealpha=1, fancybox=True,)
# plt.xlabel(r'$n$')
# plt.ylabel('Time Complexity - Matrix Inversion')
# plt.xlim(-1, 80)
# plt2=plt.twinx()
# plt2.tick_params(axis='y', labelcolor='seagreen')
# plt.yticks()
# plt2.set_ylabel('Time Complexity - B-Spline',color="green", rotation =270, labelpad=13)
# plt2.plot(N, (np.log2(N))*(k**pow_k)*(4**pow_s), label = 'HHL',  color='seagreen',
#           linestyle='--', linewidth=2)
# plt2.scatter(N, cost_conju, label = 'Conjugate Gradient', color='limegreen', marker='.' )
# #plt2.grid(alpha=0.3)
# plt2.legend(loc = 'lower right', framealpha=1, title = 'B-Splines')
# # plt2.set_ylim(-1,3000)
# #plt.title('Performance of Quantum SLP classifier')
# plt.savefig('complexity.png')
# plt.show()
# plt.close()



import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.ticker as mtick


# to do: mettere il grafico sul condition num in funzione di lambda
# verificare definizione esatta di k e s

############################################
#### +++++++++++ Sparsity +++++++++++++ ####
############################################
dim = 80
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
cost_conju = N*s*np.sqrt(k)

#plt.tight_layout()

plt.plot(N, (np.log2(N))*(k**pow_k)*(4**pow_s), label = 'HHL',  color='seagreen',
          linestyle='--', linewidth=2)
plt.scatter(N, cost_conju, label = 'Conjugate Gradient', color='limegreen', marker='.' )
plt.scatter(N, cost_gauss, label = 'Gauss-Jordan', color='deepskyblue', marker='+')
plt.scatter(N, cost_stressen, label = 'Stressen', color='blue', marker='+')
plt.fill_between(N, cost_hhl_dw, cost_hhl_up, color = 'lightblue',
                 label = 'HHL (no assumptions)', alpha = 0.3  )
plt.xlabel(r'$n$')
plt.ylabel('Cost Complexity')
plt.xlim(-1, 80)
plt.ylim(-1, 15000)
plt.grid(alpha=0.3)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
# Put a legend below current axis
plt.legend(loc='lower right', #bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True)
plt.savefig('results/complexity.png')
plt.show()
plt.close()
