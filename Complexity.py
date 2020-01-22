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
plt.scatter(N, cost_gauss, label = 'Gauss-Jordan', color='darkcyan', marker='+')
plt.scatter(N, cost_stressen, label = 'Stressen', color='blue', marker='+')
plt.fill_between(N, cost_hhl_dw, cost_hhl_up, color = 'lightblue',
                 label = 'HHL (no assumptions)', alpha = 0.3  )
plt.ylim(-1, 10**4)
plt.grid(alpha=0.3)
# plt2.legend(loc = 'lower right')
#plt.savefig('Performance.png')
plt.legend(loc = 'upper left', framealpha=1, fancybox=True,)
plt.xlabel(r'$N/sparsity^{-1}$')
plt.ylabel('Computational Cost')
plt2=plt.twinx()
plt2.tick_params(axis='y', labelcolor='seagreen')
# plt2.set_ylabel(r"Cost",color="seagreen")
plt2.plot(N, (np.log2(N))*(k**pow_k)*(4**pow_s), label = 'HHL',  color='seagreen',
          linestyle='--', linewidth=2)
plt2.scatter(N, cost_conju, label = 'Conjugate Gradient', color='seagreen', marker='.' )
#plt2.grid(alpha=0.3)
plt2.legend(loc = 'lower right', framealpha=1, title = 'B-Splines')
# plt2.set_ylim(-1,3000)
#plt.title('Performance of Quantum SLP classifier')
plt.savefig('splines.png')
plt.show()
plt.close()





###########################################
### +++++++ Condition Number +++++++++ ####
###########################################

dim = 3000
N = np.arange(2,dim)
s = 10
k = 10
k_up = 20
k_dw = 5
cost_hhl = (np.log2(N))*(k**2)*(s**pow_s)
cost_hhl_up = (np.log2(N))*(k_up**pow_k)*(s**pow_s)
cost_hhl_dw = (np.log2(N))*(k_dw**pow_k)*(s**pow_s)
cost_gauss = [n**3 for n in N]
cost_stressen = [n**2.8 for n in N]
cost_conju = N*s*k
cost_conju_up = N*s*k_up
cost_conju_dw = N*s*k_dw

# plt2=plt.twinx()
# plt2=plt.twiny()
# plt.tick_params(axis='y', labelcolor='g')
# plt.tick_params(axis='x', labelcolor='g')
#plt.set_ylabel(r"Cost",color="g")
plt.plot(N, (np.log2(N))*(k**pow_k)*(s**pow_s), label = 'HHL',  color='deepskyblue',
          linestyle='--', linewidth=2)
plt.fill_between(N, cost_hhl_dw, cost_hhl_up, color = 'deepskyblue', alpha = 0.1)
#                 label = 'HHL ($0.1<\kappa<20$)', alpha = 0.1  )
plt.plot(N, cost_conju, label = 'Conjugate Gradient', color='green', linestyle='dotted' )
plt.fill_between(N, cost_conju_dw, cost_conju_up, color = 'green', alpha = 0.3 )
#                 label = 'Conjugate Gradient ($0.1<\kappa<20$)', alpha = 0.1 )
plt.grid(alpha=0.3)
plt.legend(loc = 'upper left', framealpha=1, title = 'Ridge Regression')
plt.xlim(-1,dim)
#plt.title('Performance of Quantum SLP classifier')
plt.savefig('ridge.png')
plt.show()
plt.close()






############################################
#### +++++++ Condition Number +++++++++ ####
############################################

# dim = 3000
# N = np.arange(2,dim)
# s = 10
# k = 10
# k_up = 20
# k_dw = 5
# cost_hhl = (np.log2(N))*(k**2)*(s**pow_s)
# cost_hhl_up = (np.log2(N))*(k_up**pow_k)*(s**pow_s)
# cost_hhl_dw = (np.log2(N))*(k_dw**pow_k)*(s**pow_s)
# cost_gauss = [n**3 for n in N]
# cost_stressen = [n**2.8 for n in N]
# cost_conju = N*s*k
# cost_conju_up = N*s*k_up
# cost_conju_dw = N*s*k_dw
#
# # plt2=plt.twinx()
# # plt2=plt.twiny()
# # plt.tick_params(axis='y', labelcolor='g')
# # plt.tick_params(axis='x', labelcolor='g')
# #plt.set_ylabel(r"Cost",color="g")
# plt.plot(N, (np.log2(N))*(k**pow_k)*(s**pow_s), label = 'HHL',  color='deepskyblue',
#           linestyle='--', linewidth=2)
# plt.fill_between(N, cost_hhl_dw, cost_hhl_up, color = 'deepskyblue', alpha = 0.1)
# #                 label = 'HHL ($0.1<\kappa<20$)', alpha = 0.1  )
# plt.plot(N, cost_conju, label = 'Conjugate Gradient', color='green', linestyle='dotted' )
# plt.fill_between(N, cost_conju_dw, cost_conju_up, color = 'green', alpha = 0.3 )
# #                 label = 'Conjugate Gradient ($0.1<\kappa<20$)', alpha = 0.1 )
# plt.grid(alpha=0.3)
# plt.legend(loc = 'upper left', framealpha=1, title = 'Ridge Regression')
# plt.xlim(-1,dim)
# #plt.title('Performance of Quantum SLP classifier')
# plt.savefig('ridge.png')
# plt.show()
# plt.close()


















# dim = 100
#
# k = 10
# k_up = 20
# k_dw = 0.1
# cost_hhl = (np.log2(N))*(k**2)*(s**pow_s)
# cost_hhl_up = (np.log2(N))*(k_up**pow_k)*(s**pow_s)
# cost_hhl_dw = (np.log2(N))*(k_dw**pow_k)*(s**pow_s)
# cost_gauss = [n**3 for n in N]
# cost_stressen = [n**2.8 for n in N]
# cost_conju = N*s*k
# cost_conju_up = N*s*k_up
# cost_conju_dw = N*s*k_dw
#
# plt.tight_layout()
# plt.scatter(N, cost_gauss, label = 'Gauss-Jordan', color='tomato', marker='+')
# plt.scatter(N, cost_stressen, label = 'Stressen', color='salmon', marker='+')
# plt.fill_between(N, cost_hhl_dw, cost_hhl_up, color = 'lightcoral',
#                  label = 'HHL (no assumptions)', alpha = 0.3  )
# plt.ylim(-1, 400000)
# plt.xlim(-1, 200)
# #plt.legend(loc = 'lower right')
# plt.savefig('Performance.png')
# plt.legend(loc = 'upper left', framealpha=1, fancybox=True,)
# plt.xlabel(r'$N$')
# plt.ylabel('Computational Cost')
#
#
#
# dim = 3000
# N = np.arange(2,dim)
#
# k = 10
# k_up = 20
# k_dw = 0.1
# cost_hhl = (np.log2(N))*(k**2)*(s**pow_s)
# cost_hhl_up = (np.log2(N))*(k_up**pow_k)*(s**pow_s)
# cost_hhl_dw = (np.log2(N))*(k_dw**pow_k)*(s**pow_s)
# cost_gauss = [n**3 for n in N]
# cost_stressen = [n**2.8 for n in N]
# cost_conju = N*s*k
# cost_conju_up = N*s*k_up
# cost_conju_dw = N*s*k_dw
#
# plt2=plt.twinx()
# plt2=plt.twiny()
# plt2.tick_params(axis='y', labelcolor='g')
# plt2.tick_params(axis='x', labelcolor='g')
# plt2.set_ylabel(r"Cost",color="g")
# plt2.plot(N, (np.log2(N))*(k**pow_k)*(s**pow_s), label = 'HHL',  color='limegreen',
#           linestyle='--', linewidth=2)
# plt.fill_between(N, cost_hhl_dw, cost_hhl_up, color = 'green',
#                  label = 'HHL ($0.1<\kappa<20$)', alpha = 0.1  )
# plt2.plot(N, cost_conju, label = 'Conjugate Gradient', color='darkcyan', linestyle='dotted' )
# plt2.fill_between(N, cost_conju_dw, cost_conju_up, color = 'darkcyan',
#                  label = 'Conjugate Gradient ($0.1<\kappa<20$)', alpha = 0.1 )
# plt2.grid(alpha=0.3)
# plt2.legend(loc = 'upper left', framealpha=1, title = 'Ridge Regression')
# plt2.set_xlim(-1,dim)
# #plt.title('Performance of Quantum SLP classifier')
# plt.show()
# plt.close()


#
# N = np.arange(2,dim)
# c = 0.1
# k=[]
# for i in range(dim-2):
#     c = c + 1
#     k.append(c)
# k = np.array(k)
# # l = 1/k
# # s = 10
# # k_up = np.arange(1,dim-1)
# # k_dw = np.repeat(1, len(k_up))
# #
# # l_up = 1/k_up
# # l_dw = 1/k_dw
# #
# c = 0.1
# k=[]
# for i in range(dim-2):
#     c = c + 1
#     k.append(c)
# k = np.array(k)
# # l = 1/k
# # s = 10
# # k_up = np.arange(1,dim-1)
# # k_dw = np.repeat(1, len(k_up))
# #
# # l_up = 1/k_up
# # l_dw = 1/k_dw
# #

