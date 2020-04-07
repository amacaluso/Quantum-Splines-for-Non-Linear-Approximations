# Reference CG, qubic spline, equazione 6 equazione paragrafo 1

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


from Utils import *
find_N(3,10)

# to do: mettere il grafico sul condition num in funzione di lambda
# verificare definizione esatta di k e s

############################################
#### +++++++++++ Sparsity +++++++++++++ ####
############################################
dim = 80
N = np.arange(1,dim)
N_hhl = np.arange(1,dim,2)

s = N_hhl
s_up = np.arange(1,dim)
s_dw = np.repeat(2, len(s_up))
k = 2

pow_s = 2
pow_k = 2
cost_hhl_up = (np.log2(N))*(k**pow_k)*(N**pow_s)
cost_hhl_dw = (np.log2(N))*(k**pow_k)*(s_dw**pow_s)
cost_gauss = [n**3 for n in N]
cost_stressen = [n**2.8 for n in N]
cost_coppersmith = [n**2.37 for n in N]

s_fixed = 3
cost_conju = N_hhl*s_fixed*np.sqrt(k)

treshold = find_N(s_fixed, k)
#plt.tight_layout()
plt.figure(figsize=(6,3.5))
plt.fill_between(N, cost_hhl_dw, cost_hhl_up, color = 'lightblue',
                     label = 'HHL (no assumptions)', alpha = 0.3  )
plt.plot(N, cost_gauss, label = 'Gauss-Jordan', color='blue', linestyle = 'dotted')
plt.plot(N, cost_stressen, label = 'Strassen', color='deepskyblue', linestyle = 'dotted')
plt.plot(N, cost_coppersmith, label = 'Coppersmith', color='royalblue',
         linestyle = 'dotted') #(0, (3, 1, 1, 1)))
plt.scatter(N_hhl, cost_conju, label = 'Conjugate Gradient', color='limegreen', marker='+', s=15 )
plt.scatter(N_hhl, (np.log2(N_hhl))*(k**pow_k)*(s_fixed**pow_s), label = 'HHL',
            color='seagreen', marker='+', s=15) #linestyle='--', linewidth=.0001)
plt.xlabel(r'System size $(n)$')
plt.ylabel('Cost Complexity (hundreds operations)')
plt.xlim(-1, 80)
plt.ylim(-1, 500)
plt.grid(alpha=0.3)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.axvline(treshold, color = 'lightgrey', linestyle = (0, (3, 5, 1, 5)))
# Put a legend below current axis
plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.4), ncol =3, fontsize=10)
plt.xticks([0, 10, 20 ,30, 40, 50, 60, 70, 80 ], ha='left')
plt.tick_params(labelsize=12)
plt.text(81, 95, r'$s = 1$', color='grey')
plt.text(81, 220, r'$s = 3$', color='grey')
plt.text(6, 510, r'$s = n$', color='grey')
plt.text(treshold-2, -30, "{}".format(int(treshold)), color = 'grey')
#plt.yscale('log')
plt.savefig('results/complexity.png', bbox_inches='tight', dpi =400)
plt.show()
plt.close()

