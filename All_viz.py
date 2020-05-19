from Utils_Spline import *

# execfile('Sigmoid.py')
# execfile('Tanh.py')
# execfile('Relu.py')
# execfile('Elu.py')
# execfile('Elu_v2.py')

def viz_all(approach = 'Hybrid'):

    fig = plt.figure()

    x, y, qy, cy, x_fid, fid = load_data('sigmoid', approach)
    single_plot(1, x, y, qy, cy, x_fid, fid, 'Sigmoid', coord=[0.68, .1])

    x, y, qy, cy, x_fid, fid = load_data('tanh', approach)
    single_plot(2, x, y, qy, cy, x_fid, fid, 'Tanh', coord=[0.78, 0.1])


    x, y, qy, cy, x_fid, fid = load_data('relu', approach)
    single_plot(3, x, y, qy, cy, x_fid, fid, 'Relu', coord=[0.78, 0.1])


    x, y, qy, cy, x_fid, fid = load_data('elu', approach)
    single_plot(4, x, y, qy, cy, x_fid, fid, 'Elu', coord=[0.83, 0.1])


    fig.savefig('results/' +'all_' + approach + '.png', dpi = 500, bbox_inches='tight')
    plt.show()
    plt.close()

viz_all(approach = 'Hybrid')
viz_all(approach = 'Full')

#### Performance Analysis


def performance(path='results/table_performance.csv'):

    x, y, qy, cy, x_fid, fid = load_data('sigmoid', 'Hybrid')
    rss_sig_hybrid = np.sum(np.square(y - qy))
    rss_sig_classic = np.sum(np.square(y - cy))
    h_sig_fid = fid

    x, y, qy, cy, x_fid, fid = load_data('tanh', 'Hybrid')
    rss_tanh_hybrid = np.sum(np.square(y - qy))
    rss_tanh_classic = np.sum(np.square(y - cy))
    h_tanh_fid = fid

    x, y, qy, cy, x_fid, fid = load_data('relu', 'Hybrid')
    rss_relu_hybrid = np.sum(np.square(y - qy))
    rss_relu_classic = np.sum(np.square(y - cy))
    h_relu_fid = fid

    x, y, qy, cy, x_fid, fid = load_data('elu', 'Hybrid')
    rss_elu_hybrid = np.sum(np.square(y - qy))
    rss_elu_classic = np.sum(np.square(y - cy))
    h_elu_fid = fid

    x, y, qy, cy, x_fid, fid = load_data('sigmoid', 'Full')
    rss_sig_full = np.sum(np.square(y - qy))
    f_sig_fid = fid

    x, y, qy, cy, x_fid, fid = load_data('tanh', 'Full')
    rss_tanh_full = np.sum(np.square(y - qy))
    f_tanh_fid = fid

    x, y, qy, cy, x_fid, fid = load_data('relu', 'Full')
    rss_relu_full = np.sum(np.square(y - qy))
    f_relu_fid = fid

    x, y, qy, cy, x_fid, fid = load_data('elu', 'Full')
    rss_elu_full = np.sum(np.square(y - qy))
    f_elu_fid = fid

    rss_hybrid = [rss_sig_hybrid, rss_tanh_hybrid, rss_relu_hybrid, rss_elu_hybrid]
    rss_full = [rss_sig_full, rss_tanh_full, rss_relu_full, rss_elu_full]

    rss_classic = [rss_sig_classic, rss_tanh_classic, rss_relu_classic, rss_elu_classic]
    fid_avg_hybrid = [ np.average(h_sig_fid), np.average(h_tanh_fid),np.average(h_relu_fid), np.average(h_elu_fid)]
    fid_avg_full = [ np.average(f_sig_fid), np.average(f_tanh_fid),np.average(f_relu_fid), np.average(f_elu_fid)]


    tab = pd.DataFrame([pd.Series(['Sigmoid', 'Tanh', 'Relu', 'Elu']),
                        pd.Series(rss_classic), pd.Series(rss_hybrid),
                        pd.Series(rss_full), pd.Series(fid_avg_hybrid), pd.Series(fid_avg_full)])

    tab = tab.transpose()
    tab.columns = ['Function', 'RSS (classic)', 'RSS (hybrid)', 'RSS (full)',
                   'AVG Fidelity (hybrid)', 'AVG Fidelity (full)']

    tab.to_csv(path, index = False)
    return tab

tab = performance()