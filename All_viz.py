from Utils_Spline import *

execfile('Sigmoid.py')
execfile('Tanh.py')
execfile('Relu.py')
execfile('Elu.py')
execfile('Elu_v2.py')

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

    fig.savefig('results/' +'all_' + approach + '.png', dpi = 500)
    plt.show()
    plt.close()

viz_all(approach = 'Hybrid')
viz_all(approach = 'Full')

#### Performance Analysis


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
