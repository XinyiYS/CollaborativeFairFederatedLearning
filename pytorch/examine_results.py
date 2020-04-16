import os
import json
import pandas as pd
import ast
import numpy as np

from plot import plot, plot_one
from read_convergence import plot_convergence,plot_convergence_for_one

fairness_keys = [
        'standlone_vs_rrdssgd_mean',
        'standalone_vs_final_mean',
        'sharingcontribution_vs_final_mean', ]

performance_keys = [
        'rr_dssgd_avg_mean',
        'standalone_best_worker_mean',
        'CFFL_best_worker_mean',
        ]

def collect_and_compile_performance(dirname):

    fairness_rows = []
    performance_rows = []
    for folder in os.listdir(dirname):
            if not 'complete.txt' in os.listdir(os.path.join(dirname, folder)):
                continue

            n_workers = int(folder.split('_')[1][1:])
            fl_epochs = int(folder.split('-')[1])
            theta = float(folder.split('_')[6].replace('theta', ''))
            try:
                with open(os.path.join(dirname, folder, 'aggregate_dict.txt')) as dict_log:
                    aggregate_dict = json.loads(dict_log.read())
                f_data_row = ['P' + str(n_workers) + '_' + str(theta)] + [aggregate_dict[f_key][0] for f_key in fairness_keys]
                p_data_row = ['P' + str(n_workers) + '_' + str(theta)] + [aggregate_dict[p_key] * 100 for p_key in performance_keys]

                fairness_rows.append(f_data_row)
                performance_rows.append(p_data_row)
            except:
                pass
    
    shorthand_f_keys = ['Distriubted', 'CFFL' ,'Contributions_V_final' ]
    fair_df = pd.DataFrame(fairness_rows, columns=[' '] + shorthand_f_keys).set_index(' ')
    print(fair_df.to_string())
    
    fair_df.to_csv( os.path.join(dirname, 'fairness.csv'))

    shorthand_p_keys = ['Distributed', 'Standalone', 'CFFL']
    pd.options.display.float_format = '{:,.2f}'.format
    perf_df = pd.DataFrame(performance_rows, columns=[' '] + shorthand_p_keys).set_index(' ').T
    print(perf_df.to_string())
    perf_df.to_csv( os.path.join(dirname, 'performance.csv'))

    return fair_df, perf_df


def parse(folder):
    setup = {}
    param = folder.split('_')
    setup['split'] = param[0]
    setup['P'] = int(param[1][1:])
    setup['pretrain_epochs'] =int(param[2].split('-')[0][1:])
    setup['Communication Rounds'] = int(param[2].split('-')[1])
    setup['E'] = int(param[2].split('-')[2])
    setup['B'] = int(param[3][1:])
    setup['size'] = int(param[4][4:])
    setup['lr'] = float(param[5][2:])
    return setup



if __name__ == '__main__':
    dirname = 'logs/e1-100-5'
    experiment_results = plot_convergence(dirname)
    plot_convergence_for_one(dirname)
    fair_df, perf_df = collect_and_compile_performance(dirname)