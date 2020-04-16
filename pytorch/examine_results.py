import os
import json
import pandas as pd
import ast
import numpy as np

from plot import plot

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
            try:
                with open(os.path.join(dirname, folder, 'aggregate_dict.txt')) as dict_log:
                    aggregate_dict = json.loads(dict_log.read())
                f_data_row = ['P' + str(n_workers)] + [aggregate_dict[f_key][0] for f_key in fairness_keys]
                p_data_row = ['P' + str(n_workers)] + [aggregate_dict[p_key] * 100 for p_key in performance_keys]

                fairness_rows.append(f_data_row)
                performance_rows.append(p_data_row)
            except:
                pass
    
    shorthand_f_keys = ['Distriubted', 'CFFL' ,'Contributions_V_final' ]
    fair_df = pd.DataFrame(fairness_rows, columns=[' '] + shorthand_f_keys).set_index(' ')
    print(fair_df.to_string())
    fair_df.to_csv('adult_LogReg_b16_e5-100-1_lr0.001_fairness.csv')

    shorthand_p_keys = ['Distributed', 'Standalone', 'CFFL']
    pd.options.display.float_format = '{:,.2f}'.format
    perf_df = pd.DataFrame(performance_rows, columns=[' '] + shorthand_p_keys).set_index(' ').T
    print(perf_df.to_string())
    perf_df.to_csv('adult_LogReg_b16_e5-100-1_lr0.001_performance.csv')

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


def plot_convergence(dirname):
    dfs = []
    setups = []
    experiment_results = []

    for folder in os.listdir(dirname):
        if not 'complete.txt' in os.listdir(os.path.join(dirname, folder)):
            continue

        n_workers = int(folder.split('_')[1][1:])
        fl_epochs = int(folder.split('-')[1])

        columns = ['party' + str(i + 1) for i in range(n_workers)]
        
        experiments = []
        with open(os.path.join(dirname, folder, 'log'), 'r') as log:
            loginfo = log.readlines()
        # worker_fed_accs_lines = [line.replace('Workers federated accuracies:  ', '') for line in loginfo if 'Workers federated accuracies' in line]
        worker_fed_accs_lines = [line.replace('Workers CFFL      accuracies:  ', '') for line in loginfo if 'Workers CFFL      accuracies' in line]

        data_rows = []
        epoch_count = 0
        for line in worker_fed_accs_lines:
            line = ast.literal_eval(line)
            data_row = [ast.literal_eval(acc[:-1]) for acc in line]
            data_rows.append(data_row)

            epoch_count +=1
            if epoch_count == fl_epochs:
                experiments.append(np.array(data_rows))
                data_rows = []
                epoch_count = 0

        experiments = np.asarray(experiments)
        average_data = np.mean(experiments, axis=0)
        df = pd.DataFrame(average_data, columns = columns)

        dfs.append(df)

        setup = parse(folder)
        setups.append(setup)

        experiment_results.append((setup, df))

        figure_dir = os.path.join(dirname, folder, 'figure.png')
        if os.path.exists(figure_dir):
            os.remove(figure_dir)
        if not os.path.exists(figure_dir):
            plot(df,figure_dir)
            # exit()
    return experiment_results

if __name__ == '__main__':
    dirname = 'logs'
    experiment_results = plot_convergence(dirname)
    fair_df, perf_df = collect_and_compile_performance(dirname)