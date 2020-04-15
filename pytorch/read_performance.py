import os
import json
import pandas as pd

dirname = 'logs'
experimental_combinations = ['p5_e10-50-5_b16_size5000',
                             'p10_e10-50-5_b16_size10000',
                             'p20_e10-50-5_b16_size15000'
                             'p5_e10-100-5_b16_size5000',
                             'p10_e10-100-5_b16_size10000',
                             'p20_e10-100-5_b16_size15000'
                             ]

fairness_keys = [
        'standalone_vs_federated_perturbed_corr_mean',
        'standalone_vs_final_corr_mean',
        'sharingcontribution_vs_final_corr_mean', ]

performance_keys = [
        'federated_final_performance_mean',
        'standalone_best_worker_mean',
        'CFFL_best_worker_mean',
        ]

fairness_rows = []
performance_rows = []
for folder in os.listdir(dirname):
        # with open(os.path.join(dirname, folder, 'log'), 'r') as log:
            # loginfo = log.readlines()
        
        n_workers = int(folder.split('_')[1][1:])
        fl_epochs = int(folder.split('-')[1])
        if fl_epochs != 100:
            continue
        try:
            with open(os.path.join(dirname, folder, 'aggregate_dict.txt')) as dict_log:
                aggregate_dict = json.loads(dict_log.read())
            f_data_row = ['P' + str(n_workers)] + [aggregate_dict[f_key][0] for f_key in fairness_keys]
            p_data_row = ['P' + str(n_workers)] + [aggregate_dict[p_key] * 100 for p_key in performance_keys]

            fairness_rows.append(f_data_row)
            performance_rows.append(p_data_row)
        except:
            pass
            


shorthand_f_keys = ['Distriubted', 'CFFL' ,'contri_v_final' ]
df = pd.DataFrame(fairness_rows, columns=[' '] + shorthand_f_keys).set_index(' ')
print(df.to_string())

shorthand_p_keys = ['Distributed', 'Standalone', 'CFFL']
pd.options.display.float_format = '{:,.2f}'.format
df = pd.DataFrame(performance_rows, columns=[' '] + shorthand_p_keys).set_index(' ').T
print(df.to_string())

