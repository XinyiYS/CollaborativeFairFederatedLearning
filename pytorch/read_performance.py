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

keys = ['standalone_vs_final_corr_mean',
        'sharingcontribution_vs_final_corr_mean',
        'standalone_vs_federated_perturbed_corr_mean',
        'federated_final_performance_mean',
        'CFFL_best_worker_mean',
        'standalone_best_worker_mean',
        ]

data_rows = []
for folder in os.listdir(dirname):
        # with open(os.path.join(dirname, folder, 'log'), 'r') as log:
            # loginfo = log.readlines()
        try:
            with open(os.path.join(dirname, folder, 'aggregate_dict.txt')) as dict_log:
                aggregate_dict = json.loads(dict_log.read())

            data_row = [folder] + [aggregate_dict[key] for key in keys]
            data_rows.append(data_row)
        except:
            pass


df = pd.DataFrame(data_rows, columns=['setup'] + keys)
print(df.to_string())
