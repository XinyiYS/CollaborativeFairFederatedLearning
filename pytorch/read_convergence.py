import os
import json
import pandas as pd

dirname = 'logs'
experimental_combinations = ['p5_e10-50-5_b16_size5000',
                             'p10_e10-50-5_b16_size10000',
                             'p20_e10-50-5_b16_size15000'
                             'p5_e10-100-5_b16_size5000',
                             'p10_e10-100-5_b16_size10000',
                             # 'p20_e10-100-5_b16_size15000'
                             ]

import ast
import numpy as np
dfs = []

# for combi in experimental_combinations:
for folder in os.listdir(dirname):
	n_workers = int(folder.split('_')[1][1:])
	fl_epochs = int(folder.split('-')[1])
	columns = ['Worker_' + str(i + 1) for i in range(n_workers)]

	experiments = []

	with open(os.path.join(dirname, folder, 'log'), 'r') as log:
		loginfo = log.readlines()
	worker_fed_accs_lines = [line.replace('Workers federated accuracies:  ', '') for line in loginfo if 'Workers federated accuracies' in line]

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


print(dfs[1].describe())
print(dfs[1].to_string())
