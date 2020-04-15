import os
import json
import pandas as pd

dirname = 'logs'


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


import ast
import numpy as np

from plot import plot

dfs = []
setups = []
experiment_results = []

for folder in os.listdir(dirname):
	n_workers = int(folder.split('_')[1][1:])
	fl_epochs = int(folder.split('-')[1])
	if fl_epochs != 100:
		continue

	columns = ['party' + str(i + 1) for i in range(n_workers)]

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

	setup = parse(folder)
	setups.append(setup)

	experiment_results.append((setup, df))

	figure_dir = os.path.join(dirname, folder, 'figure.png')
	if os.path.exists(figure_dir):
		os.remove(figure_dir)
	if not os.path.exists(figure_dir):
		plot(df,figure_dir)
		# exit()

