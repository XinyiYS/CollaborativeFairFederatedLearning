import os
import json
import pandas as pd
import ast
import numpy as np

from plot import plot, plot_one


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


def plot_convergence_for_one(dirname, mode='best', workerId=-1):
	if mode=='best':
		# the best is the last worker
		workerId = -1

	for folder in os.listdir(dirname):
		if not 'complete.txt' in os.listdir(os.path.join(dirname, folder)):
			continue

		fl_epochs = int(folder.split('-')[1])

		experiments = []
		with open(os.path.join(dirname, folder, 'log'), 'r') as log:
			loginfo = log.readlines()
	
		worker_cffl_accs_lines = [line.replace('Workers CFFL      accuracies:  ', '') for line in loginfo if 'Workers CFFL      accuracies' in line]
		worker_standalone_accs_lines = [line.replace('Workers standlone accuracies:  ', '') for line in loginfo if 'Workers standlone accuracies:  ' in line]
		worker_dssgd_accs_lines = [line.replace('Workers DSSGD     accuracies:  ', '') for line in loginfo if 'Workers DSSGD     accuracies:  ' in line]

		data_rows = []
		epoch_count = 0
		for cffl_acc, standlone_acc, dssgd_acc in zip(worker_cffl_accs_lines, worker_standalone_accs_lines, worker_dssgd_accs_lines):
			cffl_acc = ast.literal_eval(ast.literal_eval(cffl_acc)[workerId][:-1] )
			standlone_acc = ast.literal_eval(ast.literal_eval(standlone_acc)[workerId][:-1])
			dssgd_acc = ast.literal_eval(ast.literal_eval(dssgd_acc)[workerId][:-1])
			
			data_rows.append([standlone_acc, dssgd_acc, cffl_acc])

			epoch_count +=1
			if epoch_count == fl_epochs:
				experiments.append(np.array(data_rows))
				data_rows = []
				epoch_count = 0

		experiments = np.asarray(experiments)
		average_data = np.mean(experiments, axis=0)

		df = pd.DataFrame(average_data, columns=['Standlone', 'Distributed', 'CFFL'])
		figure_dir = os.path.join(dirname, folder, 'convergence_for_one.png')
		if os.path.exists(figure_dir):
			os.remove(figure_dir)
		if not os.path.exists(figure_dir):
			plot_one(df,figure_dir)
	return

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
		worker_cffl_accs_lines = [line.replace('Workers CFFL      accuracies:  ', '') for line in loginfo if 'Workers CFFL      accuracies' in line]
		# worker_standalone_accs_lines = [line.replace('Workers standlone accuracies:  ', '') for line in loginfo if 'Workers standlone accuracies:  ' in line]
		# worker_dssgd_accs_lines = [line.replace('Workers DSSGD     accuracies:  ', '') for line in loginfo if 'Workers DSSGD     accuracies:  ' in line]

		data_rows = []
		epoch_count = 0
		for line in worker_cffl_accs_lines:
			line = ast.literal_eval(line)
			data_row = [ast.literal_eval(acc[:-1]) for acc in line]
			data_rows.append(data_row)


			epoch_count +=1
			if epoch_count == fl_epochs:
				temp_df = pd.DataFrame(data_rows, columns = columns)
				
				figure_dir = os.path.join(dirname, folder, 'exp{}.png'.format(len(experiments)+1))

				plot(temp_df, figure_dir)
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

	return experiment_results

if __name__ == '__main__':
	dirname = 'logs'
	experiment_results = plot_convergence(dirname)
	plot_convergence_for_one(dirname, mode='best', workerId=-1)
