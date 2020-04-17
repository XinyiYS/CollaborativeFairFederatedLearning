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
	setup['pretrain_epochs'] = int(param[2].split('-')[0][1:])
	setup['Communication Rounds'] = int(param[2].split('-')[1])
	setup['theta'] = float(param[6].replace('theta', ''))
	setup['E'] = int(param[2].split('-')[2])
	setup['B'] = int(param[3][1:])
	setup['size'] = int(param[4][4:])
	setup['lr'] = float(param[5][2:])
	return setup


def get_performance(accs_lines, n_runs, fl_epochs):

	final_accs_lines = [ accs_lines[ (i+1) * (fl_epochs+1) -1]     for  i in range(n_runs)  ]

	data_rows = []
	for line in final_accs_lines:
		data_row = [ast.literal_eval(acc[:-1]) for acc in ast.literal_eval(line)]
		data_rows.append(data_row)
	data_rows = np.asarray(data_rows)
	performance = np.mean(data_rows, axis=0).round(3)
	# workerId = np.mean(data_rows, axis=0).argmax()
	return performance


def get_best_cffl_worker(dirname, folder):
	# need to figure out which is the best worker
	fl_epochs = int(folder.split('-')[1])
	n_runs = int(folder[folder.find('runs')-1])

	with open(os.path.join(dirname, folder, 'log'), 'r') as log:
		loginfo = log.readlines()

	worker_cffl_accs_lines = [line.replace('Workers CFFL      accuracies:  ', '')
										   for line in loginfo if 'Workers CFFL      accuracies' in line]
	worker_standalone_accs_lines = [line.replace(
		'Workers standlone accuracies:  ', '') for line in loginfo if 'Workers standlone accuracies:  ' in line]
	worker_dssgd_accs_lines = [line.replace('Workers DSSGD     accuracies:  ', '')
												for line in loginfo if 'Workers DSSGD     accuracies:  ' in line]

	cffl_performances  = get_performance(worker_cffl_accs_lines, n_runs, fl_epochs)
	dssgd_performances = get_performance(worker_dssgd_accs_lines, n_runs, fl_epochs)
	standalone_performances = get_performance(worker_standalone_accs_lines, n_runs, fl_epochs)

	workerId = cffl_performances.argmax()
	performance_dict = {'cffl': cffl_performances, 'dssgd':dssgd_performances, 'standalone':standalone_performances}

	return workerId, performance_dict


def plot_convergence_for_one(dirname, mode='best', workerId=-1):

	for folder in os.listdir(dirname):
		if os.path.isfile(os.path.join(dirname, folder)) or not 'complete.txt' in os.listdir(os.path.join(dirname, folder)):
			continue

		if mode == 'best':
			# the best is the last worker
			workerId, _ = get_best_cffl_worker(dirname, folder)
		
		n_workers = int(folder.split('_')[1][1:])
		fl_epochs = int(folder.split('-')[1])

		experiments = []
		with open(os.path.join(dirname, folder, 'log'), 'r') as log:
			loginfo = log.readlines()

		worker_cffl_accs_lines = [line.replace('Workers CFFL      accuracies:  ', '')
											   for line in loginfo if 'Workers CFFL      accuracies' in line]
		worker_standalone_accs_lines = [line.replace(
			'Workers standlone accuracies:  ', '') for line in loginfo if 'Workers standlone accuracies:  ' in line]
		worker_dssgd_accs_lines = [line.replace('Workers DSSGD     accuracies:  ', '')
												for line in loginfo if 'Workers DSSGD     accuracies:  ' in line]


		data_rows = []
		epoch_count = 0
		for cffl_acc, standlone_acc, dssgd_acc in zip(worker_cffl_accs_lines, worker_standalone_accs_lines, worker_dssgd_accs_lines):
			cffl_acc = ast.literal_eval(ast.literal_eval(cffl_acc)[workerId][:-1])
			standlone_acc = ast.literal_eval(
				ast.literal_eval(standlone_acc)[workerId][:-1])
			dssgd_acc = ast.literal_eval(ast.literal_eval(dssgd_acc)[workerId][:-1])

			data_rows.append([standlone_acc, dssgd_acc, cffl_acc])

			epoch_count +=1
			if epoch_count == fl_epochs + 1:
				# skip one last row from performance
				data_rows=data_rows[:-1]
				experiments.append(np.array(data_rows))
				data_rows = []
				epoch_count = 0

		experiments = np.asarray(experiments)
		average_data = np.mean(experiments, axis=0)
		average_data /= 100.0 # make it percentage

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
		if os.path.isfile(os.path.join(dirname, folder)) or not 'complete.txt' in os.listdir(os.path.join(dirname, folder)):
			continue

		n_workers = int(folder.split('_')[1][1:])
		fl_epochs = int(folder.split('-')[1])

		columns = ['party' + str(i + 1) for i in range(n_workers)]
		
		experiments = []
		standlone_experiments = []
		with open(os.path.join(dirname, folder, 'log'), 'r') as log:
			loginfo = log.readlines()
		worker_cffl_accs_lines = [line.replace('Workers CFFL      accuracies:  ', '') for line in loginfo if 'Workers CFFL      accuracies' in line]
		
		worker_standalone_accs_lines = [line.replace('Workers standlone accuracies:  ', '') for line in loginfo if 'Workers standlone accuracies:  ' in line]
		# worker_dssgd_accs_lines = [line.replace('Workers DSSGD     accuracies:  ', '') for line in loginfo if 'Workers DSSGD     accuracies:  ' in line]

		data_rows = []
		standlone_data_rows = []
		epoch_count = 0
		for line, standlone_line in zip(worker_cffl_accs_lines, worker_standalone_accs_lines):

			line = ast.literal_eval(line)
			data_row = [ast.literal_eval(acc[:-1]) for acc in line]
			data_rows.append(data_row)

			standlone_line = ast.literal_eval(standlone_line)
			standlone_data_row = [ast.literal_eval(acc[:-1]) for acc in standlone_line]
			standlone_data_rows.append(standlone_data_row)

			epoch_count += 1
			if epoch_count == fl_epochs + 1:
				# read extra row to skip

				data_rows = data_rows[:-1]

				standlone_data_rows = standlone_data_rows[:-1]


				experiments.append(np.array(data_rows))
				standlone_experiments.append(np.array(standlone_data_rows))
				data_rows = []
				standlone_data_rows = []
				epoch_count = 0

		experiments = np.asarray(experiments)
		average_data = np.mean(experiments, axis=0)
		average_data /= 100.0 # make it percentage
		df = pd.DataFrame(average_data, columns = columns)
		dfs.append(df)

		setup = parse(folder)
		setups.append(setup)
		experiment_results.append((setup, df))

		standlone_experiments = np.asarray(standlone_experiments)
		standalone_average_data = np.mean(standlone_experiments, axis=0)
		standalone_average_data /= 100.0 # make it percentage
		standalone_df = pd.DataFrame(standalone_average_data, columns = columns)

		figure_dir = os.path.join(dirname, folder, 'figure.png')
		standlone_figure_dir = os.path.join(dirname, folder, 'standlone.png')
		if os.path.exists(figure_dir):
			os.remove(figure_dir)
		if not os.path.exists(figure_dir):
			plot(df,figure_dir)
			plot(standalone_df, standlone_figure_dir, standalone=True)

	return experiment_results

if __name__ == '__main__':
	dirname = 'logs'
	experiment_results = plot_convergence(dirname)
	plot_convergence_for_one(dirname, mode='best', workerId=-1)
