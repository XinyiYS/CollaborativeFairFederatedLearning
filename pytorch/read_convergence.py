import os
import json
import pandas as pd
import ast
import numpy as np

from plot import plot



from collections import defaultdict

keys = ['worker_model_test_accs_before  -  ', 
		'dssgd_val_accs  -  ', 
		'DSSGD_model_test_accs  -  ',
		'worker_standalone_test_accs  -  ',
		'worker_model_test_accs_after  -  ',
		'worker_model_improvements  -  ',
		'credits  -  ',
		'federated_val_acc  -  '
]

key_map = {'DSSGD_model_test_accs': 'DSSGD',
           'worker_standalone_test_accs': 'Standalone',
           'cffl_test_accs': 'CFFL',
           # 'credits':'credits'
           }


def parse(folder):
	setup = {}
	param = folder.split('_')

	setup['name'] = 'adult'
	if 'MLP_Net' in folder or 'CNN_Net' in folder or 'classimbalance' in folder:
		setup['name'] = 'mnist'
	setup['model'] = param[-1]
	if setup['model'] == 'LogisticRegression':
		setup['model'] = 'LR'

	setup['split'] = param[0]
	setup['P'] = int(param[1][1:])
	setup['pretrain_epochs'] = int(param[2].split('-')[0][1:])
	setup['Communication Rounds'] = int(param[2].split('-')[1])
	setup['theta'] = float(param[6].replace('theta', ''))
	setup['E'] = int(param[2].split('-')[2])
	setup['B'] = int(param[3][1:])
	setup['size'] = int(param[4][4:])
	setup['lr'] = float(param[5][2:])
	setup['alpha'] = int(folder[folder.find('_a')+2])
	return setup


def get_cffl_best(dirname, folder):
	performance_dicts = get_performance_dicts(dirname, folder)
	performance_dict = performance_dicts[0]
	performance_dict_pretrain = performance_dicts[1]

	avg_accs = {}
	for key in key_map:
	    avg_acc = np.asarray(performance_dict[key]).mean(axis=0)
	    avg_acc = avg_acc[:-1]  # exclude the last repeated line
	    avg_acc = avg_acc[:, 1:]  # exclude the freerider
	    # print(avg_accs.shape)

	    avg_accs[key_map[key]] = avg_acc

	cffl_accs = avg_accs['CFFL']
	standalone_accs = avg_accs['Standalone']
	dssgd_accs = avg_accs['DSSGD']

	best_worker_ind = cffl_accs[-1].argmax()

	cffl_accs_pretrain = np.asarray(performance_dict_pretrain['cffl_test_accs']).mean(axis=0)
	cffl_accs_pretrain = cffl_accs_pretrain[:-1][:, 1:]


	return  [dssgd_accs[-1][best_worker_ind], standalone_accs[-1][best_worker_ind], cffl_accs[-1][best_worker_ind], cffl_accs_pretrain[-1][best_worker_ind] ]


def save_acc_dfs(dirname, folder, dfs):
	directory = os.path.join(dirname, folder, 'acc_dfs')
	try:
		os.mkdir(directory)
	except:
		pass
	[df.to_csv(os.path.join(directory, csv_name), index=False) for df, csv_name in zip(dfs, ['cffl.csv', 'standalone.csv', 'worker.csv'])]
	print('saving computed csvs to: ', directory)
	
	return


def get_performance_dicts(dirname, folder):
    logfiles = ['performance_dict.log', 'performance_dict_pretrain.log']
    performance_dicts = []
    for logfile in logfiles:
        with open(os.path.join(dirname, folder, logfile), 'r') as log:

            lines = log.read()
            lines = lines.replace('}{"shard_sizes"', '}\n{"shard_sizes"')
            split_lines = lines.split('\n')

        performance_dict = {}
        loaded_temp_dicts = [json.loads(exp) for exp in split_lines]
        dict_keys = loaded_temp_dicts[0].keys()
        # print(logfile, dict_keys)

        for key in dict_keys:
            performance_dict[key] = [temp_dict[key] for temp_dict in loaded_temp_dicts]

        performance_dicts.append(performance_dict)
    return performance_dicts

def plot_convergence(dirname):
	dfs = []
	setups = []
	experiment_results = []

	for folder in os.listdir(dirname):
		if os.path.isfile(os.path.join(dirname, folder)) or not 'complete.txt' in os.listdir(os.path.join(dirname, folder)):
			continue

		performance_dicts = get_performance_dicts(dirname, folder)
		performance_dict = performance_dicts[0]
		performance_dict_pretrain = performance_dicts[1]

		setup = parse(folder)
		n_workers = setup['P']
		columns = ['party' + str(i + 1) for i in range(n_workers)]

		avg_dfs = {}
		for key in key_map:
		    avg_accs = np.asarray(performance_dict[key]).mean(axis=0)
		    avg_accs = avg_accs[:-1]  # exclude the last repeated line
		    avg_accs = avg_accs[:, 1:]  # exclude the freerider
		    # print(avg_accs.shape)

		    avg_dfs[key_map[key]] = pd.DataFrame(data=avg_accs, columns=columns)

		cffl_df = avg_dfs['CFFL']
		standalone_df = avg_dfs['Standalone']
		dssgd_df = avg_dfs['DSSGD']

		best_worker_ind = cffl_df.iloc[-1].argmax()

		cffl_avg_acc_pretrain = np.asarray(performance_dict_pretrain['cffl_test_accs']).mean(axis=0)
		cffl_avg_acc_pretrain = cffl_avg_acc_pretrain[:-1][:, 1:]
		cffl_df_pretrain = pd.DataFrame(data=cffl_avg_acc_pretrain, columns=columns)

		worker_df = pd.DataFrame(data={'Standlone': standalone_df.iloc[:, best_worker_ind],
		                               'Distributed': dssgd_df.iloc[:, best_worker_ind],
		                               'CFFL (w pretrain)': cffl_df_pretrain.iloc[:, best_worker_ind],
		                               'CFFL (w/o pretrain)': cffl_df.iloc[:, best_worker_ind],
		                               })

		cffl_figure_dir = os.path.join(dirname, folder, 'figure.png')
		standlone_figure_dir = os.path.join(dirname, folder, 'standlone.png')
		worker_figure_dir = os.path.join(dirname, folder, 'convergence_for_one.png')



		if os.path.exists(cffl_figure_dir):
			os.remove(cffl_figure_dir)
		plot(cffl_df, cffl_figure_dir, name=setup['name'], plot_type=0)

		if os.path.exists(standlone_figure_dir):
			os.remove(standlone_figure_dir)
		plot(standalone_df, standlone_figure_dir, name=setup['name'], plot_type=1)

		if os.path.exists(worker_figure_dir):
			os.remove(worker_figure_dir)
		plot(worker_df, worker_figure_dir, name=setup['name'], plot_type=2)
	return


if __name__ == '__main__':
	dirname = 'logs'
	experiment_results = plot_convergence(dirname)


'''

def load_acc_dfs(dirname, folder):
	directory = os.path.join(dirname, folder, 'acc_dfs')
	try:
		cffl_df = pd.read_csv(os.path.join(directory, 'cffl.csv'),  index_col=False)
		standalone_df = pd.read_csv(os.path.join(directory, 'standalone.csv'),  index_col=False)
		worker_df = pd.read_csv(os.path.join(directory, 'worker.csv'),  index_col=False)
		return cffl_df, standalone_df, worker_df
	except:
		return 


def get_acc_lines(dirname, folder):
	logfiles = ['performance_dict.log', 'performance_dict_pretrain.log']

	for logfile in logfiles:

		with open(os.path.join(dirname, folder, logfile), 'r') as log:
		   loglines = log.readlines()

		acc_lines = defaultdict(list)
		for line in loglines:
		   for key in key_map:
			   if key in line:
				   acc_lines[key_map[key]].append(ast.literal_eval(line.replace(key,'')) [:-1])
	return acc_lines



def find_without_pretraining(dirname, folder, best_worker_ind):
	setup = parse(folder)
	pretrain_epochs = setup['pretrain_epochs']
	nopretrain_folder = folder.replace('_e{}-'.format(str(pretrain_epochs)), '_e0-')
	acc_lines = get_acc_lines(dirname, nopretrain_folder)
	cffl_accs_lines = acc_lines['CFFL']
	return np.asarray(cffl_accs_lines).mean(axis=0)[:, best_worker_ind]


def get_fairness(acc_lines):
	from scipy.stats import pearsonr
	Distributed_f = 0
	CFFL_f = 0
	
	DSSGD_accs = np.asarray(acc_lines['DSSGD'])
	Standalone_accs = np.asarray(acc_lines['Standalone'])
	CFFL_accs = np.asarray(acc_lines['CFFL'])

	n_experiments = DSSGD_accs.shape[0]
	for experiment in range(n_experiments):
		Distributed_f += pearsonr(DSSGD_accs[experiment][-1], Standalone_accs[experiment][-1])[0]
		CFFL_f += pearsonr(CFFL_accs[experiment][-1], Standalone_accs[experiment][-1])[0]

	Distributed_f /= n_experiments
	CFFL_f /= n_experiments
	return Distributed_f,CFFL_f


def plot_convergence(dirname):
	dfs = []
	setups = []
	experiment_results = []

	for folder in os.listdir(dirname):
		if os.path.isfile(os.path.join(dirname, folder)) or not 'complete.txt' in os.listdir(os.path.join(dirname, folder)):
			continue

		setup = parse(folder)
		loaded = load_acc_dfs(dirname, folder)
		if setup['pretrain_epochs'] == 0:
			continue

		if not loaded:

			# read accuracies from the folder
			acc_lines = get_acc_lines(dirname, folder)


			n_workers = int(folder.split('_')[1][1:])
			columns = ['party' + str(i + 1) for i in range(n_workers)]


			fairness = get_fairness(acc_lines)
			# construct nparrays, and then dfs
			avg_dfs = {}
			for com_protocol, acc_lines in acc_lines.items():
				avg_acc = np.asarray(acc_lines).mean(axis=0)
				avg_dfs[com_protocol] = pd.DataFrame(data=avg_acc, columns=columns)                

			cffl_df = avg_dfs['CFFL']
			standalone_df = avg_dfs['Standalone']
			dssgd_df = avg_dfs['DSSGD']

			best_worker_ind = cffl_df.iloc[-1].argmax()


			worker_df = pd.DataFrame(data = 
				{'Standlone':standalone_df.iloc[:, best_worker_ind],
				 'Distributed':dssgd_df.iloc[:, best_worker_ind],
					'CFFL (w pretrain)': cffl_df.iloc[:, best_worker_ind]})

			# save_acc_dfs(dirname, folder, [cffl_df, standalone_df, worker_df])
			# print(worker_df.head())
		else:
			cffl_df, standalone_df, worker_df = loaded

		cffl_figure_dir = os.path.join(dirname, folder, 'figure.png')
		standlone_figure_dir = os.path.join(dirname, folder, 'standlone.png')
		worker_figure_dir = os.path.join(dirname, folder, 'convergence_for_one.png')

		try:
			worker_acc_without_pretraining = find_without_pretraining(dirname, folder, best_worker_ind)
			worker_df['CFFL (w/o pretrain)'] = worker_acc_without_pretraining
		except Exception as e:
			print(str(e))
			#  no corresponding w/o pretrain experimental results
			pass
		


		if os.path.exists(cffl_figure_dir):
			os.remove(cffl_figure_dir)
		plot(cffl_df, cffl_figure_dir, name=setup['name'], plot_type=0)

		if os.path.exists(standlone_figure_dir):
			os.remove(standlone_figure_dir)
		plot(standalone_df, standlone_figure_dir, name=setup['name'], plot_type=1)

		if os.path.exists(worker_figure_dir):
			os.remove(worker_figure_dir)
		plot(worker_df, worker_figure_dir, name=setup['name'], plot_type=2)

	return


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
			plot(df, figure_dir, standalone=True)

	return


def get_performance(accs_lines, n_runs, fl_epochs):
	"""
	Arguments: in the accs_lines read from the log file, could be standalone, cffl, dssgd

	Returns the performance over the fl_epochs, averaged over n_runs
	"""

	final_accs_lines = [accs_lines[
		(i + 1) * (fl_epochs + 1) - 1] for i in range(n_runs)]

	data_rows = []
	for line in final_accs_lines:
		data_row = [ast.literal_eval(acc[:-1])
					for acc in ast.literal_eval(line)]
		data_rows.append(data_row)
	data_rows = np.asarray(data_rows)
	performance = np.mean(data_rows, axis=0).round(3)
	# workerId = np.mean(data_rows, axis=0).argmax()
	return performance


def get_averaged_results(accs_lines, fl_epochs):
	data_rows = []
	experiments = []
	epoch_count = 0
	for line in accs_lines:
		line = ast.literal_eval(line)
		data_row = [ast.literal_eval(acc[:-1]) for acc in line]
		data_rows.append(data_row)

		epoch_count += 1
		if epoch_count == fl_epochs + 1:
			# read extra row to skip
			data_rows = data_rows[:-1]

			experiments.append(np.array(data_rows))
			data_rows = []
			epoch_count = 0

	experiments = np.asarray(experiments)
	average_data = np.mean(experiments, axis=0)
	average_data /= 100.0  # make it percentage
	return average_data


def get_averaged_results_by_workerId(zipped_accs_lines, fl_epochs, workerId):
	data_rows = []
	epoch_count = 0
	experiments = []
	for lines in zipped_accs_lines:
		data_row = [ast.literal_eval(ast.literal_eval(
			line)[workerId][:-1]) for line in lines]
		data_rows.append(data_row)

		epoch_count += 1
		if epoch_count == fl_epochs + 1:
			# skip one last row from performance analysis
			experiments.append(np.array(data_rows[:-1]))
			data_rows = []
			epoch_count = 0
	return np.asarray(experiments).mean(axis=0) / 100.0


def get_best_cffl_worker(dirname, folder):
	"""
	Arguments: dirname: directory the folder is in, to find the folder
					folder: the folder that contains the experiment

	Returns: the best cffl worker among the n_runs of the experiments, determined through the average test accs
					<optional> performance_dict: averaged performance over n_runs, for all workers, for three types of models
	"""

	# need to figure out which is the best worker
	fl_epochs = int(folder.split('-')[1])
	n_runs = int(folder[folder.find('runs') - 1])

	with open(os.path.join(dirname, folder, 'log'), 'r') as log:
		loginfo = log.readlines()

	worker_cffl_accs_lines = [line.replace('Workers CFFL      accuracies:  ', '')
							  for line in loginfo if 'Workers CFFL      accuracies' in line]
	worker_standalone_accs_lines = [line.replace(
		'Workers standlone accuracies:  ', '') for line in loginfo if 'Workers standlone accuracies:  ' in line]
	worker_dssgd_accs_lines = [line.replace('Workers DSSGD     accuracies:  ', '')
							   for line in loginfo if 'Workers DSSGD     accuracies:  ' in line]

	cffl_performances = get_performance(
		worker_cffl_accs_lines, n_runs, fl_epochs)
	dssgd_performances = get_performance(
		worker_dssgd_accs_lines, n_runs, fl_epochs)
	standalone_performances = get_performance(
		worker_standalone_accs_lines, n_runs, fl_epochs)

	workerId = cffl_performances.argmax()
	performance_dict = {'cffl': cffl_performances,
						'dssgd': dssgd_performances, 'standalone': standalone_performances}

	return workerId, performance_dict


'''
