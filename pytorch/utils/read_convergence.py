import os
import json
import pandas as pd
import ast
import numpy as np

import matplotlib.pyplot as plt

from collections import defaultdict


from .plot import plot


key_map = {'DSSGD_model_test_accs': 'DSSGD',
			'fedavg_model_test_accs' : 'Fedavg',
			'participant_standalone_test_accs': 'Standalone',
			'cffl_test_accs': 'CFFL',
			'reputations': 'reputations',
		   }


def parse(dirname, folder):
	setup = {}

	try:
		with open(os.path.join(dirname,folder, 'settings_dict.txt'), 'r') as settings:
			lines = settings.readlines()
		for line in lines:
			key, value = line.split(':', 1)
			setup[key.strip()] = value.strip()
	except Exception as e:
		print(str(e))
		print('Error reading from the settings_dict.txt, reading from the name of the folder')

	if setup:
		setup['model'] = setup['model_fn'].split('.')[-1][:-2]
		setup['P'] = int(setup['n_participants'])
		setup['E'] = int(setup['fl_individual_epochs'])
		setup['Communication Rounds'] = int(setup['fl_epochs'])
		setup['size'] = int(setup['sample_size_cap'])
		setup['B'] = int(setup['batch_size'])
		setup['lr'] = float(setup['lr'])
		setup['alpha'] = float(setup['alpha'])
		setup['theta'] = float(setup['theta'])
		setup['n_freeriders'] = int(setup['n_freeriders'])
		setup['pretrain_epochs'] = int(setup['pretrain_epochs'])
	else:
		param = folder.split('_')

		setup['dataset'] = 'adult'
		if 'MLP_Net' in folder or 'CNN_Net' in folder or 'classimbalance' in folder:
			setup['dataset'] = 'mnist'
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
	fedavg_accs = avg_accs['Fedavg']

	best_participant_ind = cffl_accs[-1].argmax()

	cffl_accs_pretrain = np.asarray(performance_dict_pretrain['cffl_test_accs']).mean(axis=0)
	cffl_accs_pretrain = cffl_accs_pretrain[:-1][:, 1:]


	return  [ fedavg_accs[-1][best_participant_ind], dssgd_accs[-1][best_participant_ind], standalone_accs[-1][best_participant_ind], cffl_accs[-1][best_participant_ind], cffl_accs_pretrain[-1][best_participant_ind] ]


def save_acc_dfs(dirname, folder, dfs):
	directory = os.path.join(dirname, folder, 'acc_dfs')
	try:
		os.mkdir(directory)
	except:
		pass
	[df.to_csv(os.path.join(directory, csv_name), index=False) for df, csv_name in zip(dfs, ['cffl.csv', 'standalone.csv', 'participant.csv'])]
	print('saving computed csvs to: ', directory)
	
	return


def get_performance_dicts(dirname, folder):
	logfiles = ['performance_dict.log', 'performance_dict_pretrain.log']
	performance_dicts = []
	for logfile in logfiles:
		with open(os.path.join(dirname, folder, logfile), 'r') as log:

			lines = log.readlines()

		performance_dict = {}
		loaded_temp_dicts = [json.loads(exp) for exp in lines]
		dict_keys = loaded_temp_dicts[0].keys()

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

		setup = parse(dirname, folder)
		n_participants = setup['P']
		columns = ['party' + str(i + 1) for i in range(n_participants)]

		n_freeriders = 0
		free_riders = []
		avg_dfs = {}
		for key in key_map:
			avg_accs = np.asarray(performance_dict[key]).mean(axis=0)
			if key != 'reputations':
				avg_accs = avg_accs[:-1]  # exclude the last repeated line

			n_freeriders = avg_accs.shape[1] - n_participants
			if n_freeriders > 0:
				free_riders  =  ['free' + str(i + 1) for i in range(n_freeriders)]


			avg_dfs[key_map[key]] = pd.DataFrame(data=avg_accs, columns=free_riders + columns)

		reputation_threshold = np.asarray(performance_dict['reputation_threshold']).mean(axis=0)
		reputation_threshold_pretrain = np.asarray(performance_dict_pretrain['reputation_threshold']).mean(axis=0)

		reputations_df = avg_dfs['reputations']
		reputations_df['threshold'] = reputation_threshold

		cffl_df = avg_dfs['CFFL']
		standalone_df = avg_dfs['Standalone']
		dssgd_df = avg_dfs['DSSGD']
		fedavg_df = avg_dfs['Fedavg']

		best_participant_ind = cffl_df.iloc[-1].argmax()

		reputations_avg_pretrain = np.nanmean(np.asarray(performance_dict_pretrain['reputations']), axis=0)

		reputations_df_pretrain = pd.DataFrame(data=reputations_avg_pretrain, columns = free_riders + columns)
		reputations_df_pretrain['threshold'] = reputation_threshold_pretrain


		cffl_avg_acc_pretrain = np.asarray(performance_dict_pretrain['cffl_test_accs']).mean(axis=0)[:-1]
		cffl_df_pretrain = pd.DataFrame(data=cffl_avg_acc_pretrain, columns=free_riders + columns)

		participant_df = pd.DataFrame(data={'Standlone': standalone_df.iloc[:, best_participant_ind],
									   'DSSGD': dssgd_df.iloc[:, best_participant_ind],
									   'FedAvg':fedavg_df.iloc[:, best_participant_ind],
									   'CFFL (w pretrain)': cffl_df_pretrain.iloc[:, best_participant_ind],
									   'CFFL (w/o pretrain)': cffl_df.iloc[:, best_participant_ind],
									   })

		reputations_figure_dir = os.path.join(dirname, folder, 'reputations.png')
		reputations_pretrain_figure_dir = os.path.join(dirname, folder, 'reputations_pretrain.png')
		cffl_figure_dir = os.path.join(dirname, folder, 'figure.png')
		cffl_pretrain_figure_dir = os.path.join(dirname, folder, 'figure_pretrain.png')

		standlone_figure_dir = os.path.join(dirname, folder, 'standlone.png')
		participant_figure_dir = os.path.join(dirname, folder, 'convergence_for_one.png')


		if os.path.exists(cffl_figure_dir):
			os.remove(cffl_figure_dir)

		plot(cffl_df, cffl_figure_dir, name=setup['dataset'], plot_type=0, split=setup['split'])
		plot(cffl_df_pretrain, cffl_pretrain_figure_dir, name=setup['dataset'], plot_type=0, split=setup['split'])



		if os.path.exists(reputations_figure_dir):
			os.remove(reputations_figure_dir)

		reputation_top = 1. / n_participants * 1.5
		reputation_bottom = -0.01

		plot(reputations_df, reputations_figure_dir, name=setup['dataset'].capitalize(), plot_type=0, ylabel='Reputations', top=reputation_top, bottom=reputation_bottom)
		plot(reputations_df_pretrain, reputations_pretrain_figure_dir, name=setup['dataset'].capitalize() + ' pretrain', plot_type=0, ylabel='Reputations',top=reputation_top, bottom=reputation_bottom)

		# plot(reputations_df, reputations_figure_dir, name=setup['dataset'], plot_type=0)

		if os.path.exists(standlone_figure_dir):
			os.remove(standlone_figure_dir)
		plot(standalone_df, standlone_figure_dir, name=setup['dataset'], plot_type=1)

		if os.path.exists(participant_figure_dir):
			os.remove(participant_figure_dir)
		plot(participant_df, participant_figure_dir, name=setup['dataset'], plot_type=2)

	return
