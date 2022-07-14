import os
import shutil
import json
import pandas as pd
import ast
import numpy as np

from utils.read_convergence import plot_convergence, parse, get_cffl_best

fairness_keys = [
		'standalone_vs_fedavg_mean',
		'standalone_vs_rrdssgd_mean',
		'standalone_vs_final_mean',
		]

def collect_and_compile_performance(dirname):

	fairness_rows = []
	performance_rows = []
	for folder in os.listdir(dirname):
		if os.path.isfile(os.path.join(dirname, folder)) or not 'complete.txt' in os.listdir(os.path.join(dirname, folder)):
			continue

		setup = parse(dirname, folder)
		n_participants = setup['P']
		fl_epochs = setup['Communication Rounds']
		theta = setup['theta']

		try:
			with open(os.path.join(dirname, folder, 'aggregate_dict.txt')) as dict_log:
				aggregate_dict = json.loads(dict_log.read())

			with open(os.path.join(dirname, folder, 'aggregate_dict_pretrain.txt')) as dict_log:
				aggregate_dict_pretrain = json.loads(dict_log.read())

			f_data_row = ['P' + str(n_participants) + '_' + str(theta)] + [aggregate_dict[f_key][0] for f_key in fairness_keys]
			f_data_row.append(aggregate_dict_pretrain['standalone_vs_final_mean'][0])

			p_data_row = ['P' + str(n_participants) + '_' + str(theta)] + [aggregate_dict['rr_fedavg_best'][0],
																	aggregate_dict['rr_dssgd_best'][0], 
																	aggregate_dict['standalone_best_participant'][0], 
																	aggregate_dict['CFFL_best_participant'][0], 
																	aggregate_dict_pretrain['CFFL_best_participant'][0]
																	]

			fairness_rows.append(f_data_row)
			performance_rows.append(p_data_row)
		except Exception as e:
			print("Compiling fairness and accuracy csvs")
			print(e)

	shorthand_f_keys = ['Fedavg', 'DSSGD', 'CFFL', 'CFFL pretrain']
	fair_df = pd.DataFrame(fairness_rows, columns=[' '] + shorthand_f_keys).set_index(' ')
	fair_df = fair_df.sort_values(' ')
	print(fair_df.to_markdown())
	
	print(os.path.join(dirname, 'fairness.csv'))
	fair_df.to_csv( os.path.join(dirname, 'fairness.csv'))

	shorthand_p_keys = ['Fedavg', 'DSSGD', 'Standalone', 'CFFL', 'CFFL pretrain']
	pd.options.display.float_format = '{:,.2f}'.format
	perf_df = pd.DataFrame(performance_rows, columns=[' '] + shorthand_p_keys).set_index(' ').T
	perf_df = perf_df[sorted(perf_df.columns)]
	print(perf_df.to_markdown())
	perf_df.to_csv( os.path.join(dirname, 'performance.csv'))

	return fair_df, perf_df


def collate_pngs(dirname):
	os.makedirs(os.path.join(dirname, 'figures'), exist_ok=True)
	figures_dir = os.path.join(dirname, 'figures')
	
	for directory in os.listdir(dirname):
		if os.path.isfile(os.path.join(dirname, directory)) or not 'complete.txt' in os.listdir(os.path.join(dirname, directory)):
			continue

		setup = parse(dirname, directory)

		subdir = os.path.join(dirname, directory)

		figure_name = '{}_{}_p{}e{}_cffl_localepoch{}_localbatch{}_lr{}_upload{}_pretrain0.png'.format(
			setup['dataset'],  setup['model'],
			setup['P'], setup['Communication Rounds'],
			setup['E'], setup['B'],
			str(setup['lr']).replace('.', ''),
			str(setup['theta']).replace('.', '').rstrip('0'))
		pastfig_name = figure_name.replace('_pretrain0','')
		if os.path.exists(os.path.join(figures_dir, pastfig_name)):
			os.remove(os.path.join(figures_dir, pastfig_name))
		shutil.copy(os.path.join(subdir,'figure.png'),  os.path.join(figures_dir, figure_name) )
		shutil.copy(os.path.join(subdir,'figure_pretrain.png'),  os.path.join(figures_dir, figure_name.replace('pretrain0','pretrain1')) )

		standalone_name = '{}_{}_p{}e{}_standalone.png'.format(
			setup['dataset'],  setup['model'],
			setup['P'], setup['Communication Rounds'])
		shutil.copy(os.path.join(subdir,'standlone.png'),   os.path.join(figures_dir, standalone_name) )

		convergence_name = '{}_{}_p{}e{}_upload{}_convergence.png'.format(
			setup['dataset'], setup['model'],
			setup['P'], setup['Communication Rounds'],
			str(setup['theta']).replace('.', '').rstrip('0'))
		shutil.copy(os.path.join(subdir,'convergence_for_one.png'),   os.path.join(figures_dir, convergence_name) )

	return


def examine(dirname):
	experiment_results = plot_convergence(dirname)
	collate_pngs(dirname)
	fair_df, perf_df = collect_and_compile_performance(dirname)

if __name__ == '__main__':
		
	"""
	Give the directory to the experiment to dirname
	"""
	dirname = 'cifar10/Experiments_2020-08-06-01:21'
	examine(dirname)
