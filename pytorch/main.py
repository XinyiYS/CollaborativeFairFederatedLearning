import os
import sys
import json
import copy
import time
import datetime
import random
from itertools import product

import numpy as np
import torch
from torch import nn, optim

from utils.Data_Prepper import Data_Prepper
from utils.Federated_Learner import Federated_Learner
from examine_results import examine

from torch.multiprocessing import Pool, Process, set_start_method

def init_mp():
	try:
		 set_start_method('spawn',force=True)
	except RuntimeError as e:
		print('Setting mp start method problem:', str(e))
		pass

def write_aggregate_dict(performance_dicts, filename):

	keys = ['standalone_best_participant', 'CFFL_best_participant', 'rr_dssgd_best', 'rr_fedavg_best',
		'standalone_vs_rrdssgd', 'standalone_vs_final', 'standalone_vs_fedavg']

	aggregate_dict = {}
	for key in keys:
		list_of_performance = [performance_dict[key] for performance_dict in performance_dicts]
		aggregate_dict[key] = np.array(list_of_performance).tolist()
		aggregate_dict[key +'_mean'] = np.mean(aggregate_dict[key], axis=0).tolist()
		aggregate_dict[key +'_std'] = np.std(aggregate_dict[key], axis=0).tolist()

		'''
		print(key, aggregate_dict[key])
		print(key +'_mean', aggregate_dict[key +'_mean'])
			# result_list.append(res)
		print(key +'_std', aggregate_dict[key +'_std'])
		'''
	with open(filename, 'w') as file:
		file.write(json.dumps(aggregate_dict))
	return


def run_experiments(args, repeat=5, logs_dir='logs'):
	update_gpu(args)
	init_deterministic()

	# init steps
	model_name = str(args['model_fn']).split('.')[-1][:-2]
	subdir = "{}_p{}_e{}-{}-{}_b{}_size{}_lr{}_theta{}_{}runs_{}_a{}_fr{}_{}".format(args['dataset']+'@'+args['split'],args['n_participants'], 
							args['pretrain_epochs'], args['fl_epochs'], args['fl_individual_epochs'],
							args['batch_size'], args['sample_size_cap'], args['lr'], args['theta'],
							str(repeat), args['aggregate_mode'], args['alpha'],args['n_freeriders'], model_name,
							)
	logdir = os.path.join(logs_dir, subdir)
	os.makedirs(logdir, exist_ok=True)

	if 'complete.txt' in os.listdir(logdir):
		return

	with open(os.path.join(logdir,'settings_dict.txt'), 'w') as file:
		[file.write(key + ' : ' + str(value) + '\n') for key,value in args.items()]

	log = open(os.path.join(logdir, 'log'), "w")
	sys.stdout = log
	print("Experimental settings are: ", args, '\n')

	performance_dicts = []
	performance_dicts_pretrain = []
	

	# for the repeats of the experiment
	# only need to prepare the data once
	data_prep = Data_Prepper(args['dataset'], 
		train_batch_size=args['batch_size'], n_participants=args['n_participants'], sample_size_cap=args['sample_size_cap'], 
		train_val_split_ratio=args['train_val_split_ratio'], device=args['device'], args_dict=args)

	for i in range(repeat):
		print()
		print("Experiment : No.{}/{}".format(str(i+1) ,str(repeat)))
		# data_prep = Data_Prepper(args['dataset'], train_batch_size=args['batch_size'], sample_size_cap=args['sample_size_cap'], train_val_split_ratio=args['train_val_split_ratio'])
		federated_learner = Federated_Learner(args, data_prep)

		# train
		federated_learner.train()
		# analyze
		federated_learner.get_fairness_analysis()

		performance_dicts.append(federated_learner.performance_dict)
		
		with open(os.path.join(logdir, 'performance_dict.log'), 'a') as log:
			log.write(json.dumps(federated_learner.performance_dict))
			log.write('\n')

		performance_dicts_pretrain.append(federated_learner.performance_dict_pretrain)
		with open(os.path.join(logdir, 'performance_dict_pretrain.log'), 'a') as log:
			log.write(json.dumps(federated_learner.performance_dict_pretrain))
			log.write('\n')


	write_aggregate_dict(performance_dicts, os.path.join(logdir, 'aggregate_dict.txt'))
	write_aggregate_dict(performance_dicts_pretrain, os.path.join(logdir, 'aggregate_dict_pretrain.txt'))

	with open(os.path.join(logdir, 'complete.txt'), 'w') as file:
		file.write('complete')

	return

def get_parallel_groups(experiment_args, parallel_size=4):
	experiment_args = np.asarray(experiment_args)
	from math import ceil
	return np.array_split(experiment_args, ceil(len(experiment_args)/parallel_size))

def run_experiments_full(experiment_args, repeat=1):


	ts = time.time()
	st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H:%M')
	experiment_dir = 'Experiments_{}'.format(st)
	experiment_dir = os.path.join("{}".format(experiment_args[0]['dataset']), experiment_dir)

	os.makedirs(experiment_dir, exist_ok=True)
	

	for args in experiment_args:
		run_experiments(args, repeat, experiment_dir)

	try:
		examine(experiment_dir)
	except Exception as e:
		print(str(e))
	return

from utils.arguments import adult_args, mnist_args, names_args, update_gpu, cifar_cnn_args, mr_args, sst_args, imdb_args

def init_deterministic():
	# call init_deterministic() in each run_experiments function call

	torch.manual_seed(1234)
	np.random.seed(1234)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	random.seed(1234)

if __name__ == '__main__':
	# init steps	

	experiment_args = []	
	args = copy.deepcopy(mnist_args)
	for n_participants, sample_size_cap in [[5, 3000], [10, 6000], [20, 12000]]:
		args['n_participants'] = n_participants
		args['sample_size_cap'] = sample_size_cap
		for theta in [0.1, 1]:
			args['theta'] = theta

			if n_participants == 5:
				args['lr'] = 0.15
			else:
				args['lr'] = 0.25

			experiment_args.append(copy.deepcopy(args))
	run_experiments_full(experiment_args, repeat=1)

	experiment_args = []	
	args = copy.deepcopy(mnist_args)
	for n_participants, sample_size_cap in [[5, 3000], [10, 6000], [20, 12000]]:
		args['n_participants'] = n_participants
		args['sample_size_cap'] = sample_size_cap

		args['split'] = 'classimbalance'
		args['reputation_threshold_coef'] = 1 / 6.0
		for theta in [0.1, 1]:
			args['theta'] = theta

			args['lr'] = 0.15
			args['fl_individual_epochs'] = 1

			experiment_args.append(copy.deepcopy(args))
	run_experiments_full(experiment_args, repeat=1)



	experiment_args = []	
	args = copy.deepcopy(adult_args)
	for n_participants, sample_size_cap in [[5, 4000], [10, 8000], [20, 16000]]:
		args['n_participants'] = n_participants
		args['sample_size_cap'] = sample_size_cap
		for theta in [0.1, 1]:
			args['theta'] = theta

			experiment_args.append(copy.deepcopy(args))
	run_experiments_full(experiment_args, repeat=1)



	experiment_args = []	
	args = copy.deepcopy(adult_args)
	for n_participants, sample_size_cap in [[5, 4000], [10, 8000], [20, 16000]]:
		args['n_participants'] = n_participants
		args['sample_size_cap'] = sample_size_cap
		args['n_freeriders'] = 1
		args['reputation_threshold_coef'] = 2 / 3.0
		for theta in [0.1, 1]:
			args['theta'] = theta

			experiment_args.append(copy.deepcopy(args))
	run_experiments_full(experiment_args, repeat=1)