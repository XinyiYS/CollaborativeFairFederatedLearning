import os
import sys
import json
from itertools import product

import numpy as np
import torch
from torch import nn, optim

from utils.Worker import Worker
from utils.Data_Prepper import Data_Prepper
from utils.Federated_Learner import Federated_Learner
# from utils.models import LogisticRegression, MLP_LogReg, MLP_Net, CNN_Net


def run_experiments(args, repeat=5):
	# init steps
	logs_dir = 'logs'
	model_name = str(args['model_fn']).split('.')[-1][:-2]
	subdir = "{}_p{}_e{}-{}-{}_b{}_size{}_lr{}_theta{}_{}runs_{}_a{}_{}".format(args['dataset']+'@'+args['split'],args['n_workers'], 
							args['pretrain_epochs'], args['fl_epochs'], args['fl_individual_epochs'],
							args['batch_size'], args['sample_size_cap'], args['lr'], args['theta'],
							str(repeat), args['aggregate_mode'], args['alpha'],model_name,
							)
	logdir = os.path.join(logs_dir, subdir)


	try:
		os.mkdir(logdir)
	except:
		pass
	
	if 'complete.txt' in os.listdir(logdir):
		return

	with open(os.path.join(logdir,'settings_dict.txt'), 'w') as file:
		[file.write(key + ' : ' + str(value) + '\n') for key,value in args.items()]

	log = open(os.path.join(logdir, 'log'), "w")
	sys.stdout = log
	print("Experimental settings are: ", args, '\n')

	
	performance_dicts = []
	for i in range(repeat):
		print("Experiment : No.{}/{}".format(str(i+1) ,str(repeat)))
		data_prep = Data_Prepper(args['dataset'], train_batch_size=args['batch_size'], sample_size_cap=args['sample_size_cap'], train_val_split_ratio=args['train_val_split_ratio'])
		federated_learner = Federated_Learner(args, data_prep)

		# train
		federated_learner.train()
		# analyze
		federated_learner.get_fairness_analysis()

		performance_dicts.append(federated_learner.performance_dict)
	
	keys = ['standalone_vs_final', 'standlone_vs_rrdssgd', 'sharingcontribution_vs_final',
			'rr_dssgd_avg', 'CFFL_best_worker', 'standalone_best_worker',
			# 'sharingcontribution_vs_improvements'
			 ]
	
	aggregate_dict = {}
	for key in keys:
		list_of_performance = [performance_dict[key] for performance_dict in performance_dicts]
		aggregate_dict[key] = np.array(list_of_performance).tolist()
		aggregate_dict[key +'_mean'] = np.mean(aggregate_dict[key], axis=0).tolist()
		aggregate_dict[key +'_std'] = np.std(aggregate_dict[key], axis=0).tolist()

		print(key, aggregate_dict[key])
		print(key +'_mean', aggregate_dict[key +'_mean'])
		print(key +'_std', aggregate_dict[key +'_std'])

	with open(os.path.join(logdir, 'aggregate_dict.txt'), 'w') as file:
		file.write(json.dumps(aggregate_dict))

	with open(os.path.join(logdir, 'complete.txt'), 'w') as file:
		file.write('complete')
	return



from arguments import adult_args, mnist_args, names_args

if __name__ == '__main__':
	# # init steps
	# for n_workers, sample_size_cap, fl_epochs in [[5, 5000, 100],[10, 10000, 100],[20, 15000, 100]]:

	# args = adult_args # mnist_args

	# n_workers, sample_size_cap, fl_epochs = [5, 3000, 20]

	# for n_workers, sample_size_cap, fl_epochs in [ [5, 5000, 100], [10, 10000, 100]]:
	args = adult_args

	args['pretrain_epochs'] = 10
	args['theta'] = 1
	for n_workers, sample_size_cap in [  [5, 5000], [10, 10000], [20, 15000]]:
		args['n_workers'] = n_workers
		args['sample_size_cap'] = sample_size_cap
		run_experiments(args, 5)



