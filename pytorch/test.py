import os
import sys
import json
import random
import numpy as np
import torch

from torch import nn, optim

from utils.Data_Prepper import Data_Prepper
from utils.Federated_Learner import Federated_Learner


def run_experiments(args, repeat=5, logs_dir='logs'):
	update_gpu(args)
	init_deterministic()

	# init steps
	print("Experimental settings are: ", args)
	
	performance_dicts = []
	performance_dicts_pretrain = []

	data_prep = Data_Prepper(args['dataset'], 
		train_batch_size=args['batch_size'], n_participants=args['n_participants'], sample_size_cap=args['sample_size_cap'], 
		train_val_split_ratio=args['train_val_split_ratio'], device=args['device'], args_dict=args)

	for i in range(repeat):

		print("Experiment : No.{}/{}".format(str(i+1) ,str(repeat)))
		federated_learner = Federated_Learner(args, data_prep)

		# train
		federated_learner.train()

		# analyze
		federated_learner.get_fairness_analysis()

		performance_dicts.append(federated_learner.performance_dict)
		performance_dicts_pretrain.append(federated_learner.performance_dict_pretrain)


	keys = ['standalone_best_participant', 'CFFL_best_participant', 'rr_dssgd_best', 'rr_fedavg_best',
		'standalone_vs_rrdssgd', 'standalone_vs_final', 'standalone_vs_fedavg']

	print("Printing the statistic over {} repeat of experiments below.".format(repeat))

	print("for all without pretraining:")
	aggregate_dict = {}
	for key in keys:
		list_of_performance = [performance_dict[key] for performance_dict in performance_dicts]
		aggregate_dict[key] = np.array(list_of_performance).tolist()
		aggregate_dict[key +'_mean'] = np.mean(aggregate_dict[key], axis=0).tolist()
		aggregate_dict[key +'_std'] = np.std(aggregate_dict[key], axis=0).tolist()

		# print(key, aggregate_dict[key])
		print(key +'_mean', np.around(aggregate_dict[key +'_mean'], 3))
		# print(key +'_std', aggregate_dict[key +'_std'])
	
	print()
	print("for all with pretraining:")
	aggregate_dict = {}
	for key in keys:
		list_of_performance = [performance_dict[key] for performance_dict in performance_dicts_pretrain]
		aggregate_dict[key] = np.array(list_of_performance).tolist()
		aggregate_dict[key +'_mean'] = np.mean(aggregate_dict[key], axis=0).tolist()
		aggregate_dict[key +'_std'] = np.std(aggregate_dict[key], axis=0).tolist()

		# print(key, aggregate_dict[key])
		print(key +'_mean', np.around(aggregate_dict[key +'_mean'], 3) )
		# print(key +'_std', aggregate_dict[key +'_std'])

	return


from utils.arguments import adult_args, mnist_args, names_args, update_gpu, mr_args, sst_args, imdb_args, cifar_cnn_args

def init_deterministic():
	torch.manual_seed(1234)
	np.random.seed(1234)
	random.seed(1234)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

import copy
if __name__ == '__main__':
	# init steps

	args = mnist_args
	args['theta'] = 0.1
	args['n_participants'] = 10
	args['sample_size_cap'] = 6000
	# args = cifar_cnn_args
	args['batch_size'] = 16
	args['gamma'] = 0.977
	# args['optimizer_fn'] = optim.SGD
	args['lr'] = 0.25
	args['fl_epochs'] = 30
	# args['dssgd_lr'] = 5e-3
	# args['split']='powerlaw'
	args['grad_clip']= 0.01
	args['pretrain_epochs']= 2
	# args['pretraining_lr']=0.005
	# args['dssgd_lr']=0.005
	args['aggregate_mode']= 'mean'
	# args['n_freeriders'] = 1
	args['download'] = 'topk'
	run_experiments(args, 2)
	
	exit()