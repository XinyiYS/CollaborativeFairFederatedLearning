import os
import sys
import json

import numpy as np
import torch

from torch import nn, optim

from utils.Worker import Worker
from utils.Data_Prepper import Data_Prepper
from utils.Federated_Learner import Federated_Learner


def run_experiments(args, repeat=1):
	# init steps
	print("Experimental settings are: ", args)
	
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

		# print(key, aggregate_dict[key])
		print(key +'_mean', aggregate_dict[key +'_mean'])
		# print(key +'_std', aggregate_dict[key +'_std'])
	return


from arguments import adult_args, mnist_args, names_args


if __name__ == '__main__':
	# init steps
	# args = mnist_args
	args = adult_args
	n_workers, sample_size_cap, fl_epochs = [10, 5000, 100]
	theta = 0.1
	args['n_workers'] = n_workers
	args['sample_size_cap'] = sample_size_cap
	args['fl_epochs'] = fl_epochs
	args['theta'] = theta

	run_experiments(args, 1)



'''
lr 0.01
grad 1
alpha 9
5, 5000
sum
theta = 0.1
MLP
COMPLETE 2runs
0.7099053032066631


standlone_vs_rrdssgd  -  [0.2423363740302561]             
standalone_vs_final  -  [0.9004758650358347]              
sharingcontribution_vs_final  -  [0.5165547977805267]     
sharingcontribution_vs_improvements  -  [-0.99839354629301
02]                                                       
standalone_best_worker  -  0.802854597568512              
CFFL_best_worker  -  0.8048617243766785                   
rr_dssgd_avg  -  0.8086529612541199                       
----                                                      
standalone_vs_final_mean [0.8324694986800758]             
standlone_vs_rrdssgd_mean [0.07644799475104855]          
sharingcontribution_vs_final_mean [0.5856037622584093]    
rr_dssgd_avg_mean 0.8084121084213258                      
CFFL_best_worker_mean 0.804014241695404                   
standalone_best_worker_mean 0.8015164852142334 
COMPLETE 5runs


lr 0.01
grad 1
alpha 9
10, 10000
TESTING console right



lr 0.01
grad 1
alpha 9
20, 15000
TESTING console right


Experimental settings are:  {'device': device(type='cuda'), 'dataset': 'adult', 
'sample_size_cap': 15000, 
'n_workers': 20, 'split': 'powerlaw', 'theta': 0.1, 'batch_size': 16, 'train_val_split_ratio': 0.9, 
'alpha': 9, 'epoch_sample_size': inf, 
'model_fn': <class 'utils.models.MLP'>, 'optimizer_fn': <class 'torch.optim.sgd.SGD'>, 'loss_fn': NLLLoss(), 
'lr': 0.001, 'grad_clip': 1, 'pretrain_epochs': 5, 'fl_epochs': 100, 'fl_individual_epochs': 5, 
'aggregate_mode': 'sum'}
NOT WORKING


Experimental settings are:  {'device': device(type='cuda'), 'dataset': 'adult', 
'sample_size_cap': 15000, 
'n_workers': 20, 'split': 'powerlaw', 'theta': 0.1, 'batch_size': 16, 'train_val_split_ratio': 0.9, 
'alpha': 5, 'epoch_sample_size': inf, 
'model_fn': <class 'utils.models.MLP'>, 'optimizer_fn': <class 'torch.optim.sgd.SGD'>, 'loss_fn': NLLLoss(),
'lr': 0.001, 'grad_clip': 0.01, 'pretrain_epochs': 5, 'fl_epochs': 100, 'fl_individual_epochs': 5, 
'aggregate_mode': 'sum'}
NOT WORKING



lr 0.001
grad 1
alpha 9
10, 10000
NOT WORKING
TESTING console right


lr 0.001
grad 0.01
alpha 5
5, 5000
NOW WORKING
0.28317842597407206


lr 0.001
grad 1
alpha 5
5, 5000
sum
theta = 0.1
MLP
COMPLETE
standlone_vs_rrdssgd  -  [-0.556325662779587]             │
standalone_vs_final  -  [0.9726401078779605]              │
sharingcontribution_vs_final  -  [0.7207063092866214]     │
sharingcontribution_vs_improvements  -  [-0.99007228006438│
09]                                                       │
standalone_best_worker  -  0.8010704517364502             │
CFFL_best_worker  -  0.8033006191253662                   │
rr_dssgd_avg  -  0.8099464535713196                       │
----                                                      │
standalone_vs_final_mean [0.7320808477684464]             │
standlone_vs_rrdssgd_mean [-0.026425673064771726]         │
sharingcontribution_vs_final_mean [0.4925209437343247]    │
rr_dssgd_avg_mean 0.8062444031238556                      │
CFFL_best_worker_mean 0.8000668883323669                  │
standalone_best_worker_mean 0.7971676886081696    













lr 0.001
grad 1
alpha 5
10, 10000
sum
theta = 0.1
MLP
TESTING console right

0.11256940307775254
NOT WORKING


Experimental settings are:  {'device': device(type='cuda'), 'dataset': 'adult', 
'sample_size_cap': 10000, 'n_workers': 10, 'split': 'powerlaw', 'theta': 0.1, 'batch_size': 16, 'train_val_split_ratio': 0.9, 
'alpha': 5, 'epoch_sample_size': inf, 
'model_fn': <class 'utils.models.MLP'>, 'optimizer_fn': <class 'torch.optim.sgd.SGD'>, 'loss_fn': NLLLoss(), 
'lr': 0.01, 'grad_clip': 1, 'pretrain_epochs': 5, 'fl_epochs': 100, 'fl_individual_epochs': 5, 
'aggregate_mode': 'sum'}
NOT WORKING


Experimental settings are:  {'device': device(type='cuda'), 'dataset': 'adult', 
'sample_size_cap': 5000, 'n_workers': 5, 'split': 'powerlaw', 'theta': 0.1, 'batch_size': 16, 'train_val_split_ratio': 0.9, 
'alpha': 5, 'epoch_sample_size': inf,
'model_fn': <class 'utils.models.MLP'>, 'optimizer_fn': <class 'torch.optim.sgd.SGD'>, 'loss_fn': NLLLoss(), 
'lr': 0.001, 'grad_clip': 0.005, 'pretrain_epochs': 5, 'fl_epochs': 100, 'fl_individual_epochs': 5, 
'aggregate_mode': 'sum'}
NOW WORKING

Experimental settings are:  {'device': device(type='cuda'), 'dataset': 'adult', 
'sample_size_cap': 5000, 'n_workers': 5, 'split': 'powerlaw', 'theta': 0.1, 'batch_size': 16, 'train_val_split_ratio': 0.9, 
'alpha': 5, 'epoch_sample_size': inf, 
'model_fn': <class 'utils.models.MLP'>, 'optimizer_fn': <class 'torch.optim.sgd.SGD'>, 'loss_fn': NLLLoss(), 
'lr': 0.001, 'grad_clip': 10, 'pretrain_epochs': 5, 'fl_epochs': 100, 'fl_individual_epochs': 5, 'aggregate_mode': 'sum'}
NOW WORKING

Experimental settings are:  {'device': device(type='cuda'), 'dataset': 'adult', 
'sample_size_cap': 2500, 'n_workers': 5, 'split': 'powerlaw', 'theta': 0.1, 'batch_size': 16, 'train_val_split_ratio': 0.9, 
'alpha': 5, 'epoch_sample_size': inf, 
'model_fn': <class 'utils.models.MLP'>, 'optimizer_fn': <class 'torch.optim.sgd.SGD'>, 'loss_fn': NLLLoss(), 
'lr': 0.001, 'grad_clip': 0.0001, 'pretrain_epochs': 5, 'fl_epochs': 100, 'fl_individual_epochs': 5, 'aggregate_mode': 'sum'}
COMPLETE
fairness works but not converged


Experimental settings are:  {'device': device(type='cuda'), 'dataset': 'adult', 
'sample_size_cap': 5000, 'n_workers': 5, 'split': 'powerlaw', 'theta': 0.1, 'batch_size': 16, 'train_val_split_ratio': 0.9, 
'alpha': 5, 'epoch_sample_size': inf, 
'model_fn': <class 'utils.models.MLP'>, 'optimizer_fn': <class 'torch.optim.sgd.SGD'>, 'loss_fn': NLLLoss(), 
'lr': 0.001, 'grad_clip': 0.0001, 'pretrain_epochs': 5, 'fl_epochs': 100, 'fl_individual_epochs': 5, 
'aggregate_mode': 'sum'}
COMPLETE
fairness works but not converged

'''