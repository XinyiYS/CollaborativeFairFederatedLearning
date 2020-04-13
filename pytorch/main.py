import sys
import torch

from torch import nn, optim

from utils.Worker import Worker
from utils.Data_Prepper import Data_Prepper
from utils.Federated_Learner import Federated_Learner
from utils.models import LogisticRegression, MLP_LogReg, MLP_Net, CNN_Net


use_cuda = True
args = {

	# system parameters
	'device': torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu"),


	# setting parameters
	'dataset': 'adult',
	'n_workers': 5,
	'split': 'power_law',
	'sharing_lambda': 0.1,  # privacy level -> at most (sharing_lambda * num_of_parameters) updates
	'batch_size' : 16,
	'train_val_split_ratio': 0.05,

	# model parameters
	'model_fn': LogisticRegression,
	'optimizer_fn': optim.SGD,
	'loss_fn': nn.CrossEntropyLoss(),
	'lr': 0.0001,

	# training parameters
	'pretrain_epochs': 1,
	'fl_epochs': 5,
	'fl_individual_epochs': 5,
}

if __name__ == '__main__':
	# init steps

	log = open("logs/experiment.log", "a")
	sys.stdout = log

	print("Experimental settings are: ", args)
	
	data_prep = Data_Prepper(args['dataset'], train_batch_size=args['batch_size'], train_val_split_ratio=args['train_val_split_ratio'])
	federated_learner = Federated_Learner(args, data_prep)


	# train
	federated_learner.train()
	# analyze
	federated_learner.get_fairness_analysis()
