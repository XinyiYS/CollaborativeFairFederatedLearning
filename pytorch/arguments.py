import torch
from torch import nn, optim

from utils.models import LogisticRegression, MLP, MLP_Net, CNN_Net, RNN

use_cuda = True
cuda_available = torch.cuda.is_available()


def update_gpu(args):
	if 'cuda' in str(args['device']):
		args['device'] = torch.device('cuda:{}'.format(args['gpu']))

adult_args = {
	# system parameters
	'gpu': 0,
	'device': torch.device("cuda" if cuda_available and use_cuda else "cpu"),
	# setting parameters
	'dataset': 'adult',
	'sample_size_cap': 5000,
	'n_workers': 10,
	'split': 'powerlaw', #classimbalance
	'theta': 0.1,  # privacy level -> at most (theta * num_of_parameters) updates
	'batch_size': 10, # use this batch_size
	'train_val_split_ratio': 0.9,
	'alpha': 5,
	'epoch_sample_size':float("Inf"),

	# model parameters
	'model_fn': MLP,
	'optimizer_fn': optim.SGD,
	'loss_fn': nn.NLLLoss(),  #CrossEntropyLoss NLLLoss
	'lr': 0.1,  # lr 0.01 works for B = 10
	'grad_clip': 0.01,

	# training parameters
	'pretrain_epochs': 5,
	'fl_epochs': 100,
	'fl_individual_epochs': 5,
	'aggregate_mode':'sum',  # 'mean', 'sum', 'credit-sum'
}

mnist_args = {
	# system parameters
	'gpu': 0,
	'device': torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu"),
	# setting parameters
	'dataset': 'mnist',
	'sample_size_cap': 3000,
	'n_workers': 5,
	'split': 'powerlaw', #or 'powerlaw'
	'theta': 0.1,  # privacy level -> at most (theta * num_of_parameters) updates
	'batch_size' : 10, 
	'train_val_split_ratio': 0.9,
	'alpha': 5,
	'epoch_sample_size':float("Inf"),


	# model parameters
	'model_fn': MLP_Net,
	'optimizer_fn': optim.SGD,
	'loss_fn': nn.NLLLoss(), 
	'lr': 0.001,
	'grad_clip':0.001,

	# training parameters
	'pretrain_epochs': 10,
	'fl_epochs': 100,
	'fl_individual_epochs': 5,
	'aggregate_mode':'sum',  # 'mean', 'sum'
}

names_args = {
	# system parameters
	'gpu': 0,
	'device': torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu"),
	# setting parameters
	'dataset': 'names',
	'sample_size_cap': 5000,
	'n_workers': 5,
	'split': 'powerlaw', #or 'powerlaw' classimbalance
	'theta': 0.1,  # privacy level -> at most (theta * num_of_parameters) updates
	'batch_size' : 1, 
	'train_val_split_ratio': 0.9,
	'alpha': 5,
	'epoch_sample_size':10,

	# model parameters
	'model_fn': RNN,
	'optimizer_fn': optim.SGD,
	'loss_fn': nn.NLLLoss(), 
	'lr': 0.005,

	# training parameters
	'pretrain_epochs': 0,
	'fl_epochs': 100,
	'fl_individual_epochs': 5,
	'aggregate_mode':'sum',  # 'mean', 'sum', credit-sum
}

