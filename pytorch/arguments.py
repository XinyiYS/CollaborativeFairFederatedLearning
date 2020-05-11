import torch
from torch import nn, optim

from utils.models import LogisticRegression, MLP, MLP_Net, CNN_Net, RNN, CNN_Text

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
	'split': 'powerlaw',
	'theta': 0.1,  # privacy level -> at most (theta * num_of_parameters) updates
	'batch_size': 16,
	'train_val_split_ratio': 0.9,
	'alpha': 5,
	'epoch_sample_size':float("Inf"),
	'n_freeriders': 0,

	# model parameters
	'model_fn': MLP,
	'optimizer_fn': optim.SGD,
	'loss_fn': nn.NLLLoss(),  #CrossEntropyLoss NLLLoss
	'pretraining_lr' : 0.1,  # only used during pretraining for CFFL models, no decay
	'dssgd_lr': 0.001, # used for dssgd model, no decay
	'lr': 0.1, # initial lr, with decay
	'grad_clip': 0.01,
	'gamma':0.977,   #0.97**100 ~= 0.1

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
	'split': 'classimbalance', #or 'classimbalance'
	'theta': 0.1,  # privacy level -> at most (theta * num_of_parameters) updates
	'batch_size' : 10, 
	'train_val_split_ratio': 0.9,
	'alpha': 5,
	'epoch_sample_size':float("Inf"),
	'n_freeriders': 0,


	# model parameters
	'model_fn': MLP_Net,
	'optimizer_fn': optim.SGD,
	'loss_fn': nn.NLLLoss(), 
	'pretraining_lr' : 0.1, # only used during pretraining for CFFL models, no decay
	'dssgd_lr': 0.001, # used for dssgd model, no decay
	'lr': 0.001,
	'grad_clip':0.001,
	'gamma':0.955,   #0.955**100 ~= 0.01

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
	'n_freeriders': 0,


	# model parameters
	'model_fn': RNN,
	'optimizer_fn': optim.SGD,
	'loss_fn': nn.NLLLoss(), 
	'pretraining_lr' : 0.1, # only used during pretraining for CFFL models, no decay
	'dssgd_lr': 0.001, # used for dssgd model, no decay
	'lr': 0.005,
	'grad_clip':0.001,
	'gamma':0.955,   #0.955**100 ~= 0.01

	# training parameters
	'pretrain_epochs': 0,
	'fl_epochs': 100,
	'fl_individual_epochs': 5,
	'aggregate_mode':'sum',  # 'mean', 'sum', credit-sum
}


sst_args = {
	# system parameters
	'gpu': 0,
	'device': torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu"),
	'save_gpu': True,
	# setting parameters
	'dataset': 'sst',
	'sample_size_cap': 5000,
	'n_workers': 5,
	'split': 'powerlaw', #or 'powerlaw' classimbalance
	'theta': 0.1,  # privacy level -> at most (theta * num_of_parameters) updates
	'batch_size' : 1, 
	'train_val_split_ratio': 0.9,
	'alpha': 5,
	'epoch_sample_size':10,
	'n_freeriders': 0,


	# model parameters
	'model_fn': CNN_Text,
	'embed_num': 20000,
	'embed_dim': 300,
	'class_num': 5,
	'kernel_num': 128,
	'kernel_sizes': [3,4,5],

	'optimizer_fn': optim.Adam,
	'loss_fn': nn.NLLLoss(), 
	'pretraining_lr' : 0.1, # only used during pretraining for CFFL models, no decay
	'dssgd_lr': 0.001, # used for dssgd model, no decay
	'lr': 0.001,
	'grad_clip':0.001,
	'gamma':0.955,   #0.955**100 ~= 0.01

	# training parameters
	'pretrain_epochs': 0,
	'fl_epochs': 100,
	'fl_individual_epochs': 5,
	'aggregate_mode':'sum',  # 'mean', 'sum', credit-sum
}

