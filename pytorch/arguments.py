'''
args = {
	# system parameters
	'device': torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu"),
	# setting parameters
	'dataset': 'adult',
	'sample_size_cap': 5000,
	'n_workers': 5,
	'split': 'powerlaw', #classimbalance
	'theta': 0.1,  # privacy level -> at most (theta * num_of_parameters) updates
	'batch_size' : 16, # use this batch_size
	'train_val_split_ratio': 0.9,

	# model parameters
	'model_fn': LogisticRegression,
	# 'model': LogisticRegression(),
	'optimizer_fn': optim.SGD,
	'loss_fn': nn.CrossEntropyLoss(), # CrossEntropyLoss(), NLLLoss()
	'lr': 0.0001, # use this lr

	# training parameters
	'pretrain_epochs': 5,
	'fl_epochs': 100,
	'fl_individual_epochs': 5,
	'aggregate_mode':'sum',  # 'mean', 'sum'
}
'''
import torch
from torch import nn, optim

from utils.models import LogisticRegression, MLP_LogReg, MLP_Net, CNN_Net, RNN





use_cuda = True
adult_args = {
	# system parameters
	'device': torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu"),
	# setting parameters
	'dataset': 'adult',
	'sample_size_cap': 5000,
	'n_workers': 5,
	'split': 'powerlaw', #classimbalance
	'theta': 0.1,  # privacy level -> at most (theta * num_of_parameters) updates
	'batch_size' : 16, # use this batch_size
	'train_val_split_ratio': 0.9,
	'alpha': 3,

	# model parameters
	'model_fn': LogisticRegression,
	'optimizer_fn': optim.SGD,
	'loss_fn': nn.CrossEntropyLoss(), 
	'lr': 0.001, # use this lr

	# training parameters
	'pretrain_epochs': 5,
	'fl_epochs': 100,
	'fl_individual_epochs': 5,
	'aggregate_mode':'credit-sum',  # 'mean', 'sum', 'credit-sum'

}

mnist_args = {
	
	# system parameters
	'device': torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu"),
	# setting parameters
	'dataset': 'mnist',
	'sample_size_cap': 5000,
	'n_workers': 5,
	'split': 'classimbalance', #or 'powerlaw'
	'theta': 0.1,  # privacy level -> at most (theta * num_of_parameters) updates
	'batch_size' : 10, 
	'train_val_split_ratio': 0.9,
	'alpha': 1,

	# model parameters
	'model_fn': MLP_Net,
	'optimizer_fn': optim.SGD,
	'loss_fn': nn.NLLLoss(), 
	'lr': 0.1,

	# training parameters
	'pretrain_epochs': 5,
	'fl_epochs': 100,
	'fl_individual_epochs': 5,
	'aggregate_mode':'credit-sum',  # 'mean', 'sum'
}

names_args = {
	# system parameters
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