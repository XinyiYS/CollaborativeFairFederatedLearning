import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


class Data_Prepper:
	def __init__(self, name, train_batch_size, test_batch_size=1000, valid_batch_size=None, train_val_split_ratio=0.8,):
		self.name = name
		self.train_dataset, self.test_dataset = self.prepare_dataset(name)
		self.train_val_split_ratio = train_val_split_ratio

		self.init_batch_size(train_batch_size, test_batch_size, valid_batch_size)

		self.init_train_valid_idx()
		self.init_valid_loader()
		self.init_test_loader()

	def init_batch_size(self, train_batch_size, test_batch_size, valid_batch_size):
		self.train_batch_size = train_batch_size
		self.test_batch_size = test_batch_size
		self.valid_batch_size = valid_batch_size if valid_batch_size else test_batch_size

	def init_train_valid_idx(self, shuffle=True):
		self.train_idx, self.valid_idx = self.get_train_valid_indices(self.train_dataset, self.train_val_split_ratio, shuffle=shuffle)


	def init_valid_loader(self):
		self.valid_loader = DataLoader(self.train_dataset, batch_size=self.valid_batch_size, sampler=SubsetRandomSampler(self.valid_idx))

	def init_test_loader(self):
		self.test_loader = DataLoader(self.test_dataset, batch_size=self.test_batch_size)

	def get_valid_loader(self):
		return self.valid_loader

	def get_test_loader(self):
		return self.test_loader

	def get_train_valid_indices(self, train_dataset, train_val_split_ratio, shuffle=True):
		train_val_split_index = int(len(train_dataset) * train_val_split_ratio)
		indices = list(range(len(train_dataset)))
		if shuffle:
			np.random.seed(1111)
			np.random.shuffle(indices)

		return indices[train_val_split_index:], indices[:train_val_split_index]

	def get_train_loaders(self, n_workers, balanced=True, batch_size=None):
		if not batch_size:
			batch_size = self.train_batch_size
		from utils.utils import random_split
		indices_list = random_split(sample_indices=self.train_idx, m_bins=n_workers, equal=balanced)
		worker_train_loaders = [DataLoader(self.train_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(indices)) for indices in indices_list]
		return worker_train_loaders


	def prepare_dataset(self, name='adult'):
	    if name == 'adult':
	        from utils.load_adult import get_train_test
	        from utils.Custom_Dataset import Custom_Dataset
	        import torch

	        train_data, train_target, test_data, test_target = get_train_test()

	        X_train = torch.tensor(train_data.values, requires_grad=False).float()
	        y_train = torch.tensor(train_target.values, requires_grad=False).long()
	        X_test = torch.tensor(test_data.values, requires_grad=False).float()
	        y_test = torch.tensor(test_target.values, requires_grad=False).long()

	        train_set = Custom_Dataset(X_train, y_train)
	        test_set = Custom_Dataset(X_test, y_test)

	        return train_set, test_set
	    elif name == 'mnist':
	        from torchvision import datasets, transforms

	        train = datasets.MNIST('datasets/', train=True, transform=transforms.Compose([
	               transforms.Pad((2,2,2,2)),
	               transforms.ToTensor(),
	               transforms.Normalize((0.1307,), (0.3081,))
	                           ]))
	            
	        test = datasets.MNIST('datasets/', train=False, transform=transforms.Compose([
	                transforms.Pad((2,2,2,2)),
	                transforms.ToTensor(),
	                transforms.Normalize((0.1307,), (0.3081,))
	            ]))
	        return train, test