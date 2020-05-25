import os
import random
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from torchtext.data import Field, LabelField, BucketIterator

class Data_Prepper:
	def __init__(self, name, train_batch_size, n_workers, sample_size_cap=-1, test_batch_size=100, valid_batch_size=None, train_val_split_ratio=0.8, device=None):
		self.args = None
		self.name = name
		self.device = device
		self.n_workers = n_workers
		self.sample_size_cap = sample_size_cap
		self.train_val_split_ratio = train_val_split_ratio

		self.init_batch_size(train_batch_size, test_batch_size, valid_batch_size)

		if name in ['sst', 'mr', 'imdb']:
			parser = argparse.ArgumentParser(description='CNN text classificer')
			self.args = parser.parse_args()

			self.train_datasets, self.validation_dataset, self.test_dataset = self.prepare_dataset(name)

			self.valid_loader = BucketIterator(self.validation_dataset, batch_size = 500, sort_key=lambda x: len(x.text), device=self.device  )
			self.test_loader = BucketIterator(self.test_dataset, batch_size = 500, sort_key=lambda x: len(x.text), device=self.device)

			self.args.embed_num = len(self.args.text_field.vocab)
			self.args.class_num = len(self.args.label_field.vocab)
			
			self.args.embed_dim = 128
			self.args.kernel_num = 128
			self.args.kernel_sizes = [3,4,5]
			self.args.static = False
			print("Model embedding arguments:", self.args)


		else:
			self.train_dataset, self.validation_dataset, self.test_dataset = self.prepare_dataset(name)

			print('------')
			print("Train to split size: {}. Validation size: {}. Test size: {}".format(len(self.train_dataset), len(self.validation_dataset), len(self.test_dataset)))
			print('------')

			self.valid_loader = DataLoader(self.validation_dataset, batch_size=self.test_batch_size)
			self.test_loader = DataLoader(self.test_dataset, batch_size=self.test_batch_size)


	def init_batch_size(self, train_batch_size, test_batch_size, valid_batch_size):
		self.train_batch_size = train_batch_size
		self.test_batch_size = test_batch_size
		self.valid_batch_size = valid_batch_size if valid_batch_size else test_batch_size

	def get_valid_loader(self):
		return self.valid_loader

	def get_test_loader(self):
		return self.test_loader

	def get_train_loaders(self, n_workers, split='powerlaw', batch_size=None):
		if not batch_size:
			batch_size = self.train_batch_size

		if split == 'classimbalance':
			if self.name not in ['mnist','cifar10']:
				raise NotImplementedError("Calling on dataset {}. Only mnist and cifar10 are implemnted for this split".format(self.name))

			n_classes = 10			
			data_indices = [(self.train_dataset.targets == class_id).nonzero().view(-1).tolist() for class_id in range(n_classes)]
			class_sizes = np.linspace(1, n_classes, n_workers, dtype='int')
			party_mean = self.sample_size_cap // self.n_workers

			from collections import defaultdict
			party_indices = defaultdict(list)
			for party_id, class_sz in enumerate(class_sizes):	
				classes = range(class_sz) # can customize classes for each party rather than just listing
				each_class_id_size = party_mean // class_sz
				for i, class_id in enumerate(classes):
					selected_indices = data_indices[class_id][:each_class_id_size]
					data_indices[class_id] = data_indices[class_id][each_class_id_size:]
					party_indices[party_id].extend(selected_indices)

					# top up to make sure all parties have the same number of samples
					if i == len(classes) - 1 and len(party_indices[party_id]) < party_mean:
						extra_needed = party_mean - len(party_indices[party_id])
						party_indices[party_id].extend(data_indices[class_id][:extra_needed])
						data_indices[class_id] = data_indices[class_id][extra_needed:]

			indices_list = [party_index_list for party_id, party_index_list in party_indices.items()] 

		elif split == 'powerlaw':
			if self.name in ['sst', 'mr', 'imdb']:
				# sst, mr, imdb split is different from other datasets, so return here				

				self.train_loaders = [BucketIterator(train_dataset, batch_size=self.train_batch_size, device=self.device, sort_key=lambda x: len(x.text),train=True) for train_dataset in self.train_datasets]
				self.shard_sizes = [(len(train_dataset)) for train_dataset in self.train_datasets]
				return self.train_loaders

			else:
				indices_list = powerlaw(list(range(len(self.train_dataset))), n_workers)

		elif split in ['balanced','equal']:
			from utils.utils import random_split
			indices_list = random_split(sample_indices=list(range(len(self.train_dataset))), m_bins=n_workers, equal=True)
		
		elif split == 'random':
			from utils.utils import random_split
			indices_list = random_split(sample_indices=list(range(len(self.train_dataset))), m_bins=n_workers, equal=False)

		self.shard_sizes = [len(indices) for indices in indices_list]
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

			print("X train shape: ", X_train.shape)
			print("y train shape: ", y_train.shape)
			pos, neg =(y_train==1).sum().item() , (y_train==0).sum().item()
			print("Train set Positive counts: {}".format(pos),"Negative counts: {}.".format(neg), 'Split: {:.2%} - {:.2%}'.format(1. * pos/len(X_train), 1.*neg/len(X_train)))
			print("X test shape: ", X_test.shape)
			print("y test shape: ", y_test.shape)
			pos, neg =(y_test==1).sum().item() , (y_test==0).sum().item()
			print("Test set Positive counts: {}".format(pos),"Negative counts: {}.".format(neg), 'Split: {:.2%} - {:.2%}'.format(1. * pos/len(X_test), 1.*neg/len(X_test)))

			train_indices, valid_indices = get_train_valid_indices(len(X_train), self.train_val_split_ratio, self.sample_size_cap)

			train_set = Custom_Dataset(X_train[train_indices], y_train[train_indices], device=self.device)
			validation_set = Custom_Dataset(X_train[valid_indices], y_train[valid_indices], device=self.device)
			test_set = Custom_Dataset(X_test, y_test, device=self.device)

			return train_set, validation_set, test_set
		elif name == 'mnist':

			train = FastMNIST('datasets/MNIST', train=True, download=True)
			test = FastMNIST('datasets/MNIST', train=False, download=True)

			train_indices, valid_indices = get_train_valid_indices(len(train), self.train_val_split_ratio, self.sample_size_cap)
			
			from utils.Custom_Dataset import Custom_Dataset

			train_set = Custom_Dataset(train.data[train_indices], train.targets[train_indices], device=self.device)
			validation_set = Custom_Dataset(train.data[valid_indices],train.targets[valid_indices] , device=self.device)
			test_set = Custom_Dataset(test.data, test.targets, device=self.device)

			del train, test

			return train_set, validation_set, test_set

		elif name == 'cifar10':

			'''
			from torchvision import transforms			
			transform_train = transforms.Compose([
			    transforms.RandomCrop(32, padding=4),
			    transforms.RandomHorizontalFlip(),
			    transforms.ToTensor(),
			    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
			])

			transform_test = transforms.Compose([
			    transforms.ToTensor(),
			    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
			])
			'''

			train = FastCIFAR10('datasets/cifar', train=True, download=True)#, transform=transform_train)
			test = FastCIFAR10('datasets/cifar', train=False, download=True)#, transform=transform_test)

			train_indices, valid_indices = get_train_valid_indices(len(train), self.train_val_split_ratio, self.sample_size_cap)
			
			from utils.Custom_Dataset import Custom_Dataset

			train_set = Custom_Dataset(train.data[train_indices], train.targets[train_indices], device=self.device)
			validation_set = Custom_Dataset(train.data[valid_indices],train.targets[valid_indices] , device=self.device)
			test_set = Custom_Dataset(test.data, test.targets, device=self.device)
			del train, test

			return train_set, validation_set, test_set
		elif name == "sst":
			import torchtext.data as data
			text_field = data.Field(lower=True)
			from torch import long as torch_long
			label_field = LabelField(dtype = torch_long, sequential=False)

			import torchtext.datasets as datasets
			train_data, validation_data, test_data = datasets.SST.splits(text_field, label_field, fine_grained=True)

			indices_list = powerlaw(list(range(len(train_data))), self.n_workers)
			ratios = [len(indices) / len(train_data) for indices in indices_list]

			train_datasets = split_torchtext_dataset_ratios(train_data, ratios)

			text_field.build_vocab(*(train_datasets + [validation_data, test_data]))
			label_field.build_vocab(*(train_datasets + [validation_data, test_data]))

			self.args.text_field = text_field
			self.args.label_field = label_field

			return train_datasets, validation_data, test_data

		elif name == 'mr':

			import torchtext.data as data
			from utils import mydatasets

			text_field = data.Field(lower=True)
			from torch import long as torch_long
			label_field = LabelField(dtype = torch_long, sequential=False)
			# label_field = data.Field(sequential=False)

			train_data, dev_data = mydatasets.MR.splits(text_field, label_field, root='.data/mr')

			validation_data, test_data = dev_data.split(split_ratio=0.5, random_state = random.seed(1234))
			
			indices_list = powerlaw(list(range(len(train_data))), self.n_workers)
			ratios = [len(indices) / len(train_data) for indices in  indices_list]

			train_datasets = split_torchtext_dataset_ratios(train_data, ratios)

			text_field.build_vocab( *(train_datasets + [validation_data, test_data] ))
			label_field.build_vocab( *(train_datasets + [validation_data, test_data] ))

			self.args.text_field = text_field
			self.args.label_field = label_field

			return train_datasets, validation_data, test_data

		elif name == 'imdb':

			from torch import long as torch_long
			text_field = Field(tokenize = 'spacy', preprocessing = generate_bigrams)
			label_field = LabelField(dtype = torch_long)

			dirname = '.data/imdb/aclImdb'

			from torch.nn.init import normal_
			from torchtext import datasets


			train_data, test_data = datasets.IMDB.splits(text_field, label_field)

			test_data, _ = test_data.split(split_ratio=0.01 ,random_state = random.seed(1234))
			train_data, valid_data = train_data.split(split_ratio=0.8 ,random_state = random.seed(1234))

			indices_list = powerlaw(list(range(len(train_data))), self.n_workers)
			ratios = [len(indices) / len(train_data) for indices in  indices_list]

			train_datasets = split_torchtext_dataset_ratios(train_data, ratios)

			MAX_VOCAB_SIZE = 25_000

			text_field.build_vocab(*(train_datasets + [valid_data, test_data] ), max_size = MAX_VOCAB_SIZE, vectors = "glove.6B.100d",  unk_init = normal_)
			label_field.build_vocab( *(train_datasets + [valid_data, test_data] ))

			# INPUT_DIM = len(text_field.vocab)
			# OUTPUT_DIM = 1
			# EMBEDDING_DIM = 100

			PAD_IDX = text_field.vocab.stoi[text_field.pad_token]

			self.args.text_field = text_field
			self.args.label_field = label_field
			self.args.pad_idx = PAD_IDX

			return train_datasets, valid_data, test_data

		elif name == 'names':

			from utils.load_names import get_train_test
			from utils.Custom_Dataset import Custom_Dataset
			import torch
			from collections import Counter

			X_train, y_train, X_test, y_test, reference_dict = get_train_test()

			print("X train shape: ", X_train.shape)
			print("y train shape: ", y_train.shape)
			
			print("X test shape: ", X_test.shape)
			print("y test shape: ", y_test.shape)

			from utils.Custom_Dataset import Custom_Dataset
			train_set = Custom_Dataset(X_train, y_train)
			test_set = Custom_Dataset(X_test, y_test)

			return train_set, test_set


from torchvision.datasets import MNIST
class FastMNIST(MNIST):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)		
		
		self.data = self.data.unsqueeze(1).float().div(255)
		from torch.nn import ZeroPad2d
		pad = ZeroPad2d(2)
		self.data = torch.stack([pad(sample.data) for sample in self.data])

		self.targets = self.targets.long()

		self.data = self.data.sub_(0.1307).div_(0.3081)
		# Put both data and targets on GPU in advance
		self.data, self.targets = self.data, self.targets
		print('MNIST data shape {}, targets shape {}'.format(self.data.shape, self.targets.shape))

	def __getitem__(self, index):
		"""
		Args:
			index (int): Index

		Returns:
			tuple: (image, target) where target is index of the target class.
		"""
		img, target = self.data[index], self.targets[index]

		return img, target

from torchvision.datasets import CIFAR10
class FastCIFAR10(CIFAR10):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		
		# Scale data to [0,1]
		from torch import from_numpy
		self.data = from_numpy(self.data)
		self.data = self.data.float().div(255)
		self.data = self.data.permute(0, 3, 1, 2)

		self.targets = torch.Tensor(self.targets).long()

		# https://github.com/kuangliu/pytorch-cifar/blob/master/main.py
		for i, (mean, std) in enumerate(zip((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))):
			self.data[:,i].sub_(mean).div_(std)
		

		# Put both data and targets on GPU in advance
		self.data, self.targets = self.data, self.targets
		print('CIFAR10 data shape {}, targets shape {}'.format(self.data.shape, self.targets.shape))

	def __getitem__(self, index):
		"""
		Args:
			index (int): Index

		Returns:
			tuple: (image, target) where target is index of the target class.
		"""
		img, target = self.data[index], self.targets[index]

		return img, target

def powerlaw(sample_indices, n_workers, alpha=1.65911332899):
	# the smaller the alpha, the more extreme the division

	from scipy.stats import powerlaw
	import math
	party_size = int(len(sample_indices) / n_workers)
	b = np.linspace(powerlaw.ppf(0.01, alpha), powerlaw.ppf(0.99, alpha), n_workers)
	shard_sizes = list(map(math.ceil, b/sum(b)*party_size*n_workers))
	indices_list = []
	accessed = 0
	for worker_id in range(n_workers):
		indices_list.append(sample_indices[accessed:accessed + shard_sizes[worker_id]])
		accessed += shard_sizes[worker_id]
	return indices_list


def get_train_valid_indices(n_samples, train_val_split_ratio, sample_size_cap=None):
	indices = list(range(n_samples))
	random.seed(1111)
	random.shuffle(indices)
	split_point = int(n_samples * train_val_split_ratio)
	train_indices, valid_indices = indices[:split_point], indices[split_point:]
	if sample_size_cap is not None:
		train_indices = indices[:min(split_point, sample_size_cap)]

	return  train_indices, valid_indices 


def split_torchtext_dataset_ratios(data, ratios):
	train_datasets = []
	while len(ratios) > 1:
		ratio = ratios.pop(0)
		split_ratio = ratio / sum(ratios)
		train_dataset, data = data.split(split_ratio=split_ratio, random_state=random.seed(1234))
		train_datasets.append(train_dataset)
	train_datasets.append(data)

	return train_datasets


def generate_bigrams(x):
	n_grams = set(zip(*[x[i:] for i in range(2)]))
	for n_gram in n_grams:
		x.append(' '.join(n_gram))
	return x
'''

def get_df(pos, neg):
	data_rows = []
	for text in pos:
		data_rows.append(['positive', text.rstrip()])
	for text in neg:
		data_rows.append(['negative', text.rstrip()])
	return pd.DataFrame(data=data_rows, columns=['label', 'text'])

def create_data_txts_for_sst(n_workers, dirname='.data/sst'):
	train_txt = 'trees/train.txt'
	with open(os.path.join(dirname, train_txt), 'r') as file:
		train_samples = file.readlines()

	all_indices = list(range(len(train_samples)))
	random.seed(1111)
	n_samples_each = 8000 // 20
	sample_indices = random.sample(all_indices, n_samples_each * n_workers)

	foldername = "P{}_powerlaw".format(n_workers)
	if foldername in os.listdir(dirname):
		pass
	else:
		try:
			os.mkdir(os.path.join(dirname, foldername))
		except:
			pass

		indices_list = powerlaw(sample_indices, n_workers)
		for i, indices in enumerate(indices_list):
			with open(os.path.join(dirname, foldername,'P{}.txt'.format(i)) , 'w') as file:
				[file.write(train_samples[index]) for index in indices]

	return

def create_powerlaw_csvs(n_workers, dirname, train_df):

	# shuffle the train samples
	train_df = train_df.sample(frac=1)

	n_samples_each = len(train_df) // 20
	sample_indices = list(range(n_samples_each * n_workers))
	foldername = "P{}_powerlaw".format(n_workers)
	foldername = os.path.join(dirname, foldername)
	if foldername in os.listdir(dirname):
		pass
	else:
		try:
			os.mkdir(foldername)
		except:
			pass
		indices_list = powerlaw(sample_indices, n_workers)
		for i, indices in enumerate(indices_list):
			sub_df = train_df.iloc[indices]
			sub_df.to_csv(os.path.join(foldername,'P{}.csv'.format(i)), index=False)

	return



def read_samples(samples_dir):

	samples = []
	for file in os.listdir(samples_dir):
		with open(os.path.join(samples_dir, file), 'r') as line:
			samples.append(file.readlines())
	
	return [sample.rstrip() for sample in samplesl]

def create_data_csvs_for_mr(n_workers, dirname='.data/mr'):
	pos = 'rt-polaritydata/rt-polarity.pos'
	neg = 'rt-polaritydata/rt-polarity.neg'

	with open(os.path.join(dirname, pos), 'r', encoding='latin-1') as file:
		pos_samples =  file.readlines()

	with open(os.path.join(dirname, neg), 'r', encoding='latin-1') as file:
		neg_samples =  file.readlines()

	random.seed(1111)
	random.shuffle(pos_samples)
	random.shuffle(neg_samples)

	train, val, test = [], [], []
	N = len(pos_samples)

	split_points = [4000, (N-4000)//2+4000 ]
	train, val, test = np.array_split(pos_samples,split_points)
	train_, val_, test_ = np.array_split( neg_samples,split_points)

	train_df = get_df(train, train_)
	val_df = get_df(val, val_)
	test_df = get_df(test, test_)

	val_df.to_csv( os.path.join(dirname, 'val.csv') , index=False)
	test_df.to_csv( os.path.join(dirname, 'test.csv') , index=False)

	create_powerlaw_csvs(n_workers, dirname, train_df)

	return

def create_data_csvs_for_IMDB(n_workers, dirname):


	train_dir = os.path.join(dirname, 'train')
	pos_train_dir = os.path.join(train_dir, 'pos')
	neg_train_dir = os.path.join(train_dir, 'neg')

	test_dir = os.path.join(dirname, 'test')
	pos_test_dir = os.path.join(test_dir, 'pos')
	neg_test_dir = os.path.join(test_dir, 'neg')

	pos_samples = read_samples(pos_train_dir) + read_samples(pos_test_dir)
	neg_samples = read_samples(neg_train_dir) + read_samples(neg_test_dir)


	N_pos, N_neg = len(pos_samples), len(neg_samples)

	split_points = [int(N_pos*0.8), int(N_pos*0.9) ]
	train, val, test = np.array_split(pos_samples, split_points)
	
	split_points = [int(N_neg*0.8), int(N_pos*0.9) ]
	train_, val_, test_ = np.array_split(neg_samples,split_points)

	train_df = get_df(train, train_)
	val_df = get_df(val, val_)
	test_df = get_df(test, test_)

	val_df.to_csv( os.path.join(dirname, 'val.csv') , index=False)
	test_df.to_csv( os.path.join(dirname, 'test.csv') , index=False)

	create_powerlaw_csvs(n_workers, dirname, train_df)
	return
'''
