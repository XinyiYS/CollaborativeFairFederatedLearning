# To mimic the running logic for federated learning


"""
1. get datasets and distribute them to workers
2. initialize the workers' models 
3. train their models locally for 10 epochs before 
4. average the locally trained models
5. start Federated Learning
6.
7.
8.

"""

import copy
import torch
import numpy as np
import syft as sy
hook = sy.TorchHook(torch)
from torch import nn, optim
from torch.utils.data import DataLoader
from utils.Custom_Dataset import Custom_Dataset

from utils.Worker import Worker

def init_workers(n_workers, X, y, indices_list, device):
	workers = []
	for i in range(n_workers):
	    syft_worker = sy.VirtualWorker(hook, id=str(i))
	    indices = indices_list[i]
	    data = X[indices]
	    target = y[indices]
	    device = device
	    worker = Worker(syft_worker, data, target, indices, id=str(i))
	    workers.append(worker)
	return workers


# adult dataset
def prepare_dataset(name='adult', train_test=True, train=True, test=False):
	if name == 'adult':
		from utils.load_adult import get_train_test

		train_data, train_target, test_data, test_target = get_train_test()
		
		X_train = torch.tensor(train_data.values, requires_grad=True).float()
		y_train = torch.tensor(train_target.values, requires_grad=True).long()
		X_test = torch.tensor(test_data.values, requires_grad=True).float()
		y_test = torch.tensor(test_target.values, requires_grad=True).long()
		if train_test == True:
			return (X_train, y_train), (X_test, y_test)
		elif train == True:
			return X_train, y_train
		else: #test==True:
			return X_test, y_test


train, test = prepare_dataset('adult', train_test=True)
X, y = train
X_test, y_test = test


test_batch_size = 1000
test_dataset = Custom_Dataset(X_test, y_test)
test_loader = DataLoader(
    dataset=test_dataset, batch_size=test_batch_size, shuffle=True)
print("datasets preparation successful")

# secure_worker = sy.VirtualWorker(hook, id="secure_worker")

n_workers = 5
n_samples = 20000
use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")

np.random.seed(1111)
from utils.dataset_split import random_split
indices_list = random_split(n_samples=n_samples, m_bins=n_workers, equal=True)


from utils.models import LogisticRegression
workers = init_workers(n_workers, X, y, indices_list, device)

print("Workers init successful")

input_dim, output_dim = X.shape[1], 2
for worker in workers:
	model = LogisticRegression(input_dim, output_dim)
	optimizer = optim.SGD(model.parameters(), lr=1e-3)
	loss_fn = nn.CrossEntropyLoss()

	worker.init_model_optimizer(model, optimizer, loss_fn)
	worker.init_train_loader(batch_size=16)
	# worker.init_test_loader(X_test, y_test)

print("Workers' models and optimizer etc successful")


for worker in workers:
	worker.train_locally(epochs=10)
	worker.evaluate(test_loader)

print("Workers training successful")
