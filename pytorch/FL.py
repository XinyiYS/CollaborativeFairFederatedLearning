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
from torch import nn, optim
from torch.utils.data import DataLoader
from utils.Custom_Dataset import Custom_Dataset

from torch.utils.data.sampler import SubsetRandomSampler

# adult dataset
def prepare_dataset(name='adult', train_test=True, train=True, test=False):
    if name == 'adult':
        from utils.load_adult import get_train_test

        train_data, train_target, test_data, test_target = get_train_test()

        X_train = torch.tensor(train_data.values, requires_grad=False).float()
        y_train = torch.tensor(train_target.values, requires_grad=False).long()
        X_test = torch.tensor(test_data.values, requires_grad=False).float()
        y_test = torch.tensor(test_target.values, requires_grad=False).long()

        train_set = Custom_Dataset(X_train, y_train)
        test_set = Custom_Dataset(X_test, y_test)

        return train_set, test_set
        if train_test == True:
            return (X_train, y_train), (X_test, y_test)
        elif train == True:
            return X_train, y_train
        else:  # test==True:
            return X_test, y_test

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

train_dataset, test_dataset = prepare_dataset('mnist', train_test=True)

train_val_split = 0.8
train_val_split_index = int(len(train_dataset) * 0.8)

indices = list(range(len(train_dataset)))
np.random.seed(1111)
np.random.shuffle(indices)

train_idx, valid_idx = indices[train_val_split_index:], indices[:train_val_split_index]
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

batch_size = 16
# train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler,)

valid_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=valid_sampler)

test_batch_size = 1000
test_loader =  DataLoader(test_dataset, batch_size=batch_size)
print("datasets preparation successful")


from utils.models import LogisticRegression, MLP_LogReg, MLP_Net, CNN_Net

# User set argument
n_workers = 5
balanced_datasets=True

n_samples = 30000
use_cuda = True
device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu" )

model_fn = MLP_Net
loss_fn = nn.CrossEntropyLoss()

np.random.seed(1111)

from utils.utils import random_split
indices_list = random_split(sample_indices=train_idx, m_bins=n_workers, equal=balanced_datasets)


from utils.Worker import Worker

def init_workers(n_workers, train_dataset, indices_list, device):
	workers = []
	for i in range(n_workers):
		train_sampler = SubsetRandomSampler(indices_list[i])
		train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)

		model = model_fn()
		optimizer = optim.SGD(model.parameters(), lr=1e-3)

		worker = Worker(train_loader=train_loader, indices=indices, 
						id=str(i), model=model, optimizer=optimizer, loss_fn = loss_fn,
						device=device)
		workers.append(worker)
	return workers

workers = init_workers(n_workers, train_dataset, indices_list, device)
print("Workers init successful")


def distribute_points(points, marginal_contributions, epsilon=1e-4):
	# normalize so that the max is equal to n_workers - 1

	# set small values to 0
	marginal_contributions[marginal_contributions.abs() < epsilon] = 0
	if not (marginal_contributions==0).all():
		ratio = (len(points) - 1) / torch.max(marginal_contributions)
		marginal_contributions *= ratio
	print('resized contributions:', marginal_contributions)

	return points + marginal_contributions

def sort_grad_updates(grad_updates, marginal_contributions):
	# sort the grad_updates according to the marginal_contributions in a descending order
	return [(grad_update, worker_id) for grad_update, marg_contr, worker_id in sorted(zip(grad_updates, marginal_contributions, range(len(grad_updates ) )), key=lambda x:x[1], reverse=True)]


from utils.utils import averge_models, average_gradient_updates, \
	add_update_to_model, compute_grad_update, compare_models,  \
	pretrain_locally, leave_one_out_evaluate, evaluate, compute_shapley

pretrain_epochs = 1
fl_epochs = 1
fl_individual_epochs = 1

# uncomment for local pretraining
pretrain_locally(workers, pretrain_epochs, test_loader=None)
worker_model_test_accs_before = [evaluate(worker.model, test_loader, device, verbose=False)[1].tolist() for worker in workers]

models = [worker.model for worker in workers]
federated_model = averge_models(models, device=device)

points = torch.zeros((n_workers))
sharing_ledger = torch.zeros((n_workers)) 
shapley_values = torch.zeros((n_workers)) 

print("\nStart federated learning ")

for epoch in range(fl_epochs):
	grad_updates = []
	for worker in workers:
		model_before = copy.deepcopy(worker.model)		
		worker.train_locally(fl_individual_epochs)
		model_after = copy.deepcopy(worker.model)
		grad_updates.append(compute_grad_update(model_before, model_after, device=device))

	# updates the federated model in function for efficiency
	marginal_contributions = leave_one_out_evaluate(federated_model, grad_updates, valid_loader, device)
	print("Marginal contributions are: ", marginal_contributions)

	# shapley_values += compute_shapley(grad_updates, federated_model, test_loader, device)

	points = distribute_points(points, marginal_contributions)
	sorted_grad_updates = sort_grad_updates(grad_updates, marginal_contributions)

	for download_worker_id, worker in enumerate(workers):
		acquired_updates = []
		
		for grad_update, upload_worker_id in sorted_grad_updates:
			if upload_worker_id != download_worker_id and points[download_worker_id] > 1: # not self and sufficient budget
				points[download_worker_id] -= 1
				points[upload_worker_id] += 1 # paying to this worker
				acquired_updates.append(grad_update)
				sharing_ledger[upload_worker_id] += 1

		averaged_acquired_update = average_gradient_updates(acquired_updates)
		worker.model = add_update_to_model(worker.model, averaged_acquired_update, device=device)

	# averaged_update = average_gradient_updates(grad_updates, device=device)
	# federated_model = add_update_to_model(federated_model, averaged_update, device=device)
	evaluate(federated_model, test_loader, device)


worker_model_test_accs = [evaluate(worker.model, test_loader, device, verbose=False)[1].tolist() for worker in workers]
# print("The number of gradient sharing by workers:", sharing_ledger)
# print(worker_model_test_accs)

import scipy.stats
corrs = scipy.stats.pearsonr(sharing_ledger, worker_model_test_accs)
print("test_acc vs sharing ledger: ", corrs)

worker_model_improvements = [ now-before for now, before in zip(worker_model_test_accs, worker_model_test_accs_before)]
corrs = scipy.stats.pearsonr(sharing_ledger, worker_model_improvements)
print("test_acc improvements vs sharing ledger: ", corrs)

corrs = scipy.stats.pearsonr(sharing_ledger, shapley_values)
print('sharing ledge vs shapley values: ', corrs)

corrs = scipy.stats.pearsonr(shapley_values, worker_model_improvements)
print('shapley values vs model improvements: ', corrs)

print()
print('worker_model_test_accs: ', worker_model_test_accs)
print('worker_model_improvements: ', worker_model_improvements)
print('sharing ledger: ', sharing_ledger)
print('shapley values: ', shapley_values)