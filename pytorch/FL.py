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

from utils.Worker import Worker


def init_workers(n_workers, X, y, indices_list, device):
    workers = []
    for i in range(n_workers):
        indices = indices_list[i]
        data = X[indices]
        target = y[indices]
        device = device
        worker = Worker(data, target, indices, id=str(i))
        workers.append(worker)
    return workers


# adult dataset
def prepare_dataset(name='adult', train_test=True, train=True, test=False):
    if name == 'adult':
        from utils.load_adult import get_train_test

        train_data, train_target, test_data, test_target = get_train_test()

        X_train = torch.tensor(train_data.values, requires_grad=False).float()
        y_train = torch.tensor(train_target.values, requires_grad=False).long()
        X_test = torch.tensor(test_data.values, requires_grad=False).float()
        y_test = torch.tensor(test_target.values, requires_grad=False).long()
        if train_test == True:
            return (X_train, y_train), (X_test, y_test)
        elif train == True:
            return X_train, y_train
        else:  # test==True:
            return X_test, y_test


train, test = prepare_dataset('adult', train_test=True)
X, y = train
X_test, y_test = test

train_val_split = 0.8
train_val_split_index = int(len(X) * 0.8)
X_train, y_train = X[:train_val_split_index], y[:train_val_split_index]
X_val, y_val = X[train_val_split_index:], y[train_val_split_index:]


def create_data_loader(X, y, batch_size):
    dataset = Custom_Dataset(X, y)
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)


val_loader = create_data_loader(X_val, y_val, batch_size=1000)
test_loader = create_data_loader(X_test, y_test, batch_size=1000)

print("datasets preparation successful")


n_workers = 5
n_samples = 30000
use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")

np.random.seed(1111)
from utils.dataset_split import random_split
indices_list = random_split(n_samples=n_samples, m_bins=n_workers, equal=True)

from utils.models import LogisticRegression, MLP_Adult
workers = init_workers(n_workers, X, y, indices_list, device)

print("Workers init successful")

input_dim, output_dim = X.shape[1], 2
for worker in workers:
    model = MLP_Adult(input_dim, output_dim)
    optimizer = optim.SGD(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    worker.init_model_optimizer(model, optimizer, loss_fn)
    worker.init_train_loader(batch_size=16)
    worker.val_loader = val_loader

print("Workers' models and optimizer etc successful")




def evaluate(model, eval_loader):
    model.eval()
    correct = 0
    total = 0
    loss_fn = nn.CrossEntropyLoss()
    for i, (batch_data, batch_target) in enumerate(eval_loader):
        batch_data, batch_target = batch_data, batch_target
        outputs = model(batch_data)
        loss = loss_fn(outputs, batch_target)
        _, predicted = torch.max(outputs.data, 1)
        total += len(batch_target)
        # for gpu, bring the predicted and labels back to cpu for python
        # operations to work
        correct += (predicted == batch_target).sum()
    accuracy = 1. * correct / total
    print("Loss: {}. Accuracy: {:.0%}.\n".format(loss, accuracy))
    return loss, accuracy


def pretrain_locally(workers, epochs, test_loader=None):
	for worker in workers:
	    worker.train_locally(epochs=epochs)
	    if test_loader:
		    worker.evaluate(test_loader)
	print("Workers training successful")
	return

from utils.utils import averge_models, average_gradient_updates, add_update_to_model, compute_grad_update

pretrain_locally(workers, 5, test_loader)

models = [worker.model for worker in workers]
federated_model = averge_models(models)

federated_epochs = 5
for epoch in range(federated_epochs):
	grad_updates = []
	for worker in workers:
		model_before = copy.deepcopy(worker.model)		
		worker.train_locally(5)
		model_after = copy.deepcopy(worker.model)
		grad_updates.append( compute_grad_update(model_before, model_after))

	averaged_update = average_gradient_updates(grad_updates)
	federated_model = add_update_to_model(federated_model, averaged_update)
	evaluate(federated_model, test_loader)

