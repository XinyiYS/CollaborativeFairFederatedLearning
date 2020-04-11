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

from utils.Worker import Worker
from utils.Data_Prepper import Data_Prepper
from utils.Federated_Learner import Federated_Learner

from utils.models import LogisticRegression, MLP_LogReg, MLP_Net, CNN_Net


# from torch.utils.data import DataLoader
# from utils.Custom_Dataset import Custom_Dataset
# from torch.utils.data.sampler import SubsetRandomSampler


# def init_workers(n_workers, worker_train_loaders, model_fn, optimizer_fn, device, lr=0.15):
# 	assert n_workers == len(worker_train_loaders), "Num of workers is not equal to num of loaders"
# 	workers = []
# 	for i, worker_train_loader in enumerate(worker_train_loaders):
# 		model = model_fn()
# 		optimizer = optimizer_fn(model.parameters(), lr=lr)

# 		worker = Worker(train_loader=worker_train_loader,
# 						model=model, optimizer=optimizer, loss_fn=loss_fn,
# 						id=str(i), sharing_lambda=sharing_lambda,
# 						device=device)
# 		workers.append(worker)
# 	return workers

# print("datasets and valid/test loaders preparation successful")

data_prep = Data_Prepper('mnist', train_batch_size=16)


use_cuda = True


# User set argument
n_workers = 3
balanced = True
device = torch.device("cuda" if torch.cuda.is_available()
                      and use_cuda else "cpu")

model_fn = MLP_Net
optimizer_fn = optim.SGD
loss_fn = nn.CrossEntropyLoss()
loss_fn = nn.NLLLoss()
lr = 0.15
sharing_lambda = 0.1

args = {
    # system parameters
    'device': torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu"),


    # setting parameters
    'n_workers': 3,
    'balanced': True,
    'sharing_lambda': 0.1,


    # model parameters
    'model_fn': MLP_Net,
    'optimizer_fn': optim.SGD,
    'loss_fn': nn.NLLLoss(),
    'lr': 0.15,

    # training parameters
    'pretrain_epochs': 5,
    'fl_epochs': 20,
    'fl_individual_epochs': 5,
}

# worker_train_loaders = data_prep.get_train_loaders(n_workers, balanced=True)
# workers = init_workers(n_workers, worker_train_loaders, model_fn, optimizer_fn, device)

federated_learner = Federated_Learner(args, data_prep)

print("FL init successful")


exit()


def distribute_points(points, marginal_contributions, epsilon=1e-4):
    # normalize so that the max is equal to n_workers - 1

    # set small values to 0
    marginal_contributions[marginal_contributions.abs() < epsilon] = 0
    if not (marginal_contributions == 0).all():
        ratio = (len(points) - 1) / torch.max(marginal_contributions)
        marginal_contributions *= ratio
    print('resized contributions:', marginal_contributions)

    return points + marginal_contributions


def sort_grad_updates(grad_updates, marginal_contributions):
    # sort the grad_updates according to the marginal_contributions in a
    # descending order
    return [(grad_update, worker_id) for grad_update, marg_contr, worker_id in sorted(zip(grad_updates, marginal_contributions, range(len(grad_updates))), key=lambda x:x[1], reverse=True)]


from utils.utils import averge_models, average_gradient_updates, \
    add_update_to_model, compute_grad_update, compare_models,  \
    pretrain_locally, leave_one_out_evaluate, evaluate, compute_shapley

pretrain_epochs = 5
fl_epochs = 20
fl_individual_epochs = 5

# uncomment for local pretraining
pretrain_locally(workers, pretrain_epochs, test_loader=None)
worker_model_test_accs_before = [evaluate(
    worker.model, test_loader, device, verbose=False)[1].tolist() for worker in workers]

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
        grad_updates.append(compute_grad_update(
            model_before, model_after, device=device))

    # updates the federated model in function for efficiency
    marginal_contributions = leave_one_out_evaluate(
        federated_model, grad_updates, valid_loader, device)
    print("Marginal contributions are: ", marginal_contributions)

    # shapley_values += compute_shapley(grad_updates, federated_model, test_loader, device)

    points = distribute_points(points, marginal_contributions)
    sorted_grad_updates = sort_grad_updates(
        grad_updates, marginal_contributions)

    for download_worker_id, worker in enumerate(workers):
        acquired_updates = []

        for grad_update, upload_worker_id in sorted_grad_updates:
            # not self and sufficient budget
            if upload_worker_id != download_worker_id and points[download_worker_id] > 1:
                points[download_worker_id] -= 1
                points[upload_worker_id] += 1  # paying to this worker
                acquired_updates.append(grad_update)
                sharing_ledger[upload_worker_id] += 1

        averaged_acquired_update = average_gradient_updates(acquired_updates)
        worker.model = add_update_to_model(
            worker.model, averaged_acquired_update, device=device)

    # averaged_update = average_gradient_updates(grad_updates, device=device)
    # federated_model = add_update_to_model(federated_model, averaged_update, device=device)
    evaluate(federated_model, test_loader, device)


worker_model_test_accs = [evaluate(worker.model, test_loader, device, verbose=False)[
    1].tolist() for worker in workers]
# print("The number of gradient sharing by workers:", sharing_ledger)
# print(worker_model_test_accs)

import scipy.stats
corrs = scipy.stats.pearsonr(sharing_ledger, worker_model_test_accs)
print("test_acc vs sharing ledger: ", corrs)

worker_model_improvements = [
    now - before for now, before in zip(worker_model_test_accs, worker_model_test_accs_before)]
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
