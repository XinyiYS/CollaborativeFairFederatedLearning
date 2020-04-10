import copy
import torch
from torch import nn
from .Worker import Custom_Dataset
from torch.utils.data import DataLoader

def averge_models(models, device=None):
    final_model = copy.deepcopy(models[0])
    if device:
        models = [model.to(device) for model in models]
        final_model = final_model.to(device)

    averaged_parameters = average_gradient_updates([list(model.parameters()) for model in models])
    
    for param, avg_param in zip(final_model.parameters(), averaged_parameters):
        param.data = avg_param.data
    return final_model


def compute_grad_update(old_model, new_model, device=None):
    # maybe later to implement on selected layers/parameters
    if device:
        old_model, new_model = old_model.to(device), new_model.to(device)
    return [(new_param.data - old_param.data) for old_param, new_param in zip(old_model.parameters(), new_model.parameters())]


def add_gradient_updates(grad_update_1, grad_update_2):
    assert len(grad_update_1) == len(
        grad_update_2), "Lengths of the two grad_updates not equal"
    return [grad_update_1[i] + grad_update_2[i] for i in range(len(grad_update_1))]


def average_gradient_updates(grad_updates, device=None):
    if grad_updates:
        len_first = len(grad_updates[0])
        assert all(len(i) == len_first for i in grad_updates), "Different shapes of parameters. Cannot take average."
    else:
        return

    if device:
        for i, grad_update in enumerate(grad_updates):
            grad_updates[i] = [param.to(device) for param in grad_update]

    averaged_gradient_updates = []

    for i in range(len(grad_updates[0])):
        averaged_gradient_updates.append(torch.stack(
            [grad_update[i] for grad_update in grad_updates]).mean(dim=0))
    return averaged_gradient_updates


def add_update_to_model(model, update, weight=1.0, device=None):
    if not update: return model
    if device:
        model = model.to(device)
        update = [param.to(device) for param in update]
            
    for param_model, param_update in zip(model.parameters(), update):
        param_model.data += weight * param_update.data
    return model


def create_data_loader(X, y, batch_size):
    dataset = Custom_Dataset(X, y)
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

def compare_models(model1, model2):
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            return False # two models have different weights
    return True


def pretrain_locally(workers, epochs, test_loader=None):
    for worker in workers:
        worker.train_locally(epochs=epochs)
        if test_loader:
            evaluate(worker.model, test_loader, worker.device)
    print("Workers Local pretraining successful")
    return


def evaluate(model, eval_loader, device, loss_fn=nn.CrossEntropyLoss(),verbose=True):
    model.eval()
    model = model.to(device)
    correct = 0
    total = 0
    for i, (batch_data, batch_target) in enumerate(eval_loader):
        batch_data, batch_target = batch_data.to(device), batch_target.to(device)
        outputs = model(batch_data)
        loss = loss_fn(outputs, batch_target)
        _, predicted = torch.max(outputs.data, 1)
        total += len(batch_target)
        # for gpu, bring the predicted and labels back to cpu for python
        # operations to work
        correct += (predicted == batch_target).sum()
    accuracy = 1. * correct / total
    if verbose:
        print("Loss: {}. Accuracy: {:.0%}.\n".format(loss, accuracy))
    return loss, accuracy


def leave_one_out_evaluate(federated_model, grad_updates, eval_loader, device, update=True):
    if not update:
        # by using this flag, the federated_model in the main training logic is NOT updated
        # Seems to be some issue with this flag, use default True until fully investigated and fixed
        federated_model = copy.deepcopy(federated_model)

    averaged_gradient_updates = average_gradient_updates(grad_updates, device)
    federated_model = add_update_to_model(federated_model, averaged_gradient_updates, device=device)
    curr_loss, curr_val_acc = evaluate(federated_model, eval_loader, device, verbose=False)

    loo_model = copy.deepcopy(federated_model)
    loo_losses, loo_val_accs = [], []
    for grad_update in grad_updates:
        loo_model = add_update_to_model(loo_model, grad_update, weight = -1./len(grad_updates), device=device)
        loss, val_acc = evaluate(loo_model, eval_loader, device, verbose=False)
        loo_losses.append(loss)
        loo_val_accs.append(val_acc)
        loo_model = add_update_to_model(loo_model, grad_update, weight = 1./len(grad_updates), device=device)

    del loo_model
    del averaged_gradient_updates

    # scalar - 1D torch tensor subtraction -> 1D torch tensor
    marginal_contributions = curr_val_acc - torch.tensor(loo_val_accs) 

    return marginal_contributions

import numpy as np
np.random.seed(1111)


def random_split(n_samples, m_bins, equal=True):
    all_indices = np.arange(n_samples)
    np.random.shuffle(all_indices)
    if equal:
        indices_list = np.split(all_indices, m_bins)
    else:
        split_points = np.random.choice(
            n_samples - 2, m_bins - 1, replace=False) + 1
        split_points.sort()
        indices_list = np.split(all_indices, split_points)

    return indices_list
