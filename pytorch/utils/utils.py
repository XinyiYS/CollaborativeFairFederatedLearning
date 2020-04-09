import copy
import torch

def averge_models(models):
    final_model = copy.deepcopy(models[0])
    for i, model in enumerate(models):
        for param_final, param_redundant in zip(final_model.parameters(), model.parameters()):
            if i == 0:
                param_final.data = param_redundant.data * 1./ len(models)
            else:
                param_final.data += param_redundant.data * 1./ len(models)
    return final_model

def compute_grad_update(old_model, new_model):
    # maybe later to implement on selected layers/parameters
    return [(new_param.data - old_param.data) for old_param, new_param in zip(old_model.parameters(), new_model.parameters())]


def add_gradient_updates(grad_update_1, grad_update_2):
    assert len(grad_update_1) == len(grad_update_2), "Lengths of the two grad_updates not equal"
    return [ grad_update_1[i] + grad_update_2[i]  for i in range(len(grad_update_1))]

def average_gradient_updates(grad_updates):
	num_updates = len(grad_updates)
	len_first = len(grad_updates[0]) if grad_updates else None
	assert all(len(i) == len_first for i in grad_updates), "Different shapes of parameters. Cannot take average."
	averaged_gradient_updates = []

	for i in range(len(grad_updates[0])):
		averaged_gradient_updates.append(torch.stack([grad_update[i] for grad_update in grad_updates]).sum(dim=0))
	return averaged_gradient_updates


def add_update_to_model(model, update, weight=1.0):
    for param_model, param_update in zip(model.parameters(), update):
        param_model.data += weight * param_update.data
    return model