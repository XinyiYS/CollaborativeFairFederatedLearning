import syft as sy
import torch as th

sy.create_sandbox(globals(), verbose=False)

device = th.device("cuda")

"""Then search for a dataset"""

boston_data = grid.search("#boston", "#data")
boston_target = grid.search("#boston", "#target")

"""We load a model and an optimizer"""

n_features = boston_data['alice'][0].shape[1]
n_targets = 1

model = th.nn.Linear(n_features, n_targets).to(device)

"""Here we cast the data fetched in a `FederatedDataset`. See the workers which hold part of the data."""
print("A total of {} workers".format(len(boston_data.keys())))
# Cast the result in BaseDatasets
datasets = []
for worker in boston_data.keys():
    dataset = sy.BaseDataset(boston_data[worker][0], boston_target[worker][0])
    datasets.append(dataset)

# Build the FederatedDataset object
dataset = sy.FederatedDataset(datasets)
print(dataset.workers)
optimizers = {}
for worker in dataset.workers:
    optimizers[worker] = th.optim.Adam(params=model.parameters(), lr=1e-2)
    # optimizers[worker] = th.optim.SGD(model.parameters(), lr=0.05)

"""We put it in a `FederatedDataLoader` and specify options"""

train_loader = sy.FederatedDataLoader(
    dataset, batch_size=32, shuffle=False, drop_last=False)
validation_loader 
test_loader


"""And finally we iterate over epochs. You can see how similar this is compared to pure and local PyTorch training!"""
epochs = 5

from collections import defaultdict
worker_train_counts = defaultdict(int)

global_gradient_history = None
print_grad = True
for epoch in range(1, epochs + 1):
    loss_accum = 0



    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        # send the model to the respective worker
        model.send(data.location)
        worker_train_counts[data.location.id] += 1

        # get respective worker's optimizer and step once
        optimizer = optimizers[data.location.id]
        optimizer.zero_grad()
        pred = model(data)
        loss = ((pred.view(-1) - target)**2).mean()
        loss.backward()

        weights = weights_ = None
        grad = None
        if print_grad:
            model.get()
            # weights = model.weight
            unclipped_grads = [param.grad for param in list(model.parameters())] # collect the unclipped gradients for calculation

            # clip the grad
            # th.nn.utils.clip_grad_norm_(model.parameters(), )            
            model.send(data.location)

        optimizer.step()

        # get the updated model back and get the loss
        model.get()
        if print_grad:
            print('Model weights after :', model.weight)
            print_grad = False
            
        if global_gradient_history is not None:
            global_gradient_history += model.weight.grad
        else:
            global_gradient_history = model.weight.grad

        loss = loss.get()
        loss_accum += float(loss)

        if batch_idx % 8 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tBatch loss: {:.6f}'.format(
                epoch, batch_idx, len(train_loader),
                100. * batch_idx / len(train_loader), loss.item()))
    
    # print('Global gradient history: ', global_gradient_history)
    print('Total loss', loss_accum)


print('Worker train counts: ', worker_train_counts.items())