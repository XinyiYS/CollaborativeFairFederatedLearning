import torch
from torch.utils.data import DataLoader


from torch.utils.data import Dataset


class Custom_Dataset(Dataset):

    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.count = len(X)

    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class Worker():

    def __init__(self, train_loader, indices=None, model=None, optimizer=None, loss_fn=None, id=None, device=None):
        # self.data = data
        # self.target = target
        self.train_loader = train_loader
        self.indices = indices
        self.model = model
        self.optimizer = optimizer
        self.id = id
        self.device = device
        self.loss_fn = loss_fn

    def init_model_optimizer(self, model, optimizer=None, loss_fn=None):
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def init_train_loader(self, batch_size=16):
        self.dataset = Custom_Dataset(self.data, self.target)
        self.train_loader = DataLoader(
            dataset=self.dataset, batch_size=batch_size, shuffle=True)

    def train_locally(self, epochs):
        iter = 0
        self.model.train()
        self.model = self.model.to(self.device)
        for epoch in range(int(epochs)):
            for i, (batch_data, batch_target) in enumerate(self.train_loader):
                batch_data, batch_target = batch_data.to(
                    self.device), batch_target.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(batch_data)
                loss = self.loss_fn(outputs, batch_target)
                loss.backward()
                self.optimizer.step()

                iter += 1
                if iter % 2000 == 0:
                    self.evaluate_locally(iter=iter)
        return

    def evaluate_locally(self, iter=None):
        """
        unfinished yet
        """
        if not self.val_loader:
            print("Worker's own val loader not initialized")
            return
        self.model.eval()
        correct = 0
        total = 0
        for i, (batch_data, batch_target) in enumerate(self.val_loader):
            batch_data, batch_target = batch_data.to(
                self.device), batch_target.to(self.device)
            outputs = self.model(batch_data)
            loss = self.loss_fn(outputs, batch_target)
            _, predicted = torch.max(outputs.data, 1)
            total += len(batch_target)
            # for gpu, bring the predicted and labels back to cpu for python
            # operations to work
            correct += (predicted == batch_target).sum()

        accuracy = 1. * correct / total
        print("Worker: {} Iteration: {}. Loss: {}. Accuracy: {:.0%}.".format(
            self.id, iter, loss, accuracy))

        return loss, accuracy
