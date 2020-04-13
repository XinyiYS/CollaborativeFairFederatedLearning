import torch
from torch.utils.data import DataLoader


from torch.utils.data import Dataset
import utils
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

	def __init__(self, train_loader, model=None, optimizer=None, standalone_model=None, standalone_optimizer=None, loss_fn=None, id=None, sharing_lambda=0.1, device=None):
		self.train_loader = train_loader
		self.model = model
		self.optimizer = optimizer
		self.standalone_model = standalone_model
		self.standalone_optimizer = standalone_optimizer
		self.sharing_lambda = sharing_lambda
		self.id = id
		self.device = device
		self.loss_fn = loss_fn
		self.param_count = sum([p.numel() for p in self.model.parameters()])

	def init_train_loader(self, batch_size=16):
		self.dataset = Custom_Dataset(self.data, self.target)
		self.train_loader = DataLoader(
			dataset=self.dataset, batch_size=batch_size, shuffle=True)

	def train(self, epochs, grad_update=False):
		if grad_update:
			model_before = copy.deepcopy(self.model)
		iter = 0
		self.model.train()
		self.model = self.model.to(self.device)
		
		self.standalone_model.train()
		self.standalone_model = self.standalone_model.to(self.device)

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


				self.standalone_optimizer.zero_grad()
				outputs = self.standalone_model(batch_data)
				loss = self.loss_fn(outputs, batch_target)
				loss.backward()
				self.standalone_optimizer.step()

		if grad_update:
			grad_update = utils.compute_grad_update(model_before, self.model, device=self.device)
			return grad_update
