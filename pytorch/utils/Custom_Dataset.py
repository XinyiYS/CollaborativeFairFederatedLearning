from torch.utils.data import Dataset

class Custom_Dataset(Dataset):

	def __init__(self, X, y, device=None, transform=None):
		self.data = X.to(device)
		self.targets = y.to(device)
		self.count = len(X)
		self.device = device
		self.transform = transform

	def __len__(self):
		return self.count

	def __getitem__(self, idx):
		if self.transform:
			return self.transform(self.data[idx]), self.targets[idx]

		return self.data[idx], self.targets[idx]
