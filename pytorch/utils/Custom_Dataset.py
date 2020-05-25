from torch.utils.data import Dataset


class Custom_Dataset(Dataset):

	def __init__(self, X, y, device=None, transform=None):
		self.X = X.to(device)
		self.y = y.to(device)
		self.count = len(X)
		self.device = device
		self.transform = transform

	def __len__(self):
		return self.count

	def __getitem__(self, idx):
		if self.transform:
			return self.transform(self.X[idx]), self.y[idx]

		return self.X[idx], self.y[idx]
