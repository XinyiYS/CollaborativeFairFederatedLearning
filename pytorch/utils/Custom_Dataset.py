from torch.utils.data import Dataset


class Custom_Dataset(Dataset):

	def __init__(self, X, y, device=None):
		self.X = X.to(device)
		self.y = y.to(device)
		self.count = len(X)
		self.device = device

	def __len__(self):
		return self.count

	def __getitem__(self, idx):
		return self.X[idx], self.y[idx]
