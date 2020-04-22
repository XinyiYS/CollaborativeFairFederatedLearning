import torch
import torch.nn as nn
import torch.nn.functional as F


# for MNIST 32*32
class CNN_Net(nn.Module):

	def __init__(self, device=None):
		super(CNN_Net, self).__init__()
		self.conv1 = nn.Conv2d(1, 64, 3, 1)
		self.conv2 = nn.Conv2d(64, 16, 7, 1)
		self.fc1 = nn.Linear(4 * 4 * 16, 200)
		self.fc2 = nn.Linear(200, 10)

	def forward(self, x):
		x = x.view(-1, 1, 32, 32)
		x = F.tanh(self.conv1(x))
		x = F.max_pool2d(x, 2, 2)
		x = F.tanh(self.conv2(x))
		x = F.max_pool2d(x, 2, 2)
		x = x.view(-1, 4 * 4 * 16)
		x = F.tanh(self.fc1(x))
		x = self.fc2(x)
		return F.log_softmax(x, dim=1)

# for MNIST 32*32
class MLP_Net(nn.Module):

	def __init__(self, device=None):
		super(MLP_Net, self).__init__()
		self.fc1 = nn.Linear(1024, 128)
		self.fc2 = nn.Linear(128, 64)
		self.fc3 = nn.Linear(64, 10)

	def forward(self, x):
		x = x.view(-1,  1024)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return F.log_softmax(x, dim=1)


class LogisticRegression(nn.Module):

	def __init__(self, input_dim=85, output_dim=2, device=None):
		super(LogisticRegression, self).__init__()
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.linear = torch.nn.Linear(self.input_dim, self.output_dim)

	def forward(self, x):
		outputs = self.linear(x)
		return outputs


class MLP_Adult(nn.Module):

	def __init__(self, input_dim=85, output_dim=2, device=None):
		super(MLP_Adult, self).__init__()
		self.fc1 = nn.Linear(input_dim, 64)
		self.fc2 = nn.Linear(64, output_dim)

		# self.linear = torch.nn.Linear(input_dim, output_dim)

	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		return F.log_softmax(x, dim=1)

# For names language classification
class RNN(nn.Module):

	def __init__(self, input_size=57, output_size=7, hidden_size=64, device=None):
		super(RNN, self).__init__()

		self.hidden_size = hidden_size

		self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
		self.i2o = nn.Linear(input_size + hidden_size, output_size)
		self.softmax = nn.LogSoftmax(dim=1)
		self.device = device

	def forward(self, line_tensors):
		return torch.cat([self.forward_one_tensor(line_tensor) for line_tensor in line_tensors])

	def forward_one_tensor(self, line_tensor):
		hidden = self.initHidden()
		for i in range(line_tensor.size()[0]):
			if line_tensor[i][0] != -1: # ignore the padded -1 at the end
				output, hidden = self.forward_once(line_tensor[i].view(1,-1), hidden)
		return output

	def forward_once(self, input, hidden):
		combined = torch.cat((input, hidden), 1)
		hidden = self.i2h(combined)
		output = self.i2o(combined)
		output = self.softmax(output)
		return output, hidden

	def initHidden(self):
		return torch.zeros(1, self.hidden_size).to(self.device)
