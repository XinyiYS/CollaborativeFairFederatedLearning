import torch
import torch.nn as nn
from torch.autograd import Variable
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


class MLP(nn.Module):

	def __init__(self, input_dim=85, output_dim=2, device=None):
		super(MLP, self).__init__()
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



class CNN_Text(nn.Module):
	
	def __init__(self, args=None, device=None):
		super(CNN_Text,self).__init__()

		
		self.args = args
		self.device = device
		
		V = args.embed_num
		D = args.embed_dim
		C = args.class_num
		Ci = 1
		Co = args.kernel_num
		Ks = args.kernel_sizes

		self.embed = nn.Embedding(V, D)
		self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])

		'''
		self.conv13 = nn.Conv2d(Ci, Co, (3, D))
		self.conv14 = nn.Conv2d(Ci, Co, (4, D))
		self.conv15 = nn.Conv2d(Ci, Co, (5, D))
		'''
		self.dropout = nn.Dropout(0.5)
		# self.dropout = nn.Dropout(args.dropout)
		self.fc1 = nn.Linear(len(Ks)*Co, C)

	def conv_and_pool(self, x, conv):
		x = F.relu(conv(x)).squeeze(3) #(N,Co,W)
		x = F.max_pool1d(x, x.size(2)).squeeze(2)
		return x


	def forward(self, x):

		x = self.embed(x) # (W,N,D)
		x = x.permute(1,0,2) # -> (N,W,D)

		if not self.args or self.args.static:
			x = Variable(x).to(self.device)

		x = x.unsqueeze(1) # (N,Ci,W,D)
		x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1] #[(N,Co,W), ...]*len(Ks)
		x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x] #[(N,Co), ...]*len(Ks)
		x = torch.cat(x, 1)
		'''
		x1 = self.conv_and_pool(x,self.conv13) #(N,Co)
		x2 = self.conv_and_pool(x,self.conv14) #(N,Co)
		x3 = self.conv_and_pool(x,self.conv15) #(N,Co)
		x = torch.cat((x1, x2, x3), 1) # (N,len(Ks)*Co)
		'''
		x = self.dropout(x) # (N,len(Ks)*Co)
		logit = self.fc1(x) # (N,C)
		return logit