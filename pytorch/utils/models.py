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

class CNNCifar(nn.Module):
    def __init__(self, device=None):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)
 
class BasicBlock(nn.Module):
	expansion = 1
	def __init__(self, in_planes, planes, stride=1):
		super(BasicBlock, self).__init__()
		self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(planes)
		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,stride=1, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(planes)
		self.shortcut = nn.Sequential()
		if stride != 1 or in_planes != self.expansion*planes:
			self.shortcut = nn.Sequential(
				nn.Conv2d(in_planes, self.expansion*planes,kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(self.expansion*planes)
			)
	def forward(self, x):
		out = F.relu(self.bn1(self.conv1(x)))
		out = self.bn2(self.conv2(out))
		out += self.shortcut(x)
		out = F.relu(out)
		return out

class ResNet18(nn.Module):
    def __init__(self, block=BasicBlock, num_blocks=[2, 2, 2, 2], num_classes=10):
        super(ResNet18, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        # return out
        return F.log_softmax(out, dim=1)
# def ResNet18():
#     return ResNet(BasicBlock, [2, 2, 2, 2])

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
