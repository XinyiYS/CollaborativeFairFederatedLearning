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

	def __init__(self, input_dim=86, output_dim=2, device=None):
		super(LogisticRegression, self).__init__()
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.linear = torch.nn.Linear(self.input_dim, self.output_dim)

	def forward(self, x):
		outputs = self.linear(x)
		return outputs


class MLP(nn.Module):

	def __init__(self, input_dim=86, output_dim=2, device=None):
		super(MLP, self).__init__()
		self.fc1 = nn.Linear(input_dim, 32)
		self.fc2 = nn.Linear(32, output_dim)

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

# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
# LeNet
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
 
# https://www.tensorflow.org/tutorials/images/cnn
class CNNCifar_TF(nn.Module):
	def __init__(self, device=None):
		super(CNNCifar_TF, self).__init__()
		self.conv1 = nn.Conv2d(3, 32, 3)
		self.conv2 = nn.Conv2d(32, 64, 3)
		self.conv3 = nn.Conv2d(64, 64, 3)
		# self.bn1 = nn.BatchNorm2d(32)
		# self.bn2 = nn.BatchNorm2d(64)
		# self.bn3 = nn.BatchNorm2d(64)
		self.pool = nn.MaxPool2d(2, 2)
		self.fc1 = nn.Linear(64 * 4 * 4, 64)
		self.fc2 = nn.Linear(64, 10)
	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = F.relu(self.conv3(x))
		x = x.view(-1, 64 * 4 * 4)
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
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
	def __init__(self, block=BasicBlock, num_blocks=[2, 2, 2, 2], num_classes=10, device=None):
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


from torchvision import models
class ResNet18_torch(nn.Module):
	def __init__(self, device=None):
		super().__init__()

		self.resnet = models.resnet18(pretrained=False, num_classes=10)

		self.resnet.conv1 = torch.nn.Conv2d(
			3, 64, kernel_size=3, stride=1, padding=1, bias=False
		)
		self.resnet.maxpool = torch.nn.Identity()

	def forward(self, x):
		x = self.resnet(x)
		x = F.log_softmax(x, dim=1)

		return x



class AlexNet(nn.Module):
	def __init__(self, device=None, num_classes=10):
		super(AlexNet, self).__init__()
		self.features = nn.Sequential(
			nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2),
			nn.Conv2d(64, 192, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2),
			nn.Conv2d(192, 384, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(384, 256, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(256, 256, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2),
		)
		self.classifier = nn.Sequential(
			nn.Dropout(),
			nn.Linear(256 * 2 * 2, 4096),
			nn.ReLU(inplace=True),
			nn.Dropout(),
			nn.Linear(4096, 4096),
			nn.ReLU(inplace=True),
			nn.Linear(4096, num_classes),
		)

	def forward(self, x):
		x = self.features(x)
		x = x.view(x.size(0), 256 * 2 * 2)
		x = self.classifier(x)
		return x




cfg = {
	'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
	'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
	'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
	'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
	def __init__(self, vgg_name, device=None):
		super(VGG, self).__init__()
		self.features = self._make_layers(cfg[vgg_name])
		self.classifier = nn.Linear(512, 10)

	def forward(self, x):
		out = self.features(x)
		out = out.view(out.size(0), -1)
		out = self.classifier(out)
		return out

	def _make_layers(self, cfg):
		layers = []
		in_channels = 3
		for x in cfg:
			if x == 'M':
				layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
			else:
				layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
						   nn.BatchNorm2d(x),
						   nn.ReLU(inplace=True)]
				in_channels = x
		layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
		return nn.Sequential(*layers)


def VGG11(device=None):
	return VGG('VGG11',device=device)


def VGG13(device=None):
	return VGG('VGG13',device=device)


def VGG16(device=None):
	return VGG('VGG16',device=device)


def VGG19(device=None):
	return VGG('VGG19',device=device)

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
		# x = x.permute(1,0,2) # -> (N,W,D)
		# permute during loading the batches instead of in the forward function
		# in order to allow nn.DataParallel

		if not self.args or self.args.static:
			x = Variable(x).to(self.device)

		x = x.unsqueeze(1) # (W,Ci,N,D)

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
		return F.log_softmax(logit, dim=1)
		# return logit

# Sentiment analysis : binary classification
class RNN_IMDB(nn.Module):
	# def __init__(self, embed_num, embed_dim, output_dim, pad_idx):
	def __init__(self, args=None, device=None):
		super(RNN_IMDB, self).__init__()

		self.args = args
		self.device = device
		embed_num = args.embed_num
		embed_dim = args.embed_dim
		output_dim = args.class_num
		pad_idx = args.pad_idx
		
		self.embedding = nn.Embedding(embed_num, embed_dim, padding_idx=pad_idx)
		
		self.fc = nn.Linear(embed_dim, output_dim)
		
	def forward(self, text):
		
		#text = [sent len, batch size]
		embedded = self.embedding(text)		
		pooled = F.avg_pool2d(embedded, (embedded.shape[1], 1)).squeeze(1) 
		
		#pooled = [batch size, embed_dim]
		return F.log_softmax(self.fc(pooled), dim=1)

		