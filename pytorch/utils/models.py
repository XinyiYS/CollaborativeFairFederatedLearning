import torch
import torch.nn as nn
import torch.nn.functional as F


# for MNIST 32*32
class CNN_Net(nn.Module):
    def __init__(self):
        super(CNN_Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, 1)
        self.conv2 = nn.Conv2d(64, 16, 7, 1)
        self.fc1 = nn.Linear(4*4*16, 200)
        self.fc2 = nn.Linear(200, 10)

    def forward(self, x):
        x = x.view(-1, 1, 32, 32)
        x = F.tanh(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.tanh(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*16)
        x = F.tanh(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    
# for MNIST 32*32
class MLP_Net(nn.Module):
    def __init__(self):
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
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        outputs = self.linear(x)
        return outputs

class MLP_LogReg(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP_Adult, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, output_dim)

        # self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)