import torch
from torch import nn, optim

from utils.Worker import Worker
from utils.Data_Prepper import Data_Prepper
from utils.Federated_Learner import Federated_Learner
from utils.models import LogisticRegression, MLP_LogReg, MLP_Net, CNN_Net


use_cuda = True
args = {

    # system parameters
    'device': torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu"),


    # setting parameters
    'dataset': 'mnist',
    'n_workers': 3,
    'balanced': True,
    'sharing_lambda': 0.1,

    # model parameters
    'model_fn': MLP_Net,
    'optimizer_fn': optim.SGD,
    'loss_fn': nn.NLLLoss(),
    'lr': 0.15,

    # training parameters
    'pretrain_epochs': 2,
    'fl_epochs': 2,
    'fl_individual_epochs': 2,
}

if __name__ == '__main__':
    # init steps
    data_prep = Data_Prepper(args['dataset'], train_batch_size=16)
    federated_learner = Federated_Learner(args, data_prep)

    # train
    federated_learner.train()

    # analyze
    federated_learner.get_fairness_analysis()
