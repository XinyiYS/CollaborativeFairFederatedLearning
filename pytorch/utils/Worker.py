import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_value_, clip_grad_norm_
from torchtext.data import Batch
import utils
import torch.nn as nn

class Worker():

	def __init__(self, train_loader, model=None, optimizer=None,scheduler=None,
		model_pretrain=None, optimizer_pretrain=None, pretraining_lr=None, scheduler_pretrain=None,
		standalone_model=None, standalone_optimizer=None, standalone_scheduler=None,
		dssgd_model=None, dssgd_optimizer=None, dssgd_scheduler=None,
		fedavg_model=None, fedavg_optimizer=None, fedavg_scheduler=None,
		loss_fn=None, theta=0.1, grad_clip=0.01, epoch_sample_size=-1,
		device=None,id=None,is_free_rider=False):

		self.train_loader = train_loader
		self.model = model
		self.optimizer = optimizer
		self.scheduler = scheduler
		self.model_pretrain = model_pretrain
		self.optimizer_pretrain = optimizer_pretrain
		self.pretraining_lr = pretraining_lr # specifically for the pretraining period
		self.scheduler_pretrain = scheduler_pretrain
		self.standalone_model = standalone_model
		self.standalone_optimizer = standalone_optimizer
		self.standalone_scheduler = standalone_scheduler
		self.dssgd_model = dssgd_model
		self.dssgd_optimizer = dssgd_optimizer
		self.dssgd_scheduler = dssgd_scheduler
		self.fedavg_model = fedavg_model
		self.fedavg_optimizer = fedavg_optimizer
		self.fedavg_scheduler = fedavg_scheduler
		self.loss_fn = loss_fn
		self.theta = theta
		self.grad_clip = grad_clip		
		self.device = device
		self.id = id
		self.epoch_sample_size = epoch_sample_size
		self.param_count = sum([p.numel() for p in self.model.parameters()])
		self.is_free_rider = is_free_rider

	def train(self, epochs, is_pretrain=False, save_gpu=False):
		if self.is_free_rider:
			for model in [self.model, self.model_pretrain, self.dssgd_model, self.standalone_model]:
				model = model.to(self.device)
	
				for param in model.parameters():
					param.data += (torch.rand(param.data.shape) * 2 - 1).to(self.device) # * self.grad_clip
			return

		self.model_pretrain.train()
		self.model_pretrain = self.model_pretrain.to(self.device)

		self.model.train()
		self.model = self.model.to(self.device)

		self.standalone_model.train()
		self.standalone_model = self.standalone_model.to(self.device)

		self.dssgd_model.train()
		self.dssgd_model = self.dssgd_model.to(self.device)

		self.fedavg_model.train()
		self.fedavg_model = self.fedavg_model.to(self.device)
		for epoch in range(int(epochs)):
			iter = 0
			for i, batch in enumerate(self.train_loader):
				if isinstance(batch, Batch):
					batch_data, batch_target = batch.text, batch.label
					batch_data = batch_data.permute(1, 0)
					# batch_data.data.t_(), batch_target.data.sub_(1)  # batch first, index align
				else:
					batch_data, batch_target = batch[0], batch[1]

				batch_data, batch_target = batch_data.to(self.device), batch_target.to(self.device)

				# introduce separate (and slower) pretraining
				if is_pretrain:
					prev_lrs = []
					if self.pretraining_lr is not None: 
						for g in self.optimizer_pretrain.param_groups:
							prev_lrs.append(g['lr'])
							g['lr'] = self.pretraining_lr

					self.optimizer_pretrain.zero_grad()
					self.loss_fn(self.model_pretrain(batch_data), batch_target).backward()
					self.optimizer_pretrain.step()
					
					# change the lr back so it does not affect FL training
					if self.pretraining_lr is not None: 
						for g, lr in zip(self.optimizer_pretrain.param_groups, prev_lrs):
							g['lr'] = lr
					continue

				iter += len(batch_data)
				for optimizer, model in zip([self.optimizer_pretrain, self.optimizer, self.standalone_optimizer, self.dssgd_optimizer, self.fedavg_optimizer] ,
											[self.model_pretrain, self.model, self.standalone_model, self.dssgd_model, self.fedavg_model]):
					optimizer.zero_grad()
					self.loss_fn(model(batch_data), batch_target).backward()
					optimizer.step()

				if iter >= self.epoch_sample_size:
					# specifically for NLP task to terminate for training efficiency
					break

		if not is_pretrain:
			# NO lr decay during pretraining

			# print()
			# print('standalone' , self.standalone_scheduler.get_last_lr())
			# print('dssgd      ', self.dssgd_scheduler.get_last_lr())
			# print('pretrain   ', self.scheduler_pretrain.get_last_lr())
			# print('no pretrain', self.scheduler.get_last_lr())

			self.standalone_scheduler.step()
			self.scheduler_pretrain.step()
			self.scheduler.step()
			self.dssgd_scheduler.step()
			self.fedavg_scheduler.step()


		if 'cuda' in str(self.device) and save_gpu:
			cpu = torch.device('cpu')
			self.model_pretrain = self.model_pretrain.to(cpu)
			self.model = self.model.to(cpu)
			self.standalone_model = self.standalone_model.to(cpu)
			self.dssgd_model = self.dssgd_model.to(cpu)
			self.fedavg_model = self.fedavg_model.to(cpu)

