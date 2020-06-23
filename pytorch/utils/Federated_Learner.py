import time
import json
import pandas as pd
import numpy as np
import copy
from collections import defaultdict
import math
import torch
from torch import nn, optim

from utils.Data_Prepper import Data_Prepper
from utils.Worker import Worker

from utils.utils import evaluate, averge_models, \
	add_update_to_model, compute_grad_update, compare_models,  \
	add_gradient_updates


class Federated_Learner:

	def __init__(self, args, data_prepper):
		self.args = args
		self.device = args['device']
		self.device_ids = args['device_ids']
		self.save_gpu =  args['save_gpu'] if 'save_gpu' in args else False
		self.data_prepper = data_prepper
		self.n_workers = self.args['n_workers']
		self.n_freeriders = self.args['n_freeriders']

		self.valid_loader = data_prepper.get_valid_loader()
		self.test_loader = data_prepper.get_test_loader()

		self.worker_train_loaders = self.data_prepper.get_train_loaders(
			self.args['n_workers'], self.args['split'])
		self.shard_sizes = torch.tensor(self.data_prepper.shard_sizes).float()
		print("Shard sizes are: ", self.shard_sizes.tolist())
		self.init_workers()
		self.performance_dict = defaultdict(list)
		self.performance_dict_pretrain = defaultdict(list)
		self.time_dict = defaultdict(float)

	def init_workers(self):
		assert self.n_workers == len(
			self.worker_train_loaders), "Num of workers is not equal to num of loaders"
		model_fn = self.args['model_fn']
		optimizer_fn = self.args['optimizer_fn']
		lr = self.args['lr']
		dssgd_lr = self.args['dssgd_lr']
		std_lr = self.args['std_lr'] if 'std_lr' in self.args else dssgd_lr

		device = self.args['device']
		loss_fn = self.args['loss_fn']
		theta = self.args['theta']
		epoch_sample_size = self.args['epoch_sample_size']
		grad_clip = self.args['grad_clip']
		gamma = self.args['gamma']

		if self.data_prepper.name in ['sst', 'mr', 'imdb']:
			self.federated_model = model_fn(args=self.data_prepper.args, device=device)
		else:
			self.federated_model = model_fn(device=device)

		self.load_locked_model_initializations()

		if len(self.args['device_ids']) > 1:
			print("From Federaed Learner - Let's use {} gpus.".format(len(self.args['device_ids'])))
			self.federated_model = 	nn.DataParallel(self.federated_model, device_ids=self.args['device_ids'])

		self.federated_model_pretrain = copy.deepcopy(self.federated_model)

		self.workers = []
		# add in free riders
		if self.n_freeriders > 0:		
			freerider = Worker(train_loader=None,
							model=copy.deepcopy(self.federated_model),
							model_pretrain = copy.deepcopy(self.federated_model),
							standalone_model=copy.deepcopy(self.federated_model),
							dssgd_model=copy.deepcopy(self.federated_model),
							theta=theta,
							device=device,
							is_free_rider=True
							)

			self.workers += [freerider] * self.n_freeriders
			self.n_workers += self.n_freeriders
			self.shard_sizes = torch.cat([torch.zeros(self.n_freeriders), self.shard_sizes])
			# for i in range(self.n_freeriders):
			# 	self.workers.append(freerider)
			# 	self.shard_sizes.insert(0, 0)
			# 	self.n_workers+=1
		
		# possible to enumerate through various model_fns, optimizer_fns, lrs,
		# thetas, or even devices
		for i, worker_train_loader in enumerate(self.worker_train_loaders):
			model = copy.deepcopy(self.federated_model)
			optimizer = optimizer_fn(model.parameters(), lr=lr)
			scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = gamma)


			model_pretrain = copy.deepcopy(self.federated_model)
			optimizer_pretrain = optimizer_fn(model_pretrain.parameters(), lr=lr)
			scheduler_pretrain = torch.optim.lr_scheduler.ExponentialLR(optimizer_pretrain, gamma = gamma)


			standalone_model = copy.deepcopy(self.federated_model)
			standalone_optimizer = optimizer_fn(standalone_model.parameters(), lr=std_lr)
			standalone_scheduler = torch.optim.lr_scheduler.ExponentialLR(standalone_optimizer, gamma = gamma)

			dssgd_model = copy.deepcopy(self.federated_model)
			dssgd_optimizer = optimizer_fn(dssgd_model.parameters(), lr=dssgd_lr)
			# dssgd_optimizer = optimizer_fn(dssgd_model.parameters(), lr=lr)
			# 0.977 ** 100 ~= 0.1    a smaller decay rate
			dssgd_scheduler = torch.optim.lr_scheduler.ExponentialLR(dssgd_optimizer, gamma = gamma)

			worker = Worker(train_loader=worker_train_loader,
							model=model, optimizer=optimizer, scheduler=scheduler,
							model_pretrain=model_pretrain, optimizer_pretrain=optimizer_pretrain,scheduler_pretrain=scheduler_pretrain,
							pretraining_lr=self.args['pretraining_lr'],

							standalone_model=standalone_model, standalone_optimizer=standalone_optimizer, standalone_scheduler=standalone_scheduler,
							dssgd_model=dssgd_model, dssgd_optimizer=dssgd_optimizer,dssgd_scheduler=dssgd_scheduler,
							loss_fn=loss_fn, theta=theta,
							grad_clip=grad_clip, epoch_sample_size=epoch_sample_size,
							device=device,
							id=i,
							)
			self.workers.append(worker)
		return

	def train_locally(self, epochs, is_pretrain=False, save_gpu=False):

		if is_pretrain:
			for i, worker in enumerate(self.workers):
				worker.train(epochs, is_pretrain=is_pretrain)
			return

		self.filtered_updates = []
		self.filtered_updates_pretrain = []

		self.aggregated_gradient_updates = [torch.zeros(param.shape).to(self.device) for param in self.federated_model.parameters()]
		self.aggregated_gradient_updates_pretrain = [torch.zeros(param.shape).to(self.device) for param in self.federated_model.parameters()]

		worker_val_accs = []
		worker_val_accs_pretrain = []
		dssgd_val_accs = []

		for i, worker in enumerate(self.workers):
			self.timestamp = time.time()

			model_before = copy.deepcopy(worker.model)
			dssgd_model_before = copy.deepcopy(worker.dssgd_model)
			model_pretrain_before = copy.deepcopy(worker.model_pretrain)

			worker.train(epochs, is_pretrain=is_pretrain, save_gpu=save_gpu)
			model_after = copy.deepcopy(worker.model)
			dssgd_model_after = copy.deepcopy(worker.dssgd_model)
			model_pretrain_after = copy.deepcopy(worker.model_pretrain)

			self.clock('workers local training')

			# recover the model before training for clipped grad update later
			worker.model.load_state_dict(model_before.state_dict())

			raw_grad_update = compute_grad_update(model_before, model_after, device=self.device)
			del model_before, model_after  # to free up memory immediately


			'''
			# clipped stats
			columns = ['','grad_mean','clipped_count', 'clipped_ratio']
			data_rows = []

			all_update_mod = torch.cat([update.data.view(-1).abs()for update in raw_grad_update])
			n_clipped = (all_update_mod > self.args['grad_clip']).sum().item()
			data_rows.append([ 'w/o pretrain: ', all_update_mod.mean().item(), n_clipped, torch.true_divide(n_clipped, len(all_update_mod)).item() ])
			'''

			# directly add the raw gradient to the model
			add_update_to_model(worker.model, raw_grad_update, device=self.device)

			clipped_grad_update = clip_gradient_update(raw_grad_update, self.args['grad_clip'])
			# add the clipped grad to local model
			# add_update_to_model(worker.model, clipped_grad_update, device=self.device)

			filtered_grad_update = mask_grad_update_by_order(clipped_grad_update, mask_order=None, mask_percentile=worker.theta, mode=self.args['largest_criterion']) 


			fed_val_acc = self.one_on_one_evaluate(self.federated_model, worker.model, filtered_grad_update, worker.theta)
			worker_val_accs.append(fed_val_acc)

			# minus the uploaded grad updates
			# add_update_to_model(worker.model, filtered_grad_update, weight= -1.0)
			
			# register this filtered_updates for later to removed
			# NOTE that we do not minus this update because this worker may not be reputable 
			# after evaluation, meaning it does not receive allocated_grad, so no need to minus its own
			self.filtered_updates.append(filtered_grad_update)

			self.clock('gradient clipping and filtering')

			# for with pretraining

			worker.model_pretrain.load_state_dict(model_pretrain_before.state_dict())
			raw_grad_update = compute_grad_update(model_pretrain_before, model_pretrain_after, device=self.device)				
			del model_pretrain_before, model_pretrain_after

			'''
			# clipped stats
			all_update_mod = torch.cat([update.data.view(-1).abs()for update in raw_grad_update])
			n_clipped = (all_update_mod > self.args['grad_clip']).sum().item()
			data_rows.append([ 'w pretrain: ', all_update_mod.mean().item(), n_clipped, torch.true_divide(n_clipped, len(all_update_mod)).item() ])
			'''
			add_update_to_model(worker.model_pretrain, raw_grad_update, device=self.device)

			clipped_grad_update = clip_gradient_update(raw_grad_update, self.args['grad_clip'])
			# add_update_to_model(worker.model_pretrain, clipped_grad_update, device=self.device)

			filtered_grad_update = mask_grad_update_by_order(clipped_grad_update, mask_order=None, mask_percentile=worker.theta, mode=self.args['largest_criterion']) 
			

			fed_val_acc = self.one_on_one_evaluate(self.federated_model_pretrain, worker.model_pretrain, filtered_grad_update, worker.theta, is_pretrain=True)
			worker_val_accs_pretrain.append(fed_val_acc)

			# minus the uploaded grad updates
			# add_update_to_model(worker.model_pretrain, filtered_grad_update, weight= -1.0)
			self.filtered_updates_pretrain.append(filtered_grad_update)
			
			self.clock('gradient clipping and filtering for pretrain')


			# for DSSGD model

			dssgd_grad_update = compute_grad_update(dssgd_model_before, dssgd_model_after, device=self.device)
			del dssgd_model_before, dssgd_model_after  # to free up memory immediately

			'''
			# clipped stats
			all_update_mod = torch.cat([update.data.view(-1).abs()for update in dssgd_grad_update])
			n_clipped = (all_update_mod > self.args['grad_clip']).sum().item()
			data_rows.append([ 'dssgd:', all_update_mod.mean().item(), n_clipped, torch.true_divide(n_clipped, len(all_update_mod)).item() ])
			'''

			filtered_grad_update = mask_grad_update_by_order(clip_gradient_update(dssgd_grad_update, self.args['grad_clip']), mask_order=None, mask_percentile=worker.theta, mode=self.args['largest_criterion'])

			# this is executed in a fixed sequence, so the self.dssgd_model gets gradually updated and 'downloaded' by each worker
			worker.dssgd_model.load_state_dict(add_update_to_model(self.dssgd_model, filtered_grad_update).state_dict(), strict=False)
			dssgd_val_acc = evaluate(worker.dssgd_model, self.valid_loader, self.device, verbose=False)[1]
			dssgd_val_accs.append(dssgd_val_acc)

			self.clock('server aggregation dssgd')

			'''
			# clipped stats
			print(pd.DataFrame(data=data_rows, columns=columns))
			'''

		return worker_val_accs, worker_val_accs_pretrain, dssgd_val_accs


	def train(self):

		self.credits = torch.zeros((self.n_workers))
		self.credits_pretrain = torch.zeros((self.n_workers))

		self.credit_threshold_coef = self.args['credit_threshold_coef'] if 'credit_threshold_coef' in self.args else 1.0/3.0
		# init the credit_th to be a 2/3 * 1/(len(R)) instead of 0
		self.credit_threshold = compute_credit_threshold(self.n_workers,self.args['split'], self.credit_threshold_coef)
		self.credit_threshold_pretrain = compute_credit_threshold(self.n_workers,self.args['split'], self.credit_threshold_coef)

		self.R = list(range(self.n_workers))
		self.R_pretrain = list(range(self.n_workers))

		fl_epochs = self.args['fl_epochs']
		device = self.args['device']
		fl_individual_epochs = self.args['fl_individual_epochs']

		self.performance_dict['shard_sizes'] = self.shard_sizes.tolist()
		self.performance_dict_pretrain['shard_sizes'] = self.shard_sizes.tolist()

		# print("Start local pretraining ")
		self.timestamp = time.time()

		self.train_locally(self.args['pretrain_epochs'], is_pretrain=True, save_gpu=self.save_gpu)

		self.clock('pretraining')


		self.worker_model_test_accs_before = self.evaluate_workers_performance(self.test_loader)
		self.performance_dict['worker_model_test_accs_before'] = self.worker_model_test_accs_before

		self.worker_model_test_accs_before_w_pretrain = self.evaluate_workers_performance(self.test_loader, mode='pretrain')
		self.performance_dict_pretrain['worker_model_test_accs_before'] = self.worker_model_test_accs_before_w_pretrain

		self.dssgd_model = copy.deepcopy(self.federated_model).to(device)
		# each worker needs a dssgd model to compute final fairness

		_, federated_val_acc = evaluate(
			self.federated_model, self.valid_loader, device, verbose=False)
		print("CFFL server model validation accuracy : {:.4%}".format(federated_val_acc))

		_, federated_test_acc = evaluate(
			self.federated_model, self.test_loader, device, verbose=False)
		print("CFFL server model test accuracy : {:.4%}".format(federated_test_acc))


		self.clock('evaluation after pretraining')

		# print("\nStart federated learning \n")
		for epoch in range(fl_epochs):


			# 1. training locally
			worker_val_accs, worker_val_accs_pretrain, dssgd_val_accs = self.train_locally(fl_individual_epochs,save_gpu=self.save_gpu)

			# 2. update the credits and credit_threshold
			# and update the reputable parties set
			self.credits, self.credit_threshold, self.R  = compute_credits_sinh(self.credits, self.credit_threshold, self.R, worker_val_accs, alpha=self.args['alpha'], split=self.args['split'], credit_threshold_coef=self.credit_threshold_coef)
			self.credits_pretrain, self.credit_threshold_pretrain, self.R_pretrain = compute_credits_sinh(self.credits_pretrain, self.credit_threshold_pretrain, self.R_pretrain, worker_val_accs_pretrain, alpha=self.args['alpha'],split=self.args['split'],credit_threshold_coef=self.credit_threshold_coef)

			self.clock('credit updates')


			# 3. aggregate the gradients and update the federated model
			self.aggregate_gradients_and_update_federated_model()
			self.clock('aggregate gradients and update FL model')


			# 4. gradient downloads and uploads according to credits and thetas
			self.assign_updates_with_filter()
			self.clock('assign updates')


			# update the performance dict as log
			if epoch % 20 == 0:
				print()
				print('Epoch {}:'.format(epoch + 1))
				print("credits, credit threshold:", self.credit_threshold)
				print(self.credits)
				print("Reputable parties: ", self.R)
				print()
				print("credits, credit_threshold pretrain:", self.credit_threshold_pretrain)
				print(self.credits_pretrain)
				print("Reputable parties pretrain: ", self.R_pretrain)


			self.performance_summary(to_print=(epoch%20==0))


			self.performance_dict['dssgd_val_accs'].append(dssgd_val_accs)
			self.performance_dict_pretrain['dssgd_val_accs'].append(dssgd_val_accs)

			self.performance_dict['credits'].append(self.credits)
			self.performance_dict['credit_threshold'].append(self.credit_threshold)

			self.performance_dict_pretrain['credits'].append(self.credits_pretrain)
			self.performance_dict_pretrain['credit_threshold'].append(self.credit_threshold_pretrain)
			# print()
			self.clock('performance update')

		total_seconds = 0
		for key, value in self.time_dict.items():
			print(key, value)
			total_seconds+= value
		print("total seconds", total_seconds)
		print('Time_dict-----')
		print(json.dumps(self.time_dict))
		print('-----')

		self.convert_tensors_in_dicts()
		return

	def load_locked_model_initializations(self, dirname='initialized_models'):
		import os
		models_dir = os.path.join(dirname, self.args['dataset'])
		os.makedirs(models_dir, exist_ok=True)
		model_name =  self.federated_model.__class__.__name__
		model_path = os.path.join(models_dir, model_name)
		if os.path.isfile(model_path):
			try:
				self.federated_model.load_state_dict(torch.load(model_path))
				print("Successfully loaded the previously initialzied {} for {}".format(model_name, self.args['dataset']))
			except Exception as e:
				print(str(e))
				print("Due to the above error, saving {} to overwrite the existing file.".format(model_name))
				torch.save(self.federated_model.state_dict(), model_path)
		else:
			print("Saving a fresh {} model for {}.".format(model_name, self.args['dataset']))
			torch.save(self.federated_model.state_dict(), model_path)


	def one_on_one_evaluate(self, federated_model, worker_model, filtered_grad_update, theta, is_pretrain=False):
		if theta == 1 and not is_pretrain:
			fed_val_acc = evaluate(worker_model, self.valid_loader, self.device, verbose=False)[1]
		else:
			model_to_eval = copy.deepcopy(federated_model)
			add_update_to_model(model_to_eval, filtered_grad_update, device=self.device)
			fed_val_acc = evaluate(model_to_eval, self.valid_loader, self.device, verbose=False)[1]
			del model_to_eval
		return fed_val_acc

	def aggregate_gradients_and_update_federated_model(self, eta=1):
		self.aggregated_gradient_updates = [torch.zeros(param.shape).to(self.device) for param in self.federated_model.parameters()]
		self.aggregated_gradient_updates_pretrain = [torch.zeros(param.shape).to(self.device) for param in self.federated_model.parameters()]

		for i in self.R:
			filtered_grad_update = self.filtered_updates[i]
			if self.args['aggregate_mode'] == 'sum':
				weight = 1.0
			elif self.args['aggregate_mode'] == 'credit-sum':
				weight = self.credits[i]
			else: # default average
				weight = self.shard_sizes[i] * 1. / sum(self.shard_sizes)
			add_gradient_updates(self.aggregated_gradient_updates, filtered_grad_update, weight)

		add_update_to_model(self.federated_model, self.aggregated_gradient_updates, weight=eta, device=self.device)
		# self.federated_val_acc = evaluate(self.federated_model, self.valid_loader, device=self.device, verbose=False)[1]

		for i in self.R_pretrain:
			filtered_grad_update = self.filtered_updates_pretrain[i]
			if self.args['aggregate_mode'] == 'sum':
				weight = 1.0
			elif self.args['aggregate_mode'] == 'credit-sum':
				weight = self.credits_pretrain[i]
			else: # default fedavg
				weight = self.shard_sizes[i] *1. / sum(self.shard_sizes)
			add_gradient_updates(self.aggregated_gradient_updates_pretrain, filtered_grad_update, weight)

		add_update_to_model(self.federated_model_pretrain, self.aggregated_gradient_updates_pretrain, weight=eta, device=self.device)
		# self.federated_val_acc_pretrain = evaluate(self.federated_model_pretrain, self.valid_loader, device=self.device, verbose=False)[1]

	def assign_updates_with_filter(self):
		"""
		download the largest magnitude updates <credits[i] * num_param> from the server
		and filter out its own updates in the local model
		and apply to its local model
		"""
		if self.args['aggregate_mode'] == 'mean':
			weights = torch.div(self.shard_sizes , sum(self.shard_sizes) ) # fed_avg
		else:
			weights = torch.ones(self.n_workers)  

		# default download mode is 'topk'
		download = 'topk' if 'download' not in self.args else self.args['download']

		if self.args['largest_criterion'] == 'all':

			# preprocess to get the topk largest values for all (only need sort it once for the highest credit)
			# no pretrain
			absolute_values = torch.cat([update.data.view(-1).abs() for update in self.aggregated_gradient_updates])

			if download == 'random':
				random_permuted_indices = torch.randperm(len(absolute_values))
			else:
				topk, _ = torch.topk(absolute_values, int(len(absolute_values) * max(self.credits)))

			# pretrain
			absolute_values = torch.cat([update.data.view(-1).abs() for update in self.aggregated_gradient_updates_pretrain])
			if download == 'random':
				random_permuted_pretrain_indices = torch.randperm(len(absolute_values)) if download == 'random' else None
			else:
				topk_pretrain, _ = torch.topk(absolute_values, int(len(absolute_values) * max(self.credits_pretrain)))
				del _

			del absolute_values
			
			for i, worker in enumerate(self.workers):

				# no pretrain
				if i in self.R:
					agg_grad_update = copy.deepcopy(self.aggregated_gradient_updates)
					
					# num_downloads  = int(self.credits[i] * worker.param_count)
					# NEW LOGIC FOR determining how many parameters to download
					num_downloads  = int(self.credits[i]*1. / max(self.credits) *self.shard_sizes[i] *1. / max(self.shard_sizes) * worker.param_count)
					# print("worker {}, no pretrain, download quota {}".format(i, num_downloads))
					# print("total random indices len {}, download quota {}, zero count shoud be {}".format(len(random_permuted_indices), num_downloads, worker.param_count - num_downloads ))
					if download == 'random':
						assert random_permuted_indices is not None, "Uninitialized <random_permuted_indices>"
						allocated_grad = mask_grad_update_by_indices(agg_grad_update, indices=random_permuted_indices[:num_downloads])
					else:
						allocated_grad = mask_grad_update_by_magnitude(agg_grad_update, topk[num_downloads-1])
					
					add_update_to_model(worker.model, allocated_grad)
					add_update_to_model(worker.model, self.filtered_updates[i], weight=-weights[i])
					
				# with pretrain
				if i in self.R_pretrain:
					agg_grad_update = copy.deepcopy(self.aggregated_gradient_updates_pretrain)
					# num_downloads  = int(self.credits_pretrain[i] * worker.param_count)
					
					num_downloads  = int(self.credits[i]*1. / max(self.credits) *self.shard_sizes[i] *1. / max(self.shard_sizes) * worker.param_count)
					if download == 'random':
						assert random_permuted_pretrain_indices is not None, "Uninitialized <random_permuted_pretrain_indices>"
						allocated_grad = mask_grad_update_by_indices(agg_grad_update, indices=random_permuted_pretrain_indices[:num_downloads])

					else:				
						allocated_grad = mask_grad_update_by_magnitude(agg_grad_update, topk_pretrain[num_downloads-1])

					add_update_to_model(worker.model_pretrain, allocated_grad)
					add_update_to_model(worker.model_pretrain, self.filtered_updates_pretrain[i], weight=-weights[i])

		elif self.args['largest_criterion'] == 'layer':
			
			for i, worker in enumerate(self.workers):
				# no pretrain
				if i in self.R:
					allocated_grad = mask_grad_update_by_order(self.aggregated_gradient_updates, mask_order=None, mask_percentile=self.credits[i], mode='layer')
					add_update_to_model(worker.model, allocated_grad)
					add_update_to_model(worker.model, self.filtered_updates[i], weight=-weights[i])

				# with pretrain
				if i in self.R_pretrain:
					allocated_grad = mask_grad_update_by_order(self.aggregated_gradient_updates_pretrain, mask_order=None, mask_percentile=self.credits_pretrain[i], mode='layer')
					add_update_to_model(worker.model_pretrain, allocated_grad)
					add_update_to_model(worker.model_pretrain, self.filtered_updates_pretrain[i], weight=-weights[i])
		return

	def performance_summary(self, to_print=False):
		self.dssgd_models_test_accs = self.evaluate_workers_performance(self.test_loader, mode='dssgd')
		self.worker_standalone_test_accs = self.evaluate_workers_performance(self.test_loader, mode='standalone')
		self.cffl_test_accs = self.evaluate_workers_performance(self.test_loader)
		self.cffl_test_accs_w_pretrain = self.evaluate_workers_performance(self.test_loader, mode='pretrain')

		# no pretrain
		self.performance_dict['DSSGD_model_test_accs'].append(self.dssgd_models_test_accs)
		self.performance_dict['worker_standalone_test_accs'].append(self.worker_standalone_test_accs)
		self.performance_dict['cffl_test_accs'].append(self.cffl_test_accs)

		# with pretrain
		self.performance_dict_pretrain['DSSGD_model_test_accs'].append(self.dssgd_models_test_accs)
		self.performance_dict_pretrain['worker_standalone_test_accs'].append(self.worker_standalone_test_accs)
		self.performance_dict_pretrain['cffl_test_accs'].append(self.cffl_test_accs_w_pretrain)

		if to_print:
			print('Below are testset accuracies: ---')
			print('Workers standalone accuracies: ', ["{:.3%}".format(acc_std) for acc_std in self.worker_standalone_test_accs])
			print('Workers DSSGD     accuracies: ', ["{:.3%}".format(dssgd_acc) for dssgd_acc in self.dssgd_models_test_accs])
			print()
			print('Workers before    accuracies: ', ["{:.3%}".format(acc_b4) for acc_b4 in self.worker_model_test_accs_before])
			print('Workers CFFL      accuracies: ', ["{:.3%}".format(acc_aft) for acc_aft in self.cffl_test_accs])
			print()
			print('Workers before pr accuracies: ', ["{:.3%}".format(acc_b4) for acc_b4 in self.worker_model_test_accs_before_w_pretrain])
			print('Workers CFFL pret accuracies: ', ["{:.3%}".format(acc_aft) for acc_aft in self.cffl_test_accs_w_pretrain])

		return

	def convert_tensors_in_dicts(self):

		for key, value in self.performance_dict.items():
			if isinstance(value[0], torch.Tensor):
				self.performance_dict[key] = torch.stack(value).tolist()
			elif isinstance(value[0], list):
				self.performance_dict[key] = [torch.stack(v).tolist() for v in value]


		for key, value in self.performance_dict_pretrain.items():
			if isinstance(value[0], torch.Tensor):
				self.performance_dict_pretrain[key] = torch.stack(value).tolist()
			elif isinstance(value[0], list):
				self.performance_dict_pretrain[key] = [torch.stack(v).tolist() for v in value]

	def get_fairness_analysis(self):
		print("Performance and Fairness analysis: ")
		worker_thetas = [worker.theta for worker in self.workers]
	
		print('Worker credits :', self.credits.tolist())
		print('Number of reputable parties: ', len(self.R))

		print('Worker credits pretrain:', self.credits_pretrain.tolist())
		print('Number of reputable parties with pretrain: ', len(self.R_pretrain))


		# no pretrain
		worker_standalone_test_accs = self.performance_dict['worker_standalone_test_accs'][-1]
		DSSGD_model_test_accs = self.performance_dict['DSSGD_model_test_accs'][-1]
		cffl_test_accs = self.performance_dict['cffl_test_accs'][-1]

		from scipy.stats import pearsonr
		corrs = pearsonr(worker_standalone_test_accs, DSSGD_model_test_accs)
		self.performance_dict['standalone_vs_rrdssgd'].append(corrs[0])

		corrs = pearsonr(worker_standalone_test_accs, cffl_test_accs)
		self.performance_dict['standalone_vs_final'].append(corrs[0])

		self.performance_dict['CFFL_best_worker'] = max(cffl_test_accs)
		best_worker_id = np.argmax(cffl_test_accs)
		self.performance_dict['standalone_best_worker'] = worker_standalone_test_accs[best_worker_id]
		self.performance_dict['rr_dssgd_best'] = DSSGD_model_test_accs[best_worker_id]


		# with pretrain
		worker_standalone_test_accs = self.performance_dict_pretrain['worker_standalone_test_accs'][-1]
		DSSGD_model_test_accs = self.performance_dict_pretrain['DSSGD_model_test_accs'][-1]
		cffl_test_accs = self.performance_dict_pretrain['cffl_test_accs'][-1]

		corrs = pearsonr(worker_standalone_test_accs, DSSGD_model_test_accs)
		self.performance_dict_pretrain['standalone_vs_rrdssgd'].append(corrs[0])

		corrs = pearsonr(worker_standalone_test_accs, cffl_test_accs)
		self.performance_dict_pretrain['standalone_vs_final'].append(corrs[0])

		self.performance_dict_pretrain['CFFL_best_worker'] = max(cffl_test_accs)
		best_worker_id = np.argmax(cffl_test_accs)

		self.performance_dict_pretrain['standalone_best_worker'] = worker_standalone_test_accs[best_worker_id]
		self.performance_dict_pretrain['rr_dssgd_best'] = DSSGD_model_test_accs[best_worker_id]


		keys = ['standalone_best_worker', 'CFFL_best_worker', 'rr_dssgd_best', 'standalone_vs_rrdssgd', 'standalone_vs_final']
		print("----Results without pretrain")
		for key in keys:
			print(key, ' - ', self.performance_dict[key])			
		print("----")

		print("----Results with pretrain")
		for key in keys:
			print(key, ' - ', self.performance_dict_pretrain[key])
		print("----")

		return

	def evaluate_workers_performance(self, eval_loader, mode=None):
		device = self.args['device']
		if mode == 'standalone':
			return [evaluate(worker.standalone_model, eval_loader, device, verbose=False)[1] for worker in self.workers]
		elif mode == 'dssgd':
			return [evaluate(worker.dssgd_model, eval_loader, device, verbose=False)[1] for worker in self.workers]
		elif mode == 'pretrain':
			return [evaluate(worker.model_pretrain, eval_loader, device, verbose=False)[1] for worker in self.workers]
		else:
			return [evaluate(worker.model, eval_loader, device, verbose=False)[1] for worker in self.workers]
	def clock(self, key):
		self.timestamp_  = time.time()
		self.time_dict[key] += self.timestamp_ - self.timestamp
		self.timestamp = self.timestamp_


def compute_credits_sinh(credits, credit_threshold, R, val_accs, alpha=5, credit_fade=1, split='powerlaw'):
	# print('alpha used is :', alpha, ' current credits are : ', credits, ' current threshold: ', credit_threshold)
	R_size = len(R)
	total_val_accs = sum([val_accs[i] for i in R])
	for i in R:
		credit_epoch = val_accs[i] / total_val_accs
		# credit_epoch = math.sinh(alpha * val_accs[i] / total_val_accs)

		if credit_fade == 1:
			credits[i] = credits[i] * 0.2 + credit_epoch * 0.8
		else:
			credits[i] = (credits[i] + credit_epoch) * 0.5

	credits = torch.sinh(alpha * credits)

	# normalize among the reputable parties
	credits /= credits.sum().float()

	# update reputable parties
	R = [i for i in R if credits[i] >= credit_threshold]

	# isolate the non-reputable parties by setting their credits to 0	
	for i in range(len(credits)):
		if credits[i] < credit_threshold:
			credits[i] = 0

	if R_size != len(R):
		# normalize among the reputable parties
		credits /= credits.sum().float()
		credit_threshold = compute_credit_threshold(len(R), split, credit_threshold_coef)

		print("old R size : {}, new R size: {}".format(R_size, len(R)))
		print("new credit_threshold {}, credits {}".format(credit_threshold.item(), credits.tolist()))

	return credits, credit_threshold, R

def clip_gradient_update(grad_update, grad_clip):
	"""
	Return a copy of clipped grad update 

	"""
	return [torch.clamp(param.data, min=-grad_clip, max=grad_clip) for param in grad_update]


def mask_grad_update_by_order(grad_update, mask_order, mask_percentile=None, mode='all'):

	if mode == 'all':
		# mask all but the largest <mask_order> updates (by magnitude) to zero
		all_update_mod = torch.cat([update.data.view(-1).abs()
									for update in grad_update])
		if not mask_order and mask_percentile:
			mask_order = int(len(all_update_mod) * mask_percentile)
		
		if mask_order == 0:
			return mask_grad_update_by_magnitude(grad_update, float('inf'))
		else:
			topk, indices = torch.topk(all_update_mod, mask_order)
			return mask_grad_update_by_magnitude(grad_update, topk[-1])

	elif mode == 'layer': # layer wise largest-values criterion
		grad_update = copy.deepcopy(grad_update)

		for i, layer in enumerate(grad_update):
			layer_mod = layer.data.view(-1).abs()
			if mask_percentile:
				mask_order = math.ceil(len(layer_mod) * mask_percentile)

			if mask_order == 0:
				grad_update[i].data = torch.zeros(layer.data.shape, device=layer.device)
			else:
				topk, indices = torch.topk(layer_mod, min(mask_order, len(layer_mod)-1))
				grad_update[i].data[layer.data.abs() < topk[-1]] = 0
		return grad_update

def mask_grad_update_by_magnitude(grad_update, mask_constant):

	# mask all but the updates with larger magnitude than <mask_constant> to zero
	# print('Masking all gradient updates with magnitude smaller than ', mask_constant)
	grad_update = copy.deepcopy(grad_update)
	for i, update in enumerate(grad_update):
		grad_update[i].data[update.data.abs() < mask_constant] = 0
	return grad_update

def mask_grad_update_by_indices(grad_update, indices=None):
	"""
	Mask the grad.data to be 0, if the position is not in the list of indices
	If indicies is empty, mask nothing.
	
	Arguments: 
	grad_update: as in the shape of the model parameters. A list of tensors.
	indices: a tensor of integers, corresponding to the specific individual scalar values in the grad_update, 
	as if the entire grad_update is flattened.

	e.g. 
	grad_update = [[1, 2, 3], [3, 2, 1]]
	indices = [4, 5]
	returning masked grad_update = [[0, 0, 0], [0, 2, 1]]
	"""

	grad_update = copy.deepcopy(grad_update)
	if indices is None or len(indices)==0: return grad_update

	#flatten and unflatten
	flattened = torch.cat([update.data.view(-1) for update in grad_update])	
	masked = torch.zeros_like(torch.arange(len(flattened)), device=flattened.device).float()
	masked.data[indices] = flattened.data[indices]

	pointer = 0
	for m, update in enumerate(grad_update):
		size_of_update = torch.prod(torch.tensor(update.shape)).long()
		grad_update[m].data = masked[pointer: pointer + size_of_update].reshape(update.shape)
		pointer += size_of_update
	return grad_update

def compute_credit_threshold(R_size, split, coef=1.0/3.0):
	if split=='classimbalance':
		return torch.clamp(1./6 * torch.div(1., R_size), min=0, max=1).float()
	else:
		return torch.clamp(coef * torch.div(1., R_size), min=0, max=1).float()
