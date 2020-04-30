import numpy as np
import copy
from collections import defaultdict
import math
import torch
from torch import nn, optim

from utils.Data_Prepper import Data_Prepper
from utils.Worker import Worker

from utils.utils import evaluate, averge_models, aggregate_gradient_updates, \
	add_update_to_model, compute_grad_update, compare_models,  \
	leave_one_out_evaluate, one_on_one_evaluate, compute_shapley


class Federated_Learner:

	def __init__(self, args, data_prepper):
		self.args = args
		self.device = args['device']
		self.data_prepper = data_prepper
		self.n_workers = self.args['n_workers']
		self.n_freeriders = self.args['n_freeriders']

		self.valid_loader = data_prepper.get_valid_loader()
		self.test_loader = data_prepper.get_test_loader()

		self.worker_train_loaders = self.data_prepper.get_train_loaders(
			self.args['n_workers'], self.args['split'])
		self.shard_sizes = [len(index_list) for index_list in data_prepper.indices_list]
		self.init_workers()
		self.performance_dict = defaultdict(list)
		self.performance_dict_pretrain = defaultdict(list)

	def init_workers(self):
		assert self.n_workers == len(
			self.worker_train_loaders), "Num of workers is not equal to num of loaders"
		model_fn = self.args['model_fn']
		optimizer_fn = self.args['optimizer_fn']
		lr = self.args['lr']
		device = self.args['device']
		loss_fn = self.args['loss_fn']
		theta = self.args['theta']
		epoch_sample_size = self.args['epoch_sample_size']
		grad_clip = self.args['grad_clip']

		self.federated_model = model_fn(device=device)
		self.federated_model_pretrain = copy.deepcopy(self.federated_model)

		self.workers = []
		# add in free riders
		
		freerider = Worker(train_loader=None,
						model=copy.deepcopy(self.federated_model),
						model_pretrain = copy.deepcopy(self.federated_model),
						standalone_model=copy.deepcopy(self.federated_model),
						dssgd_model=copy.deepcopy(self.federated_model),
						device=device,
						is_free_rider=True
						)
		for i in range(self.n_freeriders):
			self.workers.append(freerider)
			self.shard_sizes.insert(0, 0)
			self.n_workers+=1
		
		# possible to enumerate through various model_fns, optimizer_fns, lrs,
		# thetas, or even devices
		for i, worker_train_loader in enumerate(self.worker_train_loaders):
			model = copy.deepcopy(self.federated_model)
			optimizer = optimizer_fn(model.parameters(), lr=lr)

			model_pretrain = copy.deepcopy(self.federated_model)
			optimizer_pretrain = optimizer_fn(model_pretrain.parameters(), lr=lr)

			standalone_model = copy.deepcopy(self.federated_model)
			standalone_optimizer = optimizer_fn(
				standalone_model.parameters(), lr=lr)

			dssgd_model = copy.deepcopy(self.federated_model)
			dssgd_optimizer = optimizer_fn(dssgd_model.parameters(), lr=lr)

			worker = Worker(train_loader=worker_train_loader,
							model=model, optimizer=optimizer,
							model_pretrain=model_pretrain, optimizer_pretrain=optimizer_pretrain,
							standalone_model=standalone_model, standalone_optimizer=standalone_optimizer,
							dssgd_model=dssgd_model, dssgd_optimizer=dssgd_optimizer,
							loss_fn=loss_fn, theta=theta,
							grad_clip=grad_clip, epoch_sample_size=epoch_sample_size,
							device=device,
							id=i,
							)
			self.workers.append(worker)
		return

	def train_locally(self, epochs, requires_update=False, test=False, is_pretrain=False):
		# requires grad_updates
		if requires_update:
			grad_updates = []
			grad_updates_pretrain = []
			dssgd_grad_updates = []

			for worker in self.workers:
				model_before = copy.deepcopy(worker.model)
				dssgd_model_before = copy.deepcopy(worker.dssgd_model)
				model_pretrain_before = copy.deepcopy(worker.model_pretrain)

				worker.train(epochs, is_pretrain=is_pretrain)
				model_after = copy.deepcopy(worker.model)
				dssgd_model_after = copy.deepcopy(worker.dssgd_model)
				model_pretrain_after = copy.deepcopy(worker.model_pretrain)

				grad_updates.append(compute_grad_update(
					model_before, model_after, device=self.device))
				# recover the model before training for clipped grad update later
				worker.model.load_state_dict(model_before.state_dict())
				del model_before, model_after  # to free up memory immediately

				grad_updates_pretrain.append(compute_grad_update(model_pretrain_before, 
					model_pretrain_after, device=self.device))
				del model_pretrain_before, model_pretrain_after

				dssgd_grad_updates.append(compute_grad_update(
					dssgd_model_before, dssgd_model_after, device=self.device))
				
				# NOT necessary for dssgd because the worker.dssgd_model loads the full global dssgd_model later
				# recover the model before training for clipped grad update later
				# worker.dssgd_model.load_state_dict(dssgd_model_before.state_dict())
				
				del dssgd_model_before, dssgd_model_after  # to free up memory immediately

				if test:
					evaluate(worker.model, self.test_loader, worker.device)
			return grad_updates, grad_updates_pretrain, dssgd_grad_updates
		else:
			for worker in self.workers:
				worker.train(epochs=epochs, is_pretrain=is_pretrain)
				if test:
					evaluate(worker.model, self.test_loader, worker.device)
		return

	def train(self):
		points = torch.tensor(
			[worker.theta * worker.param_count * (self.n_workers - 1) for worker in self.workers])
		credits = torch.zeros((self.n_workers))
		credit_threshold = 0
		
		credits_pretrain = torch.zeros((self.n_workers))
		credit_threshold_pretrain = 0

		worker_thetas = [worker.theta for worker in self.workers]

		fl_epochs = self.args['fl_epochs']
		device = self.args['device']
		fl_individual_epochs = self.args['fl_individual_epochs']
		aggregate_mode = self.args['aggregate_mode']

		self.performance_dict['shard_sizes'] = self.shard_sizes
		self.performance_dict_pretrain['shard_sizes'] = self.shard_sizes


		# print("Start local pretraining ")
		self.train_locally(self.args['pretrain_epochs'], is_pretrain=True)
		self.worker_model_test_accs_before = self.evaluate_workers_performance(
			self.test_loader)
		self.performance_dict[
			'worker_model_test_accs_before'] = self.worker_model_test_accs_before


		self.worker_model_test_accs_before_w_pretrain = self.evaluate_workers_performance(
			self.test_loader, mode='pretrain')
		self.performance_dict_pretrain['worker_model_test_accs_before'] = self.worker_model_test_accs_before_w_pretrain

		self.dssgd_model = copy.deepcopy(self.federated_model).to(device)
		# each worker needs a dssgd model to compute final fairness
		double_seqs = list(range(self.n_workers)) + list(range(self.n_workers))

		_, federated_val_acc = evaluate(
			self.federated_model, self.valid_loader, device, verbose=False)
		print("CFFL server model validation accuracy : {:.4%}".format(federated_val_acc))

		_, federated_test_acc = evaluate(
			self.federated_model, self.test_loader, device, verbose=False)
		print("CFFL server model test accuracy : {:.4%}".format(federated_test_acc))


		# print("\nStart federated learning \n")
		for epoch in range(fl_epochs):


			# 1. training locally, return updates, clip and filter the updates
			grad_updates, grad_updates_pretrain, dssgd_grad_updates = self.train_locally(
				fl_individual_epochs, requires_update=True)
			clip_gradients_(grad_updates, self.args['grad_clip'])
			clip_gradients_(grad_updates_pretrain, self.args['grad_clip'])
			clip_gradients_(dssgd_grad_updates, self.args['grad_clip'])
			unfiltererd_grad_updates = copy.deepcopy(grad_updates)
			unfiltererd_grad_updates_pretrain = copy.deepcopy(grad_updates_pretrain)
			grad_updates = filter_grad_updates(grad_updates, worker_thetas)
			grad_updates_pretrain = filter_grad_updates(grad_updates_pretrain, worker_thetas)

			# changes the grad_updates altogether
			aggregated_gradient_updates = aggregate_gradient_updates(
				grad_updates, device=self.device, mode=aggregate_mode, credits=credits)

			aggregated_gradient_updates_pretrain = aggregate_gradient_updates(
				grad_updates_pretrain, device=self.device, mode=aggregate_mode, credits=credits_pretrain)


			# credit_weighted_grad_updates =  [credit * update for credit, grad_update in zip(credits, grad_updates) for update in grad_update]
			# credit_aggregated_gradient_updates = aggregate_gradient_updates(grad_updates, device=self.device, mode='credit-sum', credits=credits)

			# param_frequency = [freq + (update.abs()>0).float()  for freq, update in zip(param_frequency, aggregated_gradient_updates) ]

			dssgd_grad_updates = filter_grad_updates(
				dssgd_grad_updates, worker_thetas)
			# generate a roundrobin sequence
			rr_sequence = double_seqs[
				epoch % self.n_workers: epoch % self.n_workers + self.n_workers]
			self.update_dssgd_model(rr_sequence, dssgd_grad_updates)

			# 2. update the federated_model
			self.federated_model = add_update_to_model(
				self.federated_model, aggregated_gradient_updates, device=device)

			# _, federated_val_acc = evaluate(
			# 	self.federated_model, self.valid_loader, device, verbose=False)
			# print("CFFL server model validation accuracy : {:.4%}".format(federated_val_acc))

			self.federated_model_pretrain = add_update_to_model(self.federated_model_pretrain, aggregated_gradient_updates_pretrain, device=device)
			_, federated_val_acc_pretrain = evaluate(self.federated_model_pretrain, self.valid_loader, device, verbose=False)


			dssgd_val_accs = self.evaluate_workers_performance(
				self.valid_loader, mode='dssgd')
			self.performance_dict['dssgd_val_accs'].append(dssgd_val_accs)
			self.performance_dict_pretrain['dssgd_val_accs'].append(dssgd_val_accs)
			# print("DSSGD models validation accuracies: ", ["{:.4%}".format(dssgd_val_acc) for dssgd_val_acc in dssgd_val_accs ])

			# 3.1 carry out evaluations
			'''
			loo_val_accs = leave_one_out_evaluate(self.federated_model, grad_updates, self.valid_loader, device)
			print("Leave-one-out validation accuracies : ", ["{:.4%}".format(loo_val_acc) for loo_val_acc in loo_val_accs]   )
			'''
			worker_val_accs = one_on_one_evaluate(
				self.workers, self.federated_model, grad_updates, unfiltererd_grad_updates, self.valid_loader, device)
			# print("One-on-one validation accuracies : ", ["{:.4%}".format(val_acc) for val_acc in worker_val_accs])

			# 3.2 compute credits
			# credits = compute_credits(credits, federated_val_acc, loo_val_accs, credit_threshold=credit_threshold)
			credits = compute_credits_sinh(
				credits, worker_val_accs, credit_threshold=credit_threshold, alpha=self.args['alpha'],)
			# print("Computed and normalized credits: ", credits.tolist())
			credit_threshold = torch.div( 1.0, (torch.sum(credits > 0) - 1)) * (2. / 3.)
			# to avoid NaN when only one player in C
			credit_threshold = min(torch.tensor(2./3), credit_threshold)
			# print("New credit threshold is : ", credit_threshold.item())


			worker_val_accs_pretrain = one_on_one_evaluate(self.workers, self.federated_model_pretrain, grad_updates_pretrain, unfiltererd_grad_updates_pretrain, self.valid_loader, device)
			credits_pretrain = compute_credits_sinh(credits_pretrain, worker_val_accs_pretrain, credit_threshold=credit_threshold_pretrain, alpha=self.args['alpha'],)
			credit_threshold_pretrain = torch.div( 1.0, (torch.sum(credits_pretrain > 0) - 1)) * (2. / 3.)
			credit_threshold_pretrain = min(torch.tensor(1), credit_threshold_pretrain)


			# 4. gradient downloads and uploads according to credits and thetas
			self.assign_updates_with_filter(credits, aggregated_gradient_updates, grad_updates, unfiltererd_grad_updates)
			
			self.assign_updates_with_filter(credits_pretrain, aggregated_gradient_updates_pretrain, grad_updates_pretrain, unfiltererd_grad_updates_pretrain, pretrain=True)


			# 5. evaluate the federated_model at the end of each epoch
			if epoch % 20 == 0:
				print('Epoch {}:'.format(epoch + 1))
			self.performance_summary(to_print=(epoch%20==0))

			self.performance_dict['credits'].append(credits.tolist())
			self.performance_dict['federated_val_acc'].append(federated_val_acc.item())
			self.performance_dict['credit_threshold'].append(credit_threshold.item())

			self.performance_dict_pretrain['credits'].append(credits_pretrain.tolist())
			self.performance_dict_pretrain['federated_val_acc'].append(federated_val_acc_pretrain.item())
			self.performance_dict_pretrain['credit_threshold'].append(credit_threshold_pretrain.item())
			# print()

		self.cffl_test_accs = self.evaluate_workers_performance(
			self.test_loader)

		self.cffl_test_accs_w_pretrain = self.evaluate_workers_performance(
			self.test_loader, mode='pretrain')

		self.worker_standalone_test_accs = self.evaluate_workers_performance(
			self.test_loader, mode='standalone')
		return

	def update_dssgd_model(self, sequence, dssgd_grad_updates):
		for Id in sequence:
			add_update_to_model(self.dssgd_model, dssgd_grad_updates[Id])
			self.workers[Id].dssgd_model.load_state_dict(
				self.dssgd_model.state_dict())

	def assign_updates_with_filter(self, credits, aggregated_gradient_updates, grad_updates, unfiltererd_grad_updates, pretrain=False):
		"""
		download the largest magnitude updates <credits[i] * num_param> from the server
		and filter out its own updates in the local model
		and apply to its local model
		"""

		for i, (credit, worker, worker_update, worker_update_unfiltered) in enumerate(zip(credits, self.workers, grad_updates, unfiltererd_grad_updates)):
			if credit == 0 or torch.isnan(credit):
				credit = 0

			agg_grad_update = copy.deepcopy(aggregated_gradient_updates)
			num_param_downloads = int(credit * worker.param_count)

			allocated_grad = mask_grad_update_by_order(agg_grad_update, num_param_downloads)
			for res_param_update, worker_param_update, worker_param_update_unfiltered in zip(allocated_grad, worker_update, worker_update_unfiltered):

				# filter out and remove the updates from itself
				filter_indices = (res_param_update.abs() > 0) & (worker_param_update.abs() > 0)
				res_param_update.data[filter_indices] -= worker_param_update.data[filter_indices]

				# add in back the clipped but unfiltered own grad updates
				res_param_update.data += worker_param_update_unfiltered.data

			if pretrain:
				add_update_to_model(worker.model_pretrain, allocated_grad)
			else:
				add_update_to_model(worker.model, allocated_grad)
		return

	'''
	def assign_parameters(self, credits, param_frequency):

		"""
		download the most frequently updated <credits[i] * num_param> parameters from the server
		and replace the corresponding parameters in the local model
		server needs to keep track of a parameter update frequency mapping
		"""

		freqs = torch.cat( [freq.data.view(-1) for freq in param_frequency])
		for i, (credit, worker) in enumerate( zip(credits, self.workers)):

			num_param_downloads = int(credit * worker.param_count)
			topk, _ = torch.topk(freqs, num_param_downloads)
			target_freq = topk[-1]

			for worker_param, federated_param, param_freq in zip(worker.model.parameters(), self.federated_model.parameters(), param_frequency):
				downloading_indices = param_freq > target_freq
				worker_param.data[downloading_indices] = federated_param.data[downloading_indices]

		return

	def assign_updates(self, credits, param_frequency, aggregated_gradient_updates):
		
		# download the most frequently updated <credits[i] * num_param> parameters' updates from the aggregated update
		# server needs to keep track of a parameter update frequency mapping

		freqs = torch.cat( [freq.data.view(-1) for freq in param_frequency])
		for i, (credit, worker) in enumerate( zip(credits, self.workers)):
			grad_update = copy.deepcopy(aggregated_gradient_updates)
			num_param_downloads = int(credit * worker.param_count)
			topk, _ = torch.topk(freqs, num_param_downloads)
			target_freq = topk[-1]
			for freq, update in zip(param_frequency, grad_update):
				update.data[freq < target_freq] = 0
			add_update_to_model(worker.model, grad_update)
		return


	def trade_gradients(self, points, sorted_grad_updates):
		"""
		Follows the Point Update step in Algorithm 2 in TFDP

		"""
		for download_worker_id, worker in enumerate(self.workers):
			downloaded_updates = []
			for grad_update, upload_worker_id in sorted_grad_updates:
				# skip itself
				if upload_worker_id != download_worker_id:

					upload_worker = self.workers[upload_worker_id]
					upload_threshold = upload_worker.theta * upload_worker.param_count

					download_budget = points[download_worker_id]

					trade_count = int(min(upload_threshold, download_budget))

					points[download_worker_id] -= trade_count
					points[upload_worker_id] += trade_count

					downloaded_updates.append(mask_grad_update_by_order(grad_update, trade_count))
					self.sharing_ledger[upload_worker_id] += trade_count

			averaged_downloaded_update = aggreagate_gradient_updates(downloaded_updates, device=worker.device, mode='mean')
			# print(averaged_downloaded_update)
			backup_model = copy.deepcopy(worker.model)
			worker.model = add_update_to_model(worker.model, averaged_downloaded_update, device=worker.device)
		return
	'''

	def performance_summary(self, to_print=False):

		self.dssgd_models_test_accs = self.evaluate_workers_performance(self.test_loader, mode='dssgd')
		self.worker_standalone_test_accs = self.evaluate_workers_performance(self.test_loader, mode='standalone')
		self.cffl_test_accs = self.evaluate_workers_performance(self.test_loader)
		self.cffl_test_accs_w_pretrain = self.evaluate_workers_performance(self.test_loader, mode='pretrain')

		# no pretrain
		self.performance_dict['DSSGD_model_test_accs'].append(
			self.dssgd_models_test_accs)
		self.performance_dict['worker_standalone_test_accs'].append(
			self.worker_standalone_test_accs)
		self.performance_dict['cffl_test_accs'].append(
			self.cffl_test_accs)

		# with pretrain
		self.performance_dict_pretrain['DSSGD_model_test_accs'].append(
			self.dssgd_models_test_accs)
		self.performance_dict_pretrain['worker_standalone_test_accs'].append(
			self.worker_standalone_test_accs)
		self.performance_dict_pretrain['cffl_test_accs'].append(
			self.cffl_test_accs_w_pretrain)

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

	def get_fairness_analysis(self):
		print("Performance and Fairness analysis: ")
		worker_thetas = [worker.theta for worker in self.workers]
		sharing_contributions = (torch.tensor(
			self.shard_sizes) * torch.tensor(worker_thetas)).tolist()
		print('Workers sharing_contributions : ', sharing_contributions)

		self.performance_summary()

		# no pretrain
		credits = self.performance_dict['credits'][-1]
		remaining_workers_indices = [i for i, credit in enumerate(credits)]
		remaining_standalone_test_accs =  [ self.worker_standalone_test_accs[i] for i in remaining_workers_indices]
		remaining_dssgd_test_accs =  [ self.dssgd_models_test_accs[i] for i in remaining_workers_indices]
		remaining_cffl_test_accs =  [ self.cffl_test_accs[i] for i in remaining_workers_indices]


		from scipy.stats import pearsonr
		corrs = pearsonr(remaining_standalone_test_accs,
						 remaining_dssgd_test_accs)
		self.performance_dict['standlone_vs_rrdssgd'].append(corrs[0])

		corrs = pearsonr(remaining_standalone_test_accs,
						 remaining_cffl_test_accs)
		self.performance_dict['standalone_vs_final'].append(corrs[0])


		self.performance_dict['CFFL_best_worker'] = max(self.cffl_test_accs)
		best_worker_id = np.argmax(self.cffl_test_accs)
		self.performance_dict['standalone_best_worker'] = self.worker_standalone_test_accs[best_worker_id]

		self.performance_dict['rr_dssgd_best'] = self.dssgd_models_test_accs[best_worker_id]

		# with pretrain
		credits_pretrain = self.performance_dict_pretrain['credits'][-1]
		remaining_workers_indices = [i for i, credit in enumerate(credits_pretrain)]
		remaining_standalone_test_accs =  [ self.worker_standalone_test_accs[i] for i in remaining_workers_indices]
		remaining_dssgd_test_accs =  [ self.dssgd_models_test_accs[i] for i in remaining_workers_indices]
		remaining_cffl_test_accs_w_pretrain =  [ self.cffl_test_accs_w_pretrain[i] for i in remaining_workers_indices]

		corrs = pearsonr(remaining_standalone_test_accs,
						 remaining_dssgd_test_accs)
		self.performance_dict_pretrain['standlone_vs_rrdssgd'].append(corrs[0])

		corrs = pearsonr(remaining_standalone_test_accs,
						 remaining_cffl_test_accs_w_pretrain)
		self.performance_dict_pretrain['standalone_vs_final'].append(corrs[0])


		self.performance_dict_pretrain['CFFL_best_worker'] = max(self.cffl_test_accs_w_pretrain)
		best_worker_id = np.argmax(self.cffl_test_accs_w_pretrain)

		self.performance_dict_pretrain['standalone_best_worker'] = self.worker_standalone_test_accs[best_worker_id]
		self.performance_dict_pretrain['rr_dssgd_best'] = self.dssgd_models_test_accs[best_worker_id]


		keys = ['standalone_best_worker', 'CFFL_best_worker', 'rr_dssgd_best', 'standlone_vs_rrdssgd', 'standalone_vs_final']
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
			return [evaluate(worker.standalone_model, eval_loader, device, verbose=False)[1].tolist() for worker in self.workers]
		elif mode == 'dssgd':
			return [evaluate(worker.dssgd_model, eval_loader, device, verbose=False)[1].tolist() for worker in self.workers]
		elif mode == 'pretrain':
			return [evaluate(worker.model_pretrain, eval_loader, device, verbose=False)[1].tolist() for worker in self.workers]
		else:
			return [evaluate(worker.model, eval_loader, device, verbose=False)[1].tolist() for worker in self.workers]


def compute_credits_sinh(credits, val_accs, credit_threshold, alpha=5, credit_fade=1):
	# print('alpha used is :', alpha, ' current credits are : ', credits, ' current threshold: ', credit_threshold)
	total = sum(val_accs)
	for i, (credit, val_acc) in enumerate(zip(credits, val_accs)):
		credit_epoch = val_acc / total
		if credit_fade == 1:
			credits[i] = credit * 0.2 + credit_epoch * 0.8
		else:
			credits[i] = (credit + credit_epoch) * 0.5

		if credits[i] < credit_threshold:
			credits[i] = 0

		credits[i] = math.sinh(alpha * credits[i])


	if torch.sum(credits) == 0:
		print('credits all zero')
	credits = credits / torch.sum(credits)
	credits[credits < credit_threshold] = 0

	if torch.sum(credits) == 0:
		print('credits all zero')
	# print("computed normalized credits are: ", credits/ torch.sum(credits))
	return credits / torch.sum(credits)


def clip_gradients_(grad_updates, grad_clip):
	grad_clip = abs(grad_clip)
	for grad_update in grad_updates:
		for param in grad_update:
			param.data = torch.clamp(param.data, min=-grad_clip, max=grad_clip)

def filter_grad_updates(grad_updates, thetas):
	"""
	Filter the grad_updates by the largest magnitude criterion top m%

	"""
	return [mask_grad_update_by_order(grad_update, mask_order=None, mask_percentile=theta) for grad_update, theta in zip(grad_updates, thetas)]


def mask_grad_update_by_order(grad_update, mask_order, mask_percentile=None):
	# mask all but the largest <mask_order> updates (by magnitude) to zero
	all_update_mod = torch.cat([update.data.view(-1).abs()
								for update in grad_update])
	if not mask_order and mask_percentile:
		mask_order = int(len(all_update_mod) * mask_percentile)
	
	if mask_order == 0:
		return mask_grad_update_by_magnitude(grad_update, 0)
	else:
		topk, indices = torch.topk(all_update_mod, mask_order)
		return mask_grad_update_by_magnitude(grad_update, topk[-1])


def mask_grad_update_by_magnitude(grad_update, mask_constant):
	# mask all but the updates with larger magnitude than <mask_constant> to zero
	# print('Masking all gradient updates with magnitude smaller than ', mask_constant)
	for i, update in enumerate(grad_update):
		grad_update[i].data[update.data.abs() < mask_constant] = 0
	return grad_update


'''
def credit_curve(x):
	from math import exp
	return 1. / (1 + exp(-15 * (x - 0.5)))

def compute_credits(credits, federated_val_acc, leave_one_out_val_accs, credit_threshold, credit_fade=0):
	for i, credit in enumerate(credits):
		gain = federated_val_acc / (federated_val_acc + leave_one_out_val_accs[i])
		if credit >= credit_threshold:
			if credit_fade == 1:
				credits[i] = 0.2 * credits[i] + 0.8 * credit_curve(gain)
			else:
				credits[i] = 0.5 * (credits[i] + credit_curve(gain) )
		else:
			credits[i] = 0
	return credits / torch.sum(credits)


def sort_grad_updates(grad_updates, marginal_contributions):
	# sort the grad_updates by marginal_contributions (descending order)
	return [(grad_update, worker_id) for grad_update, marg_contr, worker_id in sorted(zip(grad_updates, marginal_contributions, range(len(grad_updates))), key=lambda x:x[1], reverse=True) ]

'''
