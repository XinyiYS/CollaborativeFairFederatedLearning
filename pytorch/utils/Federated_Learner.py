import copy
import torch
from torch import nn, optim

from utils.Data_Prepper import Data_Prepper
from utils.Worker import Worker

from utils.utils import evaluate, averge_models, average_gradient_updates, \
	add_update_to_model, compute_grad_update, compare_models,  \
	leave_one_out_evaluate, compute_shapley


class Federated_Learner:

	def __init__(self, args, data_prepper):
		self.args = args
		self.data_prepper = data_prepper
		self.n_workers = self.args['n_workers']

		self.valid_loader = data_prepper.get_valid_loader()
		self.test_loader = data_prepper.get_test_loader()

		self.worker_train_loaders = self.data_prepper.get_train_loaders(
			self.args['n_workers'], self.args['balanced'])
		self.init_workers()

	def init_workers(self):
		assert self.n_workers == len(
			self.worker_train_loaders), "Num of workers is not equal to num of loaders"
		model_fn = self.args['model_fn']
		optimizer_fn = self.args['optimizer_fn']
		lr = self.args['lr']
		device = self.args['device']
		loss_fn = self.args['loss_fn']
		sharing_lambda = self.args['sharing_lambda']

		self.workers = []
		# possible to enumerate through various model_fns, optimizer_fns, lrs,
		# sharing_lambdas, or even devices
		for i, worker_train_loader in enumerate(self.worker_train_loaders):
			model = model_fn()
			optimizer = optimizer_fn(model.parameters(), lr=lr)

			worker = Worker(train_loader=worker_train_loader,
							model=model, optimizer=optimizer, loss_fn=loss_fn,
							id=str(i), sharing_lambda=sharing_lambda,
							device=device)
			self.workers.append(worker)
		return

	def pretrain_locally(self, epochs, test=False, parallelize=True):        
		for worker in self.workers:
			worker.train_locally(epochs=epochs)
			if test:
				evaluate(worker.model, self.test_loader, worker.device)
		return


	def train(self):
		print("Start local pretraining ")
		self.pretrain_locally(self.args['pretrain_epochs'])
		self.worker_model_test_accs_before = self.evaluate_workers_performance(self.test_loader)
		self.sharing_ledger = torch.zeros((self.n_workers))
		self.shapley_values = torch.zeros((self.n_workers))

		federated_model = averge_models(
			[worker.model for worker in self.workers])
		print("Performance of an average model of the pretrained local models")
		evaluate(federated_model, self.test_loader, self.args['device'], loss_fn=self.args['loss_fn'], verbose=True)

		points = torch.zeros((self.n_workers))

		fl_epochs = self.args['fl_epochs']
		device = self.args['device']
		fl_individual_epochs = self.args['fl_individual_epochs']

		print("Start federated learning ")
		for epoch in range(fl_epochs):
			grad_updates = []
			for worker in self.workers:
				model_before = copy.deepcopy(worker.model)
				worker.train_locally(fl_individual_epochs)
				model_after = copy.deepcopy(worker.model)
				grad_updates.append(compute_grad_update(
					model_before, model_after, device=device))

			# updates the federated model in function for efficiency
			marginal_contributions = leave_one_out_evaluate(
				federated_model, grad_updates, self.valid_loader, device)
			print("Marginal contributions are: ", marginal_contributions)

			# self.shapley_values += compute_shapley(grad_updates, federated_model, test_loader, device)

			points = distribute_points(points, marginal_contributions)
			sorted_grad_updates = sort_grad_updates(
				grad_updates, marginal_contributions)

			for download_worker_id, worker in enumerate(self.workers):
				acquired_updates = []
				for grad_update, upload_worker_id in sorted_grad_updates:
					# not self and sufficient budget
					if upload_worker_id != download_worker_id and points[download_worker_id] > 1:
						points[download_worker_id] -= 1
						points[upload_worker_id] += 1  # paying to this worker
						acquired_updates.append(grad_update)
						self.sharing_ledger[upload_worker_id] += 1

				averaged_acquired_update = average_gradient_updates(
					acquired_updates)
				worker.model = add_update_to_model(
					worker.model, averaged_acquired_update, device=device)
			evaluate(federated_model, self.test_loader, self.args['device'], loss_fn=self.args['loss_fn'])

		self.worker_model_test_accs_after = self.evaluate_workers_performance(self.test_loader)
		self.federated_model = federated_model
		return

	def get_fairness_analysis(self):

		sharing_ledger = self.sharing_ledger
		worker_model_test_accs_after = self.worker_model_test_accs_after
		worker_model_test_accs_before = self.worker_model_test_accs_before
		shapley_values = self.shapley_values

		import scipy.stats
		corrs = scipy.stats.pearsonr(
			sharing_ledger, worker_model_test_accs_after)
		print("test_acc vs sharing ledger: ", corrs)

		worker_model_improvements = [now - before for now, before in zip(
			worker_model_test_accs_after, worker_model_test_accs_before)]
		corrs = scipy.stats.pearsonr(sharing_ledger, worker_model_improvements)
		print("test_acc improvements vs sharing ledger: ", corrs)

		corrs = scipy.stats.pearsonr(sharing_ledger, shapley_values)
		print('sharing ledge vs shapley values: ', corrs)

		corrs = scipy.stats.pearsonr(shapley_values, worker_model_improvements)
		print('shapley values vs model improvements: ', corrs)

		print('worker_model_test_accs_after: ', worker_model_test_accs_after)
		print('worker_model_improvements: ', worker_model_improvements)
		print('sharing ledger: ', sharing_ledger)
		print('shapley values: ', shapley_values)
		return

	def evaluate_workers_performance(self, eval_loader):
		device = self.args['device']
		return [evaluate(worker.model, eval_loader, device, verbose=False)[1].tolist() for worker in self.workers]


def distribute_points(points, marginal_contributions, epsilon=1e-4):
	# normalize so that the max is equal to n_workers - 1

	# set small values to 0
	marginal_contributions[marginal_contributions.abs() < epsilon] = 0
	if not (marginal_contributions == 0).all():

		# if all negative
		if (marginal_contributions < 0).all():
			ratio = (len(points) - 1) / torch.max(marginal_contributions.abs())
		else:
			ratio = (len(points) - 1) / torch.max(marginal_contributions)
		marginal_contributions *= ratio
	print('resized contributions:', marginal_contributions)
	return points + marginal_contributions


def sort_grad_updates(grad_updates, marginal_contributions):
	# sort the grad_updates according to the marginal_contributions in a
	# descending order
	return [(grad_update, worker_id) for grad_update, marg_contr, worker_id in sorted(zip(grad_updates, marginal_contributions, range(len(grad_updates))), key=lambda x:x[1], reverse=True)]
