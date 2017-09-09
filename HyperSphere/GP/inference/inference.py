import math

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from HyperSphere.GP.inference.inverse_bilinear_form import InverseBilinearForm
from HyperSphere.GP.inference.log_determinant import LogDeterminant


class Modelling(nn.Module):

	def __init__(self, train_data, model):
		super(Modelling, self).__init__()
		self.model = model
		self.train_x = train_data[0]
		self.train_y = train_data[1]

	def predict(self, pred_x):
		k_pred_train = self.model.kernel(pred_x, self.train_x)
		K_noise_inv = (self.model.kernel(self.train_x) + torch.diag(self.model.likelihood(self.train_x))).inverse()

		shared_part = torch.mm(k_pred_train, K_noise_inv)
		kernel_on_identical = torch.cat([self.model.kernel(pred_x[[i], :]) for i in range(pred_x.size(0))])

		pred_mean = torch.mm(shared_part, self.train_y - self.model.mean(self.train_x)) + self.model.mean(pred_x)
		pred_var = kernel_on_identical - (shared_part * k_pred_train).sum(1, keepdim=True)
		return pred_mean, pred_var

	def negative_log_likelihood(self):
		K_noise = self.model.kernel(self.train_x) + torch.diag(self.model.likelihood(self.train_x))
		return 0.5 * InverseBilinearForm.apply(self.train_y, K_noise, self.train_y) + 0.5 * LogDeterminant.apply(K_noise) + 0.5 * self.train_y.size(0) * math.log(2 * math.pi)

	def learning(self, n_restarts=5):
		vec_list = []
		nll_list = []
		for r in range(n_restarts):
			if r != 0:
				for module in self.model.modules():
					if hasattr(module, 'reset_parameters'):
						module.reset_parameters()

			###--------------------------------------------------###
			# This block can be modified to use other optimization method
			n_step = 100
			optimizer = optim.Adam(self.model.parameters(), lr=0.1)
			for s in range(n_step):
				optimizer.zero_grad()
				loss = self.negative_log_likelihood()
				loss.backward(retain_graph=True)
				optimizer.step()
			###--------------------------------------------------###
			vec_list.append(self.model.param_to_vec())
			nll_list.append(self.negative_log_likelihood().data.squeeze()[0])
		_, best_ind = torch.min(torch.FloatTensor(nll_list), dim=0)
		self.model.vec_to_param(vec_list[best_ind[0]])

	def sampling(self, n_sample=10, n_burnin=100, n_thin=10):
		pass

if __name__ == '__main__':
	from HyperSphere.GP.kernels.modules.squared_exponential import SquaredExponentialKernel
	from HyperSphere.GP.models.gp_regression import GPRegression
	import matplotlib.pyplot as plt
	ndata = 20
	ndim = 1
	model_for_generating = GPRegression(kernel=SquaredExponentialKernel(ndim))
	train_x = Variable(torch.FloatTensor(ndata, ndim).uniform_(-2, 2))
	chol_L = torch.potrf((model_for_generating.kernel(train_x) + torch.diag(model_for_generating.likelihood(train_x))).data, upper=False)
	train_y = torch.sin(2 * math.pi * torch.sum(train_x**2, 1, keepdim=True)**0.5) + model_for_generating.mean(train_x) + Variable(torch.mm(chol_L, torch.randn(ndata, 1)))
	train_data = (train_x, train_y)
	print('generating hyperparameters nll : %.4E' % Modelling(train_data, model_for_generating).negative_log_likelihood().data[0, 0])
	print(model_for_generating.param_to_vec())

	model_for_learning = GPRegression(kernel=SquaredExponentialKernel(ndim))
	modelling = Modelling(train_data, model_for_learning)
	modelling.learning()
	print('learned    hyperparameters nll : %.4E' % modelling.negative_log_likelihood().data[0, 0])
	print(model_for_learning.param_to_vec())

	if ndim == 1:
		pred_x = torch.linspace(-2.5, 2.5, 100).view(-1, 1)
		pred_mean, pred_var = modelling.predict(Variable(pred_x))
		print('Var : %.4E ~ %.4E' % (torch.min(pred_var.data), torch.max(pred_var.data)))
		pred_std = torch.sqrt(pred_var)
		plt.plot(train_x.data.numpy().flatten(), train_y.data.numpy().flatten(), '+')
		plt.plot(pred_x.numpy().flatten(), pred_mean.data.numpy().flatten(), 'b')
		plt.fill_between(pred_x.numpy().flatten(), (pred_mean.data - 1.96 * pred_std.data).numpy().flatten(), (pred_mean.data + 1.96 * pred_std.data).numpy().flatten(), facecolor='green', alpha=0.25)
		plt.show()
