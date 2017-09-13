import math
import numpy as np
import sampyl as smp

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from HyperSphere.GP.inference.inverse_bilinear_form import InverseBilinearForm
from HyperSphere.GP.inference.log_determinant import LogDeterminant


class Inference(nn.Module):

	def __init__(self, train_data, model):
		super(Inference, self).__init__()
		self.model = model
		self.train_x = train_data[0]
		self.train_y = train_data[1]

	def reset_parameters(self):
		for m in self.model.children():
			m.reset_parameters()

	def predict(self, pred_x, hyper=None):
		if hyper is not None:
			self.model.vec_to_param(hyper)
		k_pred_train = self.model.kernel(pred_x, self.train_x)
		K_noise = self.model.kernel(self.train_x) + torch.diag(self.model.likelihood(self.train_x))

		shared_part, _ = torch.gesv(k_pred_train.t(), K_noise)
		kernel_on_identical = torch.cat([self.model.kernel(pred_x[[i], :]) for i in range(pred_x.size(0))])

		pred_mean = torch.mm(shared_part.t(), self.train_y - self.model.mean(self.train_x)) + self.model.mean(pred_x)
		pred_var = kernel_on_identical - (shared_part.t() * k_pred_train).sum(1, keepdim=True)
		return pred_mean, pred_var

	def negative_log_likelihood(self):
		K_noise = self.model.kernel(self.train_x) + torch.diag(self.model.likelihood(self.train_x))
		return 0.5 * InverseBilinearForm.apply(self.train_y, K_noise, self.train_y) + 0.5 * LogDeterminant.apply(K_noise) + 0.5 * self.train_y.size(0) * math.log(2 * math.pi)

	def learning(self, n_restarts=10):
		vec_list = []
		nll_list = []
		for r in range(n_restarts):
			if r != 0:
				for m in self.model.children():
					m.reset_parameters()

			###--------------------------------------------------###
			# This block can be modified to use other optimization method
			n_step = 100
			optimizer = optim.Adam(self.model.parameters(), lr=0.01)
			for _ in range(n_step):
				optimizer.zero_grad()
				loss = self.negative_log_likelihood()
				loss.backward(retain_graph=True)
				optimizer.step()
			###--------------------------------------------------###

			vec_list.append(self.model.param_to_vec())
			nll_list.append(self.negative_log_likelihood().data.squeeze()[0])
		_, best_ind = torch.min(torch.FloatTensor(nll_list), dim=0)
		return vec_list[best_ind[0]].unsqueeze(0)
		# self.model.vec_to_param(vec_list[best_ind[0]])

	def sampling(self, n_sample=5, n_burnin=100, n_thin=10):
		likelihood_param_data = list(self.model.likelihood.parameters())[0].data
		def logp(hyper):
			if self.model.out_of_bounds(hyper):
				return -np.inf
			prior = self.model.prior(hyper)
			self.model.vec_to_param(torch.from_numpy(hyper).type_as(likelihood_param_data))
			likelihood = -self.negative_log_likelihood().data.squeeze()[0]
			return prior + likelihood
		start = self.model.param_to_vec()
		print(list(self.model.named_parameters()))
		###--------------------------------------------------###
		# This block can be modified to use other sampling method
		sampler = smp.Slice(logp=logp, start={'hyper': (start.cpu() if start.is_cuda else start).numpy()})
		samples = sampler.sample(n_burnin + n_thin * n_sample, burn=n_burnin, thin=n_thin)
		###--------------------------------------------------###

		return torch.stack([torch.from_numpy(elm[0]) for elm in samples], 0).type_as(likelihood_param_data)


if __name__ == '__main__':
	from HyperSphere.GP.kernels.modules.squared_exponential import SquaredExponentialKernel
	from HyperSphere.GP.models.gp_regression import GPRegression
	import matplotlib.pyplot as plt
	ndata = 10
	ndim = 1
	model_for_generating = GPRegression(kernel=SquaredExponentialKernel(ndim))
	train_x = Variable(torch.FloatTensor(ndata, ndim).uniform_(-2, 2))
	chol_L = torch.potrf((model_for_generating.kernel(train_x) + torch.diag(model_for_generating.likelihood(train_x))).data, upper=False)
	train_y = torch.sin(2 * math.pi * torch.sum(train_x**2, 1, keepdim=True)**0.5) + model_for_generating.mean(train_x) + Variable(torch.mm(chol_L, torch.randn(ndata, 1)))
	train_data = (train_x, train_y)
	generated_nll = Inference(train_data, model_for_generating).negative_log_likelihood().data[0, 0]

	model_for_learning = GPRegression(kernel=SquaredExponentialKernel(ndim))
	inference = Inference(train_data, model_for_learning)
	param_original = model_for_generating.param_to_vec()
	inference.reset_parameters()
	model_for_learning.kernel.log_amp.data[:] = np.log(torch.std(train_y.data))
	model_for_learning.kernel.log_ls.data[:] = 0.0
	model_for_learning.mean.const_mean.data[:] = torch.mean(train_y.data)
	model_for_learning.likelihood.log_noise_var.data[:] = np.log(0.001)
	param_samples_learning = inference.learning(n_restarts=10)
	inference.reset_parameters()
	model_for_learning.kernel.log_amp.data[:] = np.log(torch.std(train_y.data))
	model_for_learning.kernel.log_ls.data[:] = 0.0
	model_for_learning.mean.const_mean.data[:] = torch.mean(train_y.data)
	model_for_learning.likelihood.log_noise_var.data[:] = np.log(0.001)
	param_samples_sampling = inference.sampling()

	if ndim == 1:
		pred_x = torch.linspace(-2.5, 2.5, 100).view(-1, 1)
		fig, axes = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True)

		pred_mean, pred_var = inference.predict(Variable(pred_x), param_original)
		pred_std = torch.sqrt(pred_var)
		pred_mean = pred_mean.data
		pred_var = pred_var.data
		pred_std = pred_std.data
		axes[0].plot(train_x.data.numpy().flatten(), train_y.data.numpy().flatten(), '+')
		axes[0].plot(pred_x.numpy().flatten(), pred_mean.numpy().flatten(), 'b')
		axes[0].fill_between(pred_x.numpy().flatten(), (pred_mean - pred_std).numpy().flatten(),
		                 (pred_mean + pred_std).numpy().flatten(), facecolor='green', alpha=0.2)
		axes[0].fill_between(pred_x.numpy().flatten(), (pred_mean - 1.96 * pred_std).numpy().flatten(),
		                 (pred_mean + 1.96 * pred_std).numpy().flatten(), facecolor='green', alpha=0.2)
		axes[0].set_title('Original')

		pred_mean, pred_var = inference.predict(Variable(pred_x), param_samples_learning[0])
		pred_std = torch.sqrt(pred_var)
		pred_mean = pred_mean.data
		pred_var = pred_var.data
		pred_std = pred_std.data
		axes[1].plot(train_x.data.numpy().flatten(), train_y.data.numpy().flatten(), '+')
		axes[1].plot(pred_x.numpy().flatten(), pred_mean.numpy().flatten(), 'b')
		axes[1].fill_between(pred_x.numpy().flatten(), (pred_mean - pred_std).numpy().flatten(),
		                     (pred_mean + pred_std).numpy().flatten(), facecolor='green', alpha=0.2)
		axes[1].fill_between(pred_x.numpy().flatten(), (pred_mean - 1.96 * pred_std).numpy().flatten(),
		                     (pred_mean + 1.96 * pred_std).numpy().flatten(), facecolor='green', alpha=0.2)
		axes[1].set_title('Optimized')

		pred_mean = 0
		pred_var = 0
		pred_std = 0
		for s in range(param_samples_sampling.size()[0]):
			pred_mean_sample, pred_var_sample = inference.predict(Variable(pred_x), param_samples_sampling[s])
			pred_std_sample = torch.sqrt(pred_var_sample)
			pred_mean += pred_mean_sample.data
			pred_var += pred_var_sample.data
			pred_std += pred_std_sample.data
		pred_mean /= param_samples_sampling.size()[0]
		pred_var /= param_samples_sampling.size()[0]
		pred_std /= param_samples_sampling.size()[0]
		axes[2].plot(train_x.data.numpy().flatten(), train_y.data.numpy().flatten(), '+')
		axes[2].plot(pred_x.numpy().flatten(), pred_mean.numpy().flatten(), 'b')
		axes[2].fill_between(pred_x.numpy().flatten(), (pred_mean - pred_std).numpy().flatten(),
		                     (pred_mean + pred_std).numpy().flatten(), facecolor='green', alpha=0.2)
		axes[2].fill_between(pred_x.numpy().flatten(), (pred_mean - 1.96 * pred_std).numpy().flatten(),
		                     (pred_mean + 1.96 * pred_std).numpy().flatten(), facecolor='green', alpha=0.2)
		axes[2].set_title('Sampled')

		plt.show()