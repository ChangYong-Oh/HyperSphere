import math
import numpy as np
import sampyl as smp

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.nn.modules.module import Module
from HyperSphere.GP.inference.inverse_bilinear_form import InverseBilinearForm
from HyperSphere.GP.inference.log_determinant import LogDeterminant


class Inference(nn.Module):

	def __init__(self, train_data, model, hyper=None):
		super(Inference, self).__init__()
		self.model = model
		self.train_x = train_data[0]
		self.train_y = train_data[1]
		assert self.model.kernel.input_map(Variable(torch.ones(1, self.train_x.size(1)))).numel() == model.kernel.ndim
		self.mean_vec = None
		self.K_noise = None
		self.K_noise_inv = None
		self.matrix_update(hyper)

	def reset_parameters(self):
		self.model.reset_parameters()

	def model_param_init(self):
		if self.train_x.size(0) < 5:
			log_std = torch.std(self.train_y).log().data + 1e-4
			self.model.kernel.log_amp.data = log_std
			self.model.kernel.log_ls.data.fill_(0)
			if isinstance(self.model.kernel.input_map, Module):
				self.model.kernel.input_map.model_param_init()
			self.model.mean.const_mean.data.fill_(torch.mean(self.train_y.data))
			self.model.likelihood.log_noise_var.data.fill_(-3)
		else:
			raise RuntimeError('Initialization is called after long training')

	def matrix_update(self, hyper=None):
		if hyper is not None:
			self.model.vec_to_param(hyper)
		self.mean_vec = self.train_y - self.model.mean(self.train_x)
		self.K_noise = self.model.kernel(self.train_x) + torch.diag(self.model.likelihood(self.train_x))
		if hasattr(self.K_noise, 'data'):
			eye_mat = Variable(torch.eye(self.K_noise.size(0)).type_as(self.K_noise.data))
		else:
			eye_mat = torch.eye(self.K_noise.size(0)).type_as(self.K_noise)
		self.K_noise_inv, _ = torch.gesv(eye_mat, self.K_noise)

	def predict(self, pred_x, hyper=None):
		if hyper is not None:
			param_original = self.model.param_to_vec()
			self.matrix_update(hyper)
		k_pred_train = self.model.kernel(pred_x, self.train_x)

		shared_part = k_pred_train.mm(self.K_noise_inv)
		kernel_on_identical = torch.cat([self.model.kernel(pred_x[[i], :]) for i in range(pred_x.size(0))])

		pred_mean = torch.mm(shared_part, self.mean_vec) + self.model.mean(pred_x)
		pred_var = kernel_on_identical - (shared_part * k_pred_train).sum(1, keepdim=True)
		if hyper is not None:
			self.model.vec_to_param(param_original)
		return pred_mean, pred_var

	def negative_log_likelihood(self, hyper=None):
		if hyper is not None:
			param_original = self.model.param_to_vec()
			self.matrix_update(hyper)
		nll = 0.5 * InverseBilinearForm.apply(self.mean_vec, self.K_noise, self.mean_vec) + 0.5 * LogDeterminant.apply(self.K_noise) + 0.5 * self.train_y.size(0) * math.log(2 * math.pi)
		if hyper is not None:
			self.model.vec_to_param(param_original)
		return nll

	def learning(self, n_restarts=10):
		vec_list = []
		nll_list = []
		for r in range(n_restarts):
			if r != 0:
				for m in self.model.children():
					m.reset_parameters()

			###--------------------------------------------------###
			# This block can be modified to use other optimization method
			n_step = 50
			optimizer = optim.Adam(self.model.parameters(), lr=0.01)
			for _ in range(n_step):
				optimizer.zero_grad()
				loss = self.negative_log_likelihood(self.model.param_to_vec())
				loss.backward(retain_graph=True)
				optimizer.step()
			###--------------------------------------------------###

			vec_list.append(self.model.param_to_vec())
			nll_list.append(self.negative_log_likelihood().data.squeeze()[0])
		_, best_ind = torch.min(torch.FloatTensor(nll_list), dim=0)
		self.model.vec_to_param(vec_list[best_ind[0]])
		return vec_list[best_ind[0]].unsqueeze(0)

	def sampling(self, n_sample=10, n_burnin=100, n_thin=10):
		type_as_arg = list(self.model.likelihood.parameters())[0].data
		def logp(hyper):
			if self.model.out_of_bounds(hyper):
				return -np.inf
			prior = self.model.prior(hyper)
			likelihood = -self.negative_log_likelihood(torch.from_numpy(hyper).type_as(type_as_arg)).data.squeeze()[0]
			return prior + likelihood
		hyper_torch = self.model.param_to_vec()
		hyper_numpy = (hyper_torch.cpu() if hyper_torch.is_cuda else hyper_torch).numpy()
		###--------------------------------------------------###
		# This block can be modified to use other sampling method
		sampler = smp.Slice(logp=logp, start={'hyper': hyper_numpy}, compwise=True)
		samples = sampler.sample(n_burnin + n_thin * n_sample, burn=n_burnin + n_thin - 1, thin=n_thin)
		###--------------------------------------------------###
		self.model.vec_to_param(torch.from_numpy(samples[-1][0]).type_as(type_as_arg))
		return torch.stack([torch.from_numpy(elm[0]) for elm in samples], 0).type_as(type_as_arg)


def one_dim_plotting(ax, inference, param_samples, title_str=''):
	pred_x = torch.linspace(-2, 2, 100).view(-1, 1)
	if param_samples.dim() == 1:
		param_samples = param_samples.unsqueeze(0).clone()
	n_samples = param_samples.size()[0]
	pred_mean = 0
	pred_var = 0
	nll = 0
	pred_std = 0
	for s in range(n_samples):
		pred_mean_sample, pred_var_sample = inference.predict(Variable(pred_x), param_samples[s])
		pred_std_sample = torch.sqrt(pred_var_sample)
		pred_mean += pred_mean_sample.data
		pred_var += pred_var_sample.data
		nll += inference.negative_log_likelihood(param_samples[s]).data.squeeze()[0]
		pred_std += pred_std_sample.data
	pred_mean /= n_samples
	pred_var /= n_samples
	nll /= n_samples
	pred_std /= n_samples
	ax.plot(inference.train_x.data.numpy(), inference.train_y.data.numpy(), 'k*')
	ax.plot(pred_x.numpy().flatten(), pred_mean.numpy().flatten())
	ax.fill_between(pred_x.numpy().flatten(), (pred_mean - pred_std).numpy().flatten(),
	                (pred_mean + pred_std).numpy().flatten(), alpha=0.2)
	ax.fill_between(pred_x.numpy().flatten(), (pred_mean - 1.96 * pred_std).numpy().flatten(),
	                 (pred_mean + 1.96 * pred_std).numpy().flatten(), alpha=0.2)
	ax.set_title(title_str + '\n%.4E' % nll)


if __name__ == '__main__':
	from HyperSphere.GP.kernels.modules.squared_exponential import SquaredExponentialKernel
	from HyperSphere.GP.models.gp_regression import GPRegression
	import matplotlib.pyplot as plt
	ndata = 20
	ndim = 1
	model_for_generating = GPRegression(kernel=SquaredExponentialKernel(ndim))
	train_x = Variable(torch.FloatTensor(ndata, ndim).uniform_(-2, 2))
	chol_L = torch.potrf((model_for_generating.kernel(train_x) + torch.diag(model_for_generating.likelihood(train_x))).data, upper=False)
	train_y = model_for_generating.mean(train_x) + Variable(torch.mm(chol_L, torch.randn(ndata, 1)))
	train_data = (train_x, train_y)
	param_original = model_for_generating.param_to_vec()
	generated_nll = Inference(train_data, model_for_generating).negative_log_likelihood().data[0, 0]

	model_for_learning = GPRegression(kernel=SquaredExponentialKernel(ndim))
	inference = Inference(train_data, model_for_learning)
	model_for_learning.vec_to_param(param_original)
	param_samples_learning = inference.learning(n_restarts=10)
	model_for_learning.vec_to_param(param_original)
	param_samples_sampling = inference.sampling()

	if ndim == 1:
		pred_x = torch.linspace(-2.5, 2.5, 100).view(-1, 1)
		fig, axes = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True)

		one_dim_plotting(axes[0], inference, param_original, title_str='original')
		one_dim_plotting(axes[1], inference, param_samples_learning, title_str='optimized')
		one_dim_plotting(axes[2], inference, param_samples_sampling, title_str='sampled')

		plt.show()