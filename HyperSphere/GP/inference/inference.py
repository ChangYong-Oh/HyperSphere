import progressbar
import sys
import math
import time

import numpy as np
import scipy as sp
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
		self.output_min = torch.min(self.train_y.data)
		self.output_max = torch.max(self.train_y.data)
		self.mean_vec = None
		self.K_noise = None
		self.K_noise_inv = None
		self.K_noise_chol_U = None

	def reset_parameters(self):
		self.model.reset_parameters()

	def init_parameters(self):
		amp = torch.std(self.train_y).data[0]
		self.model.kernel.init_parameters(amp)
		self.model.mean.const_mean.data.fill_(torch.mean(self.train_y.data))
		self.model.likelihood.log_noise_var.data.fill_(np.log(amp / 1000))

	def stable_parameters(self):
		const_mean = self.model.mean.const_mean.data[0]
		kernel_log_amp = self.model.kernel.log_kernel_amp().data[0]
		log_noise_var = self.model.likelihood.log_noise_var.data[0]
		return self.output_min <= const_mean <= self.output_max

	def log_kernel_amp(self):
		return self.model.log_kernel_amp()

	def matrix_update(self, hyper=None):
		if hyper is not None:
			self.model.vec_to_param(hyper)
		self.mean_vec = self.train_y - self.model.mean(self.train_x)
		self.K_noise = self.model.kernel(self.train_x) + torch.diag(self.model.likelihood(self.train_x))

	def predict(self, pred_x, hyper=None):
		if hyper is not None:
			param_original = self.model.param_to_vec()
			self.matrix_update(hyper)
		k_pred_train = self.model.kernel(pred_x, self.train_x)

		shared_part = torch.gesv(k_pred_train.t(), self.K_noise)[0].t()
		pred_mean = torch.mm(shared_part, self.mean_vec) + self.model.mean(pred_x)
		pred_var = (self.model.kernel.forward_on_identical() - (shared_part * k_pred_train).sum(1, keepdim=True).clamp(min=0)).clamp(min=1e-8)

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
		bar = progressbar.ProgressBar(max_value=n_restarts)
		bar.update(0)
		vec_list = []
		nll_list = []
		for r in range(n_restarts):
			if r != 0:
				for m in self.model.children():
					m.reset_parameters()

			prev_loss = None
			n_step = 500
			###--------------------------------------------------###
			# This block can be modified to use other optimization method
			optimizer = optim.Adam(self.model.parameters(), lr=0.01)
			for _ in range(n_step):
				optimizer.zero_grad()
				loss = self.negative_log_likelihood(self.model.param_to_vec())
				curr_loss = loss.data.squeeze()[0]
				loss.backward(retain_graph=True)
				ftol = (prev_loss - curr_loss) / max(1, np.abs(prev_loss), np.abs(curr_loss)) if prev_loss is not None else 1
				prev_loss = curr_loss
				prev_param = self.model.param_to_vec()
				optimizer.step()
				if not self.stable_parameters() or self.model.out_of_bounds() or param_groups_nan(optimizer.param_groups) or (ftol < 1e-9):
					self.model.vec_to_param(prev_param)
					break
			###--------------------------------------------------###
			bar.update(r + 1)
			sys.stdout.flush()
			vec_list.append(self.model.param_to_vec())
			nll_list.append(self.negative_log_likelihood().data.squeeze()[0])
		best_ind = np.nanargmin(nll_list)
		self.model.vec_to_param(vec_list[best_ind])
		self.matrix_update(vec_list[best_ind])
		print('')
		return vec_list[best_ind].unsqueeze(0)

	def sampling(self, n_sample=10, n_burnin=100, n_thin=10):
		type_as_arg = list(self.model.likelihood.parameters())[0].data
		def logp(hyper):
			hyper_tensor = torch.from_numpy(hyper).type_as(type_as_arg)
			if self.model.out_of_bounds(hyper):
				return -np.inf
			param_original = self.model.param_to_vec()
			self.model.vec_to_param(hyper_tensor)
			if not self.stable_parameters():
				self.model.vec_to_param(param_original)
				return -np.inf
			prior = self.model.prior(hyper)
			likelihood = -self.negative_log_likelihood(param_original).data.squeeze()[0]
			return prior + likelihood
		hyper_torch = self.model.param_to_vec()
		hyper_numpy = (hyper_torch.cpu() if hyper_torch.is_cuda else hyper_torch).numpy()

		###--------------------------------------------------###
		# This block can be modified to use other sampling method
		start_time = time.time()
		sampler = smp.Slice(logp=logp, start={'hyper': hyper_numpy}, compwise=True)
		samples = sampler.sample(n_burnin + n_thin * n_sample, burn=n_burnin + n_thin - 1, thin=n_thin)
		print('Sampling : ' + time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time)))
		###--------------------------------------------------###
		self.model.vec_to_param(torch.from_numpy(samples[-1][0]).type_as(type_as_arg))
		self.matrix_update(torch.from_numpy(samples[-1][0]).type_as(type_as_arg))
		return torch.stack([torch.from_numpy(elm[0]) for elm in samples], 0).type_as(type_as_arg)


def param_groups_nan(param_groups):
	for group in param_groups:
		for p in group['params']:
			if (p.grad.data != p.grad.data).any():
				return True
	return False


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