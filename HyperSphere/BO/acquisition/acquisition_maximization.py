import copy

import numpy as np
import progressbar
import torch
import torch.optim as optim
from torch.autograd import Variable, grad

from HyperSphere.BO.acquisition.acquisition_functions import expected_improvement
from HyperSphere.BO.utils.sobol import sobol_generate

N_SOBOL = 10000
N_SPRAY = 10
N_INIT = 20


def suggest(inference, param_samples, x0, acquisition_function=expected_improvement, bounds=None, **kwargs):
	x = Variable(inference.train_x.data.new(1, inference.train_x.size(1)), requires_grad=True)
	if bounds is not None:
		if not hasattr(bounds, '__call__'):
			def out_of_bounds(x):
				return (x.data < bounds[0]).any() or (x.data > bounds[1]).any()
		else:
			out_of_bounds = bounds

	# for multi process, https://discuss.pytorch.org/t/copying-nn-modules-without-shared-memory/113
	bar = progressbar.ProgressBar(max_value=x0.size(0))
	bar.update(0)
	n_step = 500
	local_optima = []
	optima_value = []
	for i in range(x0.size(0)):
		x.data = x0[i].view(1, -1)
		prev_loss = None
		###--------------------------------------------------###
		# This block can be modified to use other optimization method
		optimizer = optim.Adam([x], lr=0.01)
		for _ in range(n_step):
			optimizer.zero_grad()
			loss = -acquisition(x, inference, param_samples, acquisition_function=acquisition_function, **kwargs)
			curr_loss = loss.data.squeeze()[0]
			x.grad = grad([loss], [x], retain_graph=True)[0]
			ftol = (prev_loss - curr_loss)/max(1, np.abs(prev_loss), np.abs(curr_loss)) if prev_loss is not None else 1
			if (x.grad.data != x.grad.data).any() or (ftol < 1e-9):
				break
			prev_x = x.data.clone()
			prev_loss = curr_loss
			optimizer.step()
			if bounds is not None and out_of_bounds(x):
				x.data = prev_x
				break
		###--------------------------------------------------###
		bar.update(i+1)
		local_optima.append(x.data.clone())
		optima_value.append(-acquisition(x, inference, param_samples, acquisition_function=acquisition_function, **kwargs).data.squeeze()[0])
	return local_optima[np.nanargmin(optima_value)]


def deepcopy_inference(inference, param_samples):
	inferences = []
	for s in range(param_samples.size(0)):
		model = copy.deepcopy(inference.model)
		model.vec_to_param(param_samples[s])
		inferences.append(inference.__class__((inference.train_x, inference.train_y), model))
	return inferences


def acquisition(x, inference, param_samples, acquisition_function=expected_improvement, **kwargs):
	inferences = deepcopy_inference(inference, param_samples)
	acquisition_sample_list = []
	for s in range(len(inferences)):
		pred_mean_sample, pred_var_sample = inferences[s].predict(x)
		acquisition_sample_list.append(acquisition_function(pred_mean_sample[:, 0], pred_var_sample[:, 0], **kwargs))
	return torch.stack(acquisition_sample_list, 1).sum(1, keepdim=True)


def optimization_candidates(input, output, lower_bnd, upper_bnd):
	ndim = input.size(1)
	_, min_ind = torch.min(output.data, 0)
	x0_spray_best = input.data[min_ind].view(1, -1).repeat(N_SPRAY, 1) + input.data.new(N_SPRAY, ndim).normal_() * 0.001 * (upper_bnd - lower_bnd)
	# x0_spray_new = input.data[-1].view(1, -1).repeat(N_SPRAY, 1) + input.data.new(N_SPRAY, ndim).normal_() * 0.001 * (upper_bnd - lower_bnd)
	# x0_spray = torch.cat([x0_spray_best, x0_spray_new], 0)
	x0_spray = x0_spray_best
	if hasattr(lower_bnd, 'size'):
		x0_spray[x0_spray < lower_bnd] = 2 * lower_bnd.view(1, -1).repeat(2 * N_SPRAY, 1) - x0_spray[x0_spray < lower_bnd]
	else:
		x0_spray[x0_spray < lower_bnd] = 2 * lower_bnd - x0_spray[x0_spray < lower_bnd]
	if hasattr(upper_bnd, 'size'):
		x0_spray[x0_spray > upper_bnd] = 2 * upper_bnd.view(1, -1).repeat(2 * N_SPRAY, 1) - x0_spray[x0_spray > upper_bnd]
	else:
		x0_spray[x0_spray > upper_bnd] = 2 * upper_bnd - x0_spray[x0_spray > upper_bnd]

	x0_sobol = sobol_generate(ndim, N_SOBOL, np.random.randint(0, N_SOBOL)).type_as(input.data) * (upper_bnd - lower_bnd) + lower_bnd
	x0 = torch.cat([x0_spray, input.data, x0_sobol], 0)

	return Variable(x0)


def optimization_init_points(candidates, inference, param_samples, acquisition_function=expected_improvement, **kwargs):
	ndim = candidates.size(1)
	acq_value = acquisition(candidates, inference, param_samples, acquisition_function, **kwargs).data
	nonnan_ind = acq_value == acq_value
	acq_value = acq_value[nonnan_ind]
	init_points = candidates.data[nonnan_ind.view(-1, 1).repeat(1, ndim)].view(-1, ndim)
	_, sort_ind = torch.sort(acq_value, 0, descending=True)
	is_maximum = acq_value == acq_value[sort_ind[0]]
	n_equal_maximum = torch.sum(is_maximum)
	if n_equal_maximum > N_INIT:
		shuffled_ind = torch.sort(torch.randn(n_equal_maximum), 0)[1]
		return init_points[is_maximum.view(-1, 1).repeat(1, ndim)].view(-1, ndim)[(shuffled_ind < N_INIT).view(-1, 1).repeat(1, ndim)].view(-1, ndim)
	else:
		return init_points[sort_ind][:N_INIT]


def one_dim_plotting(ax1, ax2, inference, param_samples, color, ls='-', label='', title_str=''):
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
	ax1.plot(pred_x.numpy().flatten(), pred_mean.numpy().flatten(), color=color)
	ax1.fill_between(pred_x.numpy().flatten(), (pred_mean - 1.96 * pred_std).numpy().flatten(),
	                 (pred_mean + 1.96 * pred_std).numpy().flatten(), facecolor=color, alpha=0.2)
	ax1.set_title(title_str + '\n%.4E' % nll)
	acq = acquisition(Variable(pred_x), inference, param_samples, acquisition_function=expected_improvement,
	                  reference=reference).data
	# next_point = suggest(inference, param_samples_sampling, reference=reference).numpy()
	# ax2.fill_between(pred_x.numpy().flatten(), 0, acq.numpy().flatten(), color=color, alpha=0.2, label=label)
	ax2.plot(pred_x.numpy(), acq.numpy(), color=color, ls=ls, alpha=1.0, label=label)
	ax2.legend()
	# ax2.axvline(next_point, color=color, ls='--', alpha=0.5)


if __name__ == '__main__':
	from HyperSphere.GP.kernels.modules.squared_exponential import SquaredExponentialKernel
	from HyperSphere.GP.models.gp_regression import GPRegression
	from HyperSphere.GP.inference.inference import Inference
	import matplotlib.pyplot as plt

	ndata = 6
	ndim = 1
	model_for_generating = GPRegression(kernel=SquaredExponentialKernel(ndim))
	train_x = Variable(torch.FloatTensor(ndata, ndim).uniform_(-2, 2))
	chol_L = torch.potrf(
		(model_for_generating.kernel(train_x) + torch.diag(model_for_generating.likelihood(train_x))).data, upper=False)
	train_y = model_for_generating.mean(train_x) + Variable(torch.mm(chol_L, torch.randn(ndata, 1)))
	# train_y = torch.sin(2 * math.pi * torch.sum(train_x, 1, keepdim=True)) + Variable(torch.FloatTensor(train_x.size(0), 1).normal_())
	train_data = (train_x, train_y)
	param_original = model_for_generating.param_to_vec()
	reference = torch.min(train_y.data)

	model_for_learning = GPRegression(kernel=SquaredExponentialKernel(ndim))
	inference = Inference(train_data, model_for_learning)
	model_for_learning.vec_to_param(param_original)
	param_samples_learning = inference.learning(n_restarts=10)
	model_for_learning.vec_to_param(param_original)
	param_samples_sampling = inference.sampling(n_sample=5, n_burnin=200, n_thin=10)

	if ndim == 1:
		ax11 = plt.subplot(221)
		ax11.plot(train_x.data.numpy().flatten(), train_y.data.numpy().flatten(), 'k*')
		ax11.axhline(reference, ls='--', alpha=0.5)
		ax12 = plt.subplot(222, sharex=ax11, sharey=ax11)
		ax12.plot(train_x.data.numpy().flatten(), train_y.data.numpy().flatten(), 'k*')
		ax12.axhline(reference, ls='--', alpha=0.5)
		ax21 = plt.subplot(223, sharex=ax11)
		ax22 = plt.subplot(224, sharex=ax11)

		# model_for_learning.elastic_vec_to_param(param_original, func)
		# param_original_elastic = model_for_learning.param_to_vec()
		# one_dim_plotting(axes[0, 0], axes[1, 0], inference, param_original, 'b')
		# one_dim_plotting(axes[0, 1], axes[1, 1], inference, param_original_elastic, 'r')

		step = 3
		for i in range(-step, step+1):
			if i == 0:
				one_dim_plotting(ax11, ax21, inference, param_samples_learning, color='k', label='target')
				one_dim_plotting(ax12, ax22, inference, param_samples_sampling, color='k', label='target')
			else:
				color = np.random.rand(3)
				func = lambda x: x + np.log(2) * i

				model_for_learning.elastic_vec_to_param(param_samples_learning[0], func)
				param_samples_learning_elastic = model_for_learning.param_to_vec()
				one_dim_plotting(ax11, ax21, inference, param_samples_learning_elastic, color, label=str(i))

				param_samples_sampling_elastic = param_samples_sampling.clone()
				for s in range(param_samples_sampling.size(0)):
					model_for_learning.elastic_vec_to_param(param_samples_sampling[s], func)
					param_samples_sampling_elastic[s] = model_for_learning.param_to_vec()
				one_dim_plotting(ax12, ax22, inference, param_samples_sampling_elastic, color, label=str(i))

		plt.show()



