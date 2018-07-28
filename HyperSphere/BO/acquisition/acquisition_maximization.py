import copy
import time
import psutil

import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable, grad

from HyperSphere.BO.acquisition.acquisition_functions import expected_improvement
from HyperSphere.BO.utils.sobol import sobol_generate

N_SPREAD = 20000 # Number of sobol sequence points as candidates for good initial points
N_SPRAY = 10 # Number of random perturbations of current optimum
N_INIT = 20 # Number of initial points for acquisition function maximization
N_AVAILABLE_CORE = 8 # When there is this many available cpu cores new optimization is started
MAX_OPTIMIZATION_STEP = 500


def suggest(x0, reference, inferences, acquisition_function=expected_improvement, bounds=None, pool=None):
	max_step = MAX_OPTIMIZATION_STEP
	n_init = x0.size(0)

	start_time = time.time()
	print('Acqusition function optimization with %2d inits %s has begun' % (n_init, time.strftime('%H:%M:%S', time.gmtime(start_time))))

	# Parallel version and non-parallel version behave differently.
	if pool is not None:
		# pool = torch.multiprocessing.Pool(n_init) if parallel else None
		# results = [pool.apply_async(optimize, args=(max_step, x0[p], reference, inferences, acquisition_function, bounds)) for p in range(n_init)]

		results = []
		process_started = [False] * n_init
		process_running = [False] * n_init
		process_index = 0
		while process_started.count(False) > 0:
			cpu_usage = psutil.cpu_percent(0.2)
			run_more = (100.0 - cpu_usage) * float(psutil.cpu_count()) > 100.0 * N_AVAILABLE_CORE
			if run_more:
				results.append(pool.apply_async(optimize, args=(max_step, x0[process_index], reference, inferences, acquisition_function, bounds)))
				process_started[process_index] = True
				process_running[process_index] = True
				process_index += 1
		while [not res.ready() for res in results].count(True) > 0:
			time.sleep(1)

		return_values = [res.get() for res in results]
		local_optima, optima_value = zip(*return_values)
	else:
		local_optima = []
		optima_value = []
		for p in range(n_init):
			optimum_loc, optimum_value = optimize(max_step, x0[p], reference, inferences, acquisition_function, bounds)
			local_optima.append(optimum_loc)
			optima_value.append(optimum_value)

	end_time = time.time()
	print('Acqusition function optimization ended %s(%s)' % (time.strftime('%H:%M:%S', time.gmtime(end_time)), time.strftime('%H:%M:%S', time.gmtime(end_time - start_time))))

	suggestion = local_optima[np.nanargmin(optima_value)]
	mean, std, var, stdmax, varmax = mean_std_var(suggestion, inferences)
	return suggestion, mean, std, var, stdmax, varmax


def optimize(max_step, x0, reference, inferences, acquisition_function=expected_improvement, bounds=None):
	if bounds is not None:
		if not hasattr(bounds, '__call__'):
			def out_of_bounds(x_input):
				return (x_input.data < bounds[0]).any() or (x_input.data > bounds[1]).any()
		else:
			out_of_bounds = bounds

	x = Variable(x0.clone().view(1, -1), requires_grad=True)
	prev_loss = None
	###--------------------------------------------------###
	# This block can be modified to use other optimization method
	optimizer = optim.Adam([x], lr=0.01)
	for s in range(max_step):
		optimizer.zero_grad()
		loss = -acquisition(x, reference=reference, inferences=inferences, acquisition_function=acquisition_function, in_optimization=True)
		curr_loss = loss.data.squeeze()[0]
		x.grad = grad([loss], [x], retain_graph=True)[0]
		ftol = (prev_loss - curr_loss) / max(1, np.abs(prev_loss), np.abs(curr_loss)) if prev_loss is not None else 1
		if (x.grad.data != x.grad.data).any() or (ftol < 1e-9):
			break
		prev_x = x.data.clone()
		prev_loss = curr_loss
		optimizer.step()
		if bounds is not None and out_of_bounds(x):
			x.data = prev_x
			break
	###--------------------------------------------------###
	optimum_loc = x.clone()
	optimum_value = -acquisition(x, reference=reference, inferences=inferences, acquisition_function=acquisition_function, in_optimization=True)[0].data.squeeze()[0]
	return optimum_loc, optimum_value


def deepcopy_inference(inference, param_samples):
	inferences = []
	for s in range(param_samples.size(0)):
		model = copy.deepcopy(inference.model)
		deepcopied_inference = inference.__class__((inference.train_x, inference.train_y), model)
		deepcopied_inference.cholesky_update(param_samples[s])
		inferences.append(deepcopied_inference)
	return inferences


def acquisition(x, reference, inferences, acquisition_function=expected_improvement, in_optimization=False):
	acquisition_sample_list = []
	numerically_stable_list = []
	zero_pred_var_list = []
	for s in range(len(inferences)):
		pred_dist = inferences[s].predict(x, in_optimization=in_optimization)
		pred_mean_sample = pred_dist[0]
		pred_var_sample = pred_dist[1]
		numerically_stable_list.append(pred_dist[2])
		zero_pred_var_list.append(pred_dist[3])
		acquisition_sample_list.append(acquisition_function(pred_mean_sample[:, 0], pred_var_sample[:, 0], reference=reference))
	sample_info = (np.sum(numerically_stable_list), np.sum(zero_pred_var_list), len(numerically_stable_list))
	if in_optimization:
		return torch.stack(acquisition_sample_list, 1).sum(1, keepdim=True)
	else:
		return torch.stack(acquisition_sample_list, 1).sum(1, keepdim=True), sample_info


def mean_std_var(x, inferences):
	mean_sample_list = []
	std_sample_list = []
	var_sample_list = []
	stdmax_sample_list = []
	varmax_sample_list = []
	for s in range(len(inferences)):
		pred_dist = inferences[s].predict(x)
		pred_mean_sample = pred_dist[0]
		pred_var_sample = pred_dist[1]
		pred_std_sample = pred_var_sample ** 0.5
		varmax_sample = torch.exp(inferences[s].log_kernel_amp())
		stdmax_sample = varmax_sample ** 0.5
		mean_sample_list.append(pred_mean_sample.data)
		std_sample_list.append(pred_std_sample.data)
		var_sample_list.append(pred_var_sample.data)
		stdmax_sample_list.append(stdmax_sample.data)
		varmax_sample_list.append(varmax_sample.data)
	return torch.cat(mean_sample_list, 1).mean(1, keepdim=True),  \
	       torch.cat(std_sample_list, 1).mean(1, keepdim=True), \
	       torch.cat(var_sample_list, 1).mean(1, keepdim=True), \
	       torch.cat(stdmax_sample_list).mean(0, keepdim=True), \
	       torch.cat(varmax_sample_list).mean(0, keepdim=True)


def optimization_candidates(input, output, lower_bnd, upper_bnd):
	ndim = input.size(1)
	min_ind = torch.min(output.data, 0)[1]

	x0_spray = input.data[min_ind].view(1, -1).repeat(N_SPRAY, 1) + input.data.new(N_SPRAY, ndim).normal_() * 0.001 * (upper_bnd - lower_bnd)

	if hasattr(lower_bnd, 'size'):
		x0_spray[x0_spray < lower_bnd] = 2 * lower_bnd.view(1, -1).repeat(2 * N_SPRAY, 1) - x0_spray[x0_spray < lower_bnd]
	else:
		x0_spray[x0_spray < lower_bnd] = 2 * lower_bnd - x0_spray[x0_spray < lower_bnd]
	if hasattr(upper_bnd, 'size'):
		x0_spray[x0_spray > upper_bnd] = 2 * upper_bnd.view(1, -1).repeat(2 * N_SPRAY, 1) - x0_spray[x0_spray > upper_bnd]
	else:
		x0_spray[x0_spray > upper_bnd] = 2 * upper_bnd - x0_spray[x0_spray > upper_bnd]

	if ndim <= 1100:
		x0_spread = sobol_generate(ndim, N_SPREAD, np.random.randint(0, N_SPREAD)).type_as(input.data) * (upper_bnd - lower_bnd) + lower_bnd
	else:
		x0_spread = torch.FloatTensor(N_SPREAD, ndim).uniform_().type_as(input.data) * (upper_bnd - lower_bnd) + lower_bnd
	x0 = torch.cat([input.data, x0_spray, x0_spread], 0)
	nonzero_radius_mask = torch.sum(x0 ** 2, 1) > 0
	nonzero_radius_ind = torch.sort(nonzero_radius_mask, 0, descending=True)[1][:torch.sum(nonzero_radius_mask)]
	x0 = x0.index_select(0, nonzero_radius_ind)

	return Variable(x0)


def optimization_init_points(candidates, reference, inferences, acquisition_function=expected_improvement):
	start_time = time.time()
	ndim = candidates.size(1)
	acq_value, sample_info = acquisition(candidates, reference, inferences, acquisition_function, False)
	acq_value = acq_value.data
	nonnan_ind = acq_value == acq_value
	acq_value = acq_value[nonnan_ind]
	init_points = candidates.data[nonnan_ind.view(-1, 1).repeat(1, ndim)].view(-1, ndim)
	_, sort_ind = torch.sort(acq_value, 0, descending=True)
	is_maximum = acq_value == acq_value[sort_ind[0]]
	n_equal_maximum = torch.sum(is_maximum)
	print(('Initial points selection from %d points ' % candidates.size(0)) + time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time)))
	if n_equal_maximum > N_INIT:
		shuffled_ind = torch.sort(torch.randn(n_equal_maximum), 0)[1]
		return init_points[is_maximum.view(-1, 1).repeat(1, ndim)].view(-1, ndim)[(shuffled_ind < N_INIT).view(-1, 1).repeat(1, ndim)].view(-1, ndim), sample_info
	else:
		return init_points[sort_ind][:N_INIT], sample_info


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



