import math

import torch
from torch.autograd import Variable
import torch.autograd as autograd
import torch.optim as optim

from HyperSphere.GP.inference.inference import Inference
from HyperSphere.BO.acquisition_functions import expected_improvement


def suggest(inference, param_samples, acquisition_function=expected_improvement, **kwargs):
	assert isinstance(inference, Inference)

	x = Variable(torch.FloatTensor(1, inference.train_x.size()[1]), requires_grad=True)

	###--------------------------------------------------###
	# This block can be modified to use other optimization method
	n_step = 100
	optimizer = optim.Adam([x], lr=0.01)
	for s in range(n_step):
		optimizer.zero_grad()
		x.grad = autograd.grad(-acquisition(x, inference, param_samples, acquisition_function=acquisition_function, **kwargs), x)[0]
		optimizer.step()
	###--------------------------------------------------###
	return x.data


def acquisition(x, inference, param_samples, acquisition_function=expected_improvement, **kwargs):
	acquisition_sample_list = []
	for s in range(param_samples.size()[0]):
		pred_mean_sample, pred_var_sample = inference.predict(x, param_samples[s])
		acquisition_sample_list.append(acquisition_function(pred_mean_sample, pred_var_sample, **kwargs))
	return torch.cat(acquisition_sample_list).sum()


if __name__ == '__main__':
	from HyperSphere.GP.kernels.modules.squared_exponential import SquaredExponentialKernel
	from HyperSphere.GP.models.gp_regression import GPRegression
	import matplotlib.pyplot as plt

	ndata = 50
	ndim = 1
	model_for_generating = GPRegression(kernel=SquaredExponentialKernel(ndim))
	train_x = Variable(torch.FloatTensor(ndata, ndim).uniform_(-2, 2))
	chol_L = torch.potrf(
		(model_for_generating.kernel(train_x) + torch.diag(model_for_generating.likelihood(train_x))).data, upper=False)
	train_y = torch.sin(2 * math.pi * torch.sum(train_x ** 2, 1, keepdim=True) ** 0.5) + model_for_generating.mean(
		train_x) + Variable(torch.mm(chol_L, torch.randn(ndata, 1)))
	train_data = (train_x, train_y)
	generated_nll = Inference(train_data, model_for_generating).negative_log_likelihood().data[0, 0]

	model_for_learning = GPRegression(kernel=SquaredExponentialKernel(ndim))
	inference = Inference(train_data, model_for_learning)
	param_original = model_for_generating.param_to_vec()
	inference.reset_parameters()
	param_samples_learning = inference.learning(n_restarts=10)
	inference.reset_parameters()
	param_samples_sampling = inference.sampling()

	if ndim == 1:
		pred_x = torch.linspace(-2.5, 2.5, 100).view(-1, 1)
		fig, axes = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True)

		pred_mean, pred_var = inference.predict(Variable(pred_x), param_original)
		pred_std = torch.sqrt(pred_var)
		pred_mean = pred_mean.data
		pred_var = pred_var.data
		pred_std = pred_std.data
		axes[0, 0].plot(train_x.data.numpy().flatten(), train_y.data.numpy().flatten(), '+')
		axes[0, 0].plot(pred_x.numpy().flatten(), pred_mean.numpy().flatten(), 'b')
		axes[0, 0].fill_between(pred_x.numpy().flatten(), (pred_mean - pred_std).numpy().flatten(),
		                     (pred_mean + pred_std).numpy().flatten(), facecolor='green', alpha=0.2)
		axes[0, 0].fill_between(pred_x.numpy().flatten(), (pred_mean - 1.96 * pred_std).numpy().flatten(),
		                     (pred_mean + 1.96 * pred_std).numpy().flatten(), facecolor='green', alpha=0.2)
		axes[0, 0].set_title('Original')
		acquisition = acquisition(Variable(pred_x), inference, param_original.unsqueeze(0), acquisition_function=expected_improvement,
		                          reference=torch.min(train_y)).data
		print(pred_x.size())
		print(acquisition.size())
		axes[1, 0].fill_between(pred_x.numpy().flatten(), 0, acquisition.numpy().flatten())

		pred_mean, pred_var = inference.predict(Variable(pred_x), param_samples_learning[0])
		pred_std = torch.sqrt(pred_var)
		pred_mean = pred_mean.data
		pred_var = pred_var.data
		pred_std = pred_std.data
		axes[0, 1].plot(train_x.data.numpy().flatten(), train_y.data.numpy().flatten(), '+')
		axes[0, 1].plot(pred_x.numpy().flatten(), pred_mean.numpy().flatten(), 'b')
		axes[0, 1].fill_between(pred_x.numpy().flatten(), (pred_mean - pred_std).numpy().flatten(),
		                     (pred_mean + pred_std).numpy().flatten(), facecolor='green', alpha=0.2)
		axes[0, 1].fill_between(pred_x.numpy().flatten(), (pred_mean - 1.96 * pred_std).numpy().flatten(),
		                     (pred_mean + 1.96 * pred_std).numpy().flatten(), facecolor='green', alpha=0.2)
		axes[0, 1].set_title('Optimized')
		acquisition = acquisition(Variable(pred_x), inference, param_samples_learning, acquisition_function=expected_improvement,
		                          reference=torch.min(train_y)).data
		axes[1, 1].fill_between(pred_x.numpy().flatten(), 0, acquisition.numpy().flatten())

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
		axes[0, 2].plot(train_x.data.numpy().flatten(), train_y.data.numpy().flatten(), '+')
		axes[0, 2].plot(pred_x.numpy().flatten(), pred_mean.numpy().flatten(), 'b')
		axes[0, 2].fill_between(pred_x.numpy().flatten(), (pred_mean - pred_std).numpy().flatten(),
		                     (pred_mean + pred_std).numpy().flatten(), facecolor='green', alpha=0.2)
		axes[0, 2].fill_between(pred_x.numpy().flatten(), (pred_mean - 1.96 * pred_std).numpy().flatten(),
		                     (pred_mean + 1.96 * pred_std).numpy().flatten(), facecolor='green', alpha=0.2)
		axes[0, 2].set_title('Sampled')
		acquisition = acquisition(Variable(pred_x), inference, param_samples_sampling, acquisition_function=expected_improvement,
		                          reference=torch.min(train_y)).data
		axes[1, 2].fill_between(pred_x.numpy().flatten(), 0, acquisition.numpy().flatten())

		plt.show()