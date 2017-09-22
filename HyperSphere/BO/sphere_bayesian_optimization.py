import time
import math

import torch
from torch.autograd import Variable

from HyperSphere.coordinate.transformation import rect2spherical, spherical2rect
from HyperSphere.GP.models.gp_regression import GPRegression
from HyperSphere.GP.kernels.modules.matern52 import Matern52
from HyperSphere.GP.inference.inference import Inference
from HyperSphere.BO.acquisition_maximization import suggest
from HyperSphere.feature_map.functionals import phi_periodize, phi_periodize_lp, phi_periodize_one, phi_periodize_sin

from HyperSphere.test_functions.benchmarks import branin, hartmann6

from HyperSphere.BO.bayesian_optimization_utils import model_param_init, optimization_init_points


def sphere_BO(func, n_eval=200, **kwargs):
	n_spray = 10
	n_random = 10

	if func.dim == 0:
		assert 'dim' in kwargs.keys()
		ndim = kwargs['dim']
	else:
		ndim = func.dim
	search_sphere_radius = ndim ** 0.5

	rphi_sidelength = Variable(torch.ones(ndim) * math.pi)
	rphi_sidelength.data[0] = search_sphere_radius
	rphi_sidelength.data[-1] *= 2

	x_input = Variable(torch.zeros(2, ndim))
	x_input.data[1, -2] = -search_sphere_radius
	rphi_input = rect2spherical(x_input)
	rphi_input[rphi_input != rphi_input] = 0
	phi_input = rphi_input / rphi_sidelength
	phi_input[:, 0] = torch.acos(1 - 2 * phi_input[:, 0]) / math.pi

	output = Variable(torch.zeros(x_input.size(0), 1))
	for i in range(x_input.size(0)):
		output[i] = func(x_input[i])

	kernel_input_map = phi_periodize_one

	kernel = Matern52(ndim=kernel_input_map.dim_change(ndim), input_map=kernel_input_map)
	model = GPRegression(kernel=kernel)
	model_param_init(model, output)

	time_list = [time.time()] * 2
	elapes_list = [0, 0]

	inference = Inference((rphi_input, output), model)
	inference.sampling(n_sample=100, n_burnin=0, n_thin=1)

	for e in range(n_eval):
		inference = Inference((phi_input, output), model)
		learned_params = inference.sampling(n_sample=10, n_burnin=0, n_thin=10)

		phi0 = optimization_init_points(phi_input, output, 0, 1, n_spray=n_spray, n_random=n_random)
		next_phi_point = suggest(inference, learned_params, x0=phi0, reference=torch.min(output)[0])
		next_phi_point[0, :-1] = torch.fmod(torch.fmod(torch.abs(next_phi_point[0, :-1]), 2) + 2, 2)
		next_phi_point[0, -1:] = torch.fmod(torch.fmod(torch.abs(next_phi_point[0, -1:]), 1) + 1, 1)

		# only using 2pi periodicity and spherical transformation property(smooth extension)
		# kernel_input_map should only assume that it is 2 pi periodic
		# next_rphi_point = next_phi_point * math.pi
		# next_rphi_point[0, -1] *= 2
		# next_rphi_point[0, 0] = 0.5 * search_sphere_radius * (1 - torch.cos(next_rphi_point[0, 0]))
		# next_phi_point = rect2spherical(spherical2rect(next_rphi_point))
		# next_phi_point[0, 0] = torch.acos(1 - 2 * next_phi_point[0, 0] / search_sphere_radius)

		# using pi reflection
		# kernel_input_map assumes pi reflection
		next_phi_point[0, :-1][next_phi_point[0, :-1] > 1] = 2 - next_phi_point[0, :-1][next_phi_point[0, :-1] > 1]
		next_rphi_point = next_phi_point * math.pi
		next_rphi_point[0, -1] *= 2
		next_rphi_point[0, 0:1] = 0.5 * search_sphere_radius * (1 - torch.cos(next_rphi_point[0, 0:1]))

		time_list.append(time.time())
		elapes_list.append(time_list[-1] - time_list[-2])

		phi_input = torch.cat([phi_input, Variable(next_phi_point)], 0)
		rphi_input = torch.cat([0.5 * (1 - torch.cos(phi_input[:, 0:1] * math.pi)), phi_input[:, 1:]], 1) * rphi_sidelength
		x_input = spherical2rect(rphi_input)
		output = torch.cat([output, func(x_input[-1])])

		print('')
		for d in range(rphi_input.size(0)):
			sphr_str = ('%+.4f/' % rphi_input.data[d, 0]) + '/'.join(['%+.3fpi' % (rphi_input.data[d, i]/math.pi) for i in range(1, rphi_input.size(1))])
			rect_str = '/'.join(['%+.4f' % x_input.data[d, i] for i in range(0, x_input.size(1))])
			time_str = time.strftime('%H:%M:%S', time.gmtime(time_list[d])) + '(' + time.strftime('%H:%M:%S', time.gmtime(elapes_list[d])) +')  '
			print(('%4d : ' % (d+1)) + time_str + rect_str + ' & ' + sphr_str + '    =>' + ('%12.6f (%12.6f)' % (output.data[d].squeeze()[0], torch.min(output[:d+1].data))))


if __name__ == '__main__':
	sphere_BO(hartmann6, n_eval=200)
