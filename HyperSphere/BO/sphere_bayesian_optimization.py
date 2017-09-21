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

from HyperSphere.test_functions.benchmarks import branin

from HyperSphere.BO.bayesian_optimization_utils import model_param_init, optimization_init_points


def sphere_BO(func, n_eval=200):
	n_spray = 10
	n_random = 10

	ndim = func.dim
	search_rphi_radius = 1.0

	rphi_sidelength = Variable(torch.ones(ndim) * math.pi)
	rphi_sidelength.data[0] = search_rphi_radius
	rphi_sidelength.data[-1] *= 2

	rectangle_input = Variable(torch.zeros(2, ndim))
	rectangle_input.data[1, -2] = -search_rphi_radius / 2.0
	rphi_input = rect2spherical(rectangle_input)
	rphi_input[rphi_input != rphi_input] = 0
	phi_input = rphi_input / search_rphi_radius
	phi_input[:, 0] = torch.asin(phi_input[:, 0]) * 2 / math.pi

	output = Variable(torch.zeros(rectangle_input.size(0), 1))
	for i in range(rectangle_input.size(0)):
		output[i] = func(rectangle_input[i])

	kernel_input_map = phi_periodize_one

	kernel = Matern52(ndim=ndim + kernel_input_map.dim_change, input_map=kernel_input_map)
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
		next_eval_point = suggest(inference, learned_params, x0=phi0, reference=torch.min(output)[0])
		next_eval_point[0] = torch.fmod(torch.abs(next_eval_point[0]), 2)
		next_eval_point[0][next_eval_point[0] > 1] = 2 - next_eval_point[0][next_eval_point[0] > 1]

		time_list.append(time.time())
		elapes_list.append(time_list[-1] - time_list[-2])

		phi_input = torch.cat([phi_input, next_eval_point])
		rphi_input = torch.cat([torch.sin(phi_input[:, 0:1] * math.pi / 2), phi_input[:, 1:]], 1) * search_rphi_radius
		x_input = spherical2rect(rphi_input)
		output = torch.cat([output, func(x_input[-1])])

		print('')
		for d in range(rphi_input.size(0)):
			sphr_str = ('%+.4f/' % rphi_input.data[d, 0]) + '/'.join(['%+.3fpi' % (rphi_input.data[d, i]/math.pi) for i in range(1, rphi_input.size(1))])
			rect_str = '/'.join(['%+.4f' % x_input.data[d, i] for i in range(0, x_input.size(1))])
			time_str = time.strftime('%H:%M:%S', time.gmtime(time_list[d])) + '(' + time.strftime('%H:%M:%S', time.gmtime(elapes_list[d])) +')  '
			print(('%4d : ' % (d+1)) + time_str + rect_str + ' & ' + sphr_str + '    =>' + ('%12.6f' % output.data[d].squeeze()[0]))


if __name__ == '__main__':
	sphere_BO(branin, n_eval=100)
