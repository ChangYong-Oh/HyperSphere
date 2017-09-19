import numpy as np

import torch
from torch.autograd import Variable

from HyperSphere.coordinate.transformation import rect2spherical, spherical2rect
from HyperSphere.GP.models.gp_regression import GPRegression
from HyperSphere.GP.kernels.modules.matern52 import Matern52
from HyperSphere.GP.inference.inference import Inference
from HyperSphere.BO.acquisition_maximization import suggest
from HyperSphere.feature_map.functionals import *

from HyperSphere.test_functions.benchmarks import branin


def spherical_BO(func, n_eval=200):
	ndim = func.dim
	search_cube_half_sidelength = 1
	search_sphere_radius = 1
	rectangular_input = torch.ger(torch.ones(ndim), torch.FloatTensor([0, min(search_sphere_radius / ndim ** 0.5, search_cube_half_sidelength)]))
	output = torch.zeros(rectangular_input.size(0), 1)
	for i in range(rectangular_input.size(0)):
		output[i] = func(rectangular_input[i])
	spherical_input = rect2spherical(rectangular_input)

	kernel_input_map = periodize_one

	kernel = Matern52(ndim=ndim + kernel_input_map.dim_change)
	model = GPRegression(kernel=kernel)

	for e in range(n_eval):
		inference = Inference((spherical_input, output), model)
		learned_params = inference.sampling(n_sample=10, n_burnin=100, n_thin=10)
		next_eval_point = suggest(inference, learned_params, reference=torch.min(output)[0]).data

		spherical_input = torch.cat([spherical_input, next_eval_point])
		rectangular_input = spherical2rect(spherical_input)
		output = torch.cat([output, func(rectangular_input[-1])])


if __name__ == '__main__':
	spherical_BO(branin, n_eval=10)
