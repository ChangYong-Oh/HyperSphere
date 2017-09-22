import time

from torch.autograd import Variable

from HyperSphere.GP.models.gp_regression import GPRegression
from HyperSphere.GP.kernels.modules.matern52 import Matern52
from HyperSphere.GP.inference.inference import Inference
from HyperSphere.BO.acquisition_maximization import suggest
from HyperSphere.feature_map.functionals import *

from HyperSphere.test_functions.benchmarks import branin

from HyperSphere.BO.bayesian_optimization_utils import model_param_init, optimization_init_points


def cube_BO(func, n_eval=200, **kwargs):
	n_spray = 10
	n_random = 10

	if func.dim == 0:
		assert 'dim' in kwargs.keys()
		ndim = kwargs['dim']
	else:
		ndim = func.dim
	search_cube_half_sidelength = 1

	lower_bnd = -torch.ones(ndim) * search_cube_half_sidelength
	upper_bnd = torch.ones(ndim) * search_cube_half_sidelength

	x_input = Variable(torch.ger(torch.arange(0, 2), torch.ones(ndim)))
	output = Variable(torch.zeros(x_input.size(0), 1))
	for i in range(x_input.size(0)):
		output[i] = func(x_input[i])

	kernel = Matern52(ndim=ndim)
	model = GPRegression(kernel=kernel)
	model_param_init(model, output)

	time_list = [time.time()] * 2
	elapes_list = [0, 0]

	inference = Inference((x_input, output), model)
	inference.sampling(n_sample=100, n_burnin=0, n_thin=1)

	for e in range(output.numel(), n_eval):
		inference = Inference((x_input, output), model)
		learned_params = inference.sampling(n_sample=10, n_burnin=0, n_thin=10)

		x0 = optimization_init_points(x_input, output, lower_bnd, upper_bnd, n_spray=n_spray, n_random=n_random)
		next_eval_point = suggest(inference, learned_params, x0=x0, reference=torch.min(output)[0], bounds=(lower_bnd, upper_bnd))

		time_list.append(time.time())
		elapes_list.append(time_list[-1] - time_list[-2])

		x_input = torch.cat([x_input, next_eval_point])
		output = torch.cat([output, func(x_input[-1])])

		for d in range(x_input.size(0)):
			rect_str = '/'.join(['%+.4f' % x_input.data[d, i] for i in range(0, x_input.size(1))])
			time_str = time.strftime('%H:%M:%S', time.gmtime(time_list[d])) + '(' + time.strftime('%H:%M:%S', time.gmtime(elapes_list[d])) +')  '
			print(('%4d : ' % (d+1)) + time_str + rect_str + '    =>' + ('%12.6f (%12.6f)' % (output.data[d].squeeze()[0], torch.min(output[:d+1].data))))


if __name__ == '__main__':
	cube_BO(branin, n_eval=200)
