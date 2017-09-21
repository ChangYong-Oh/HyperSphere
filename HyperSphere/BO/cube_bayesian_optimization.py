import time

from torch.autograd import Variable

from HyperSphere.GP.models.gp_regression import GPRegression
from HyperSphere.GP.kernels.modules.matern52 import Matern52
from HyperSphere.GP.inference.inference import Inference
from HyperSphere.BO.acquisition_maximization import suggest
from HyperSphere.feature_map.functionals import *

from HyperSphere.test_functions.benchmarks import branin


def cube_BO(func, n_eval=200):
	n_spray = 10
	n_random = 10

	ndim = func.dim
	search_cube_half_sidelength = 1

	lower_bnd = -torch.ones(ndim) * search_cube_half_sidelength
	upper_bnd = torch.ones(ndim) * search_cube_half_sidelength

	rectangle_input = Variable(torch.ger(torch.arange(0, 2), torch.ones(ndim)))
	output = Variable(torch.zeros(rectangle_input.size(0), 1))
	for i in range(rectangle_input.size(0)):
		output[i] = func(rectangle_input[i])

	kernel = Matern52(ndim=ndim)
	model = GPRegression(kernel=kernel)
	model.kernel.log_amp.data = torch.std(output).log().data + 1e-4
	model.kernel.log_ls.data.fill_(0)
	model.mean.const_mean.data.fill_(torch.mean(output.data))
	model.likelihood.log_noise_var.data.fill_(-3)

	time_list = [time.time()] * 2
	elapes_list = [0, 0]

	inference = Inference((rectangle_input, output), model)
	inference.sampling(n_sample=100, n_burnin=0, n_thin=1)

	for e in range(output.numel(), n_eval):
		inference = Inference((rectangle_input, output), model)
		learned_params = inference.sampling(n_sample=10, n_burnin=0, n_thin=10)

		_, min_ind = torch.min(output.data, 0)
		x0_spray = rectangle_input.data[min_ind].view(1, -1).repeat(n_spray, 1) + rectangle_input.data.new(n_spray, ndim).normal_() * 0.001 * (upper_bnd - lower_bnd)
		x0_random = rectangle_input.data.new(n_random, ndim).uniform_() * (upper_bnd - lower_bnd) + lower_bnd
		x0 = torch.cat([x0_spray, x0_random], 0)
		x0[x0 < lower_bnd] = lower_bnd.view(1, -1).repeat(n_spray + n_random, 1)[x0 < lower_bnd]
		x0[x0 > upper_bnd] = upper_bnd.view(1, -1).repeat(n_spray + n_random, 1)[x0 > upper_bnd]
		next_eval_point = suggest(inference, learned_params, x0=x0, reference=torch.min(output)[0], bounds=(lower_bnd, upper_bnd))

		time_list.append(time.time())
		elapes_list.append(time_list[-1] - time_list[-2])

		rectangle_input = torch.cat([rectangle_input, next_eval_point])
		output = torch.cat([output, func(rectangle_input[-1])])

		for d in range(rectangle_input.size(0)):
			rect_str = '/'.join(['%+.4f' % rectangle_input.data[d, i] for i in range(0, rectangle_input.size(1))])
			time_str = time.strftime('%H:%M:%S', time.gmtime(time_list[d])) + '(' + time.strftime('%H:%M:%S', time.gmtime(elapes_list[d])) +')  '
			print(('%4d : ' % (d+1)) + time_str + rect_str + '    =>' + ('%12.6f' % output.data[d].squeeze()[0]))


if __name__ == '__main__':
	cube_BO(branin, n_eval=100)
