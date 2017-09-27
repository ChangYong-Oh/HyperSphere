import os
import os.path
import pickle
import sys

import torch
from torch.autograd import Variable


EXPERIMENT_DIR = os.path.join('/'.join(os.path.realpath(__file__).split('/')[:-5]), 'Experiments/Hypersphere')


def model_param_init(model, output):
	model.kernel.log_amp.data = torch.std(output).log().data + 1e-4
	model.kernel.log_ls.data.fill_(0)
	model.mean.const_mean.data.fill_(torch.mean(output.data))
	model.likelihood.log_noise_var.data.fill_(-3)


def optimization_init_points(input, output, lower_bnd, upper_bnd, n_spray=10, n_random=10):
	ndim = input.size(1)
	_, min_ind = torch.min(output.data, 0)
	if n_spray > 0:
		x0_spray = input.data[min_ind].view(1, -1).repeat(n_spray, 1) + input.data.new(n_spray, ndim).normal_() * 0.001 * (upper_bnd - lower_bnd)
	x0_random = input.data.new(n_random, ndim).uniform_() * (upper_bnd - lower_bnd) + lower_bnd
	if n_spray > 0:
		x0 = torch.cat([x0_spray, x0_random], 0)
	else:
		x0 = x0_random
	if isinstance(lower_bnd, torch.Tensor):
		x0[x0 < lower_bnd] = lower_bnd.view(1, -1).repeat(n_spray + n_random, 1)[x0 < lower_bnd]
	elif isinstance(lower_bnd, float):
		x0[x0 < lower_bnd] = lower_bnd
	if isinstance(upper_bnd, torch.Tensor):
		x0[x0 > upper_bnd] = upper_bnd.view(1, -1).repeat(n_spray + n_random, 1)[x0 > upper_bnd]
	elif isinstance(upper_bnd, float):
		x0[x0 < upper_bnd] = upper_bnd
	return x0


def remove_last_evaluation(path):
	if not os.path.exists(path):
		path = os.path.join(EXPERIMENT_DIR, path)
	data_config_filename = os.path.join(path, 'data_config.pkl')
	data_config_file = open(data_config_filename, 'r')
	data_config = pickle.load(data_config_file)
	data_config_file.close()
	for key, value in data_config.iteritems():
		if isinstance(value, Variable) and value.dim() == 2:
			print('%12s : %4d-th evaluation %4d dimensional data whose sum is %.6E' % (key, value.size(0), value.size(1), (value.data if hasattr(value, 'data') else value).sum()))
	while True:
		sys.stdout.write('Want to remove this last evaluation?(YES/NO) : ')
		decision = sys.stdin.readline()[:-1]
		if decision == 'YES':
			for key, value in data_config.iteritems():
				if isinstance(value, Variable) and value.dim() == 2:
					data_config[key] = value[:-1]
			data_config_file = open(data_config_filename, 'w')
			pickle.dump(data_config, data_config_file)
			data_config_file.close()
			break
		elif decision == 'NO':
			break
		else:
			sys.stdout.write('Input YES or NO\n')


if __name__ == '__main__':
	remove_last_evaluation(sys.argv[1])