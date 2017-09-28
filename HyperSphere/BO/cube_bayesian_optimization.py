import os
import os.path
import pickle
import sys
import time

import numpy as np

from HyperSphere.BO.acquisition_maximization import suggest, optimization_candidates, optimization_init_points
from HyperSphere.BO.utils.datafile_utils import EXPERIMENT_DIR
from HyperSphere.GP.inference.inference import Inference
from HyperSphere.GP.kernels.modules.matern52 import Matern52
from HyperSphere.GP.models.gp_regression import GPRegression
from HyperSphere.feature_map.functionals import *
from HyperSphere.test_functions.benchmarks import *


def cube_BO(n_eval=200, **kwargs):
	if 'path' in kwargs.keys():
		path = kwargs['path']
		if not os.path.exists(path):
			path = os.path.join(EXPERIMENT_DIR, path)
		model_filename = os.path.join(path, 'model.pt')
		data_config_filename = os.path.join(path, 'data_config.pkl')

		model = torch.load(model_filename)
		data_config_file = open(data_config_filename, 'r')
		for key, value in pickle.load(data_config_file).iteritems():
			exec (key + '=value')
		data_config_file.close()

		inference = Inference((x_input, output), model)
	else:
		func = kwargs['func']
		if func.dim == 0:
			ndim = kwargs['dim']
		else:
			ndim = func.dim
		dir_list = [elm for elm in os.listdir(EXPERIMENT_DIR) if os.path.isdir(os.path.join(EXPERIMENT_DIR, elm))]
		folder_name_root = func.__name__ + '_D' + str(ndim) + '_cube'
		folder_name_suffix = [elm[len(folder_name_root):] for elm in dir_list if elm[:len(folder_name_root)] == folder_name_root]
		next_ind = 1 + np.max([int(elm) for elm in folder_name_suffix if elm.isdigit()] + [-1])
		os.makedirs(os.path.join(EXPERIMENT_DIR, folder_name_root + str(next_ind)))
		model_filename = os.path.join(EXPERIMENT_DIR, folder_name_root + str(next_ind), 'model.pt')
		data_config_filename = os.path.join(EXPERIMENT_DIR, folder_name_root + str(next_ind), 'data_config.pkl')

		search_cube_half_sidelength = 1

		lower_bnd = -torch.ones(ndim) * search_cube_half_sidelength
		upper_bnd = torch.ones(ndim) * search_cube_half_sidelength

		x_input = Variable(torch.ger(torch.arange(0, 2), torch.ones(ndim)))
		output = Variable(torch.zeros(x_input.size(0), 1))
		for i in range(x_input.size(0)):
			output[i] = func(x_input[i])

		kernel_input_map = id_transform
		model = GPRegression(kernel=Matern52(ndim=kernel_input_map.dim_change(ndim), input_map=kernel_input_map))

		time_list = [time.time()] * 2
		elapse_list = [0, 0]

		inference = Inference((x_input, output), model)
		inference.model_param_init()
		inference.sampling(n_sample=100, n_burnin=0, n_thin=1)

	stored_variable_names = locals().keys()
	ignored_variable_names = ['kwargs', 'data_config_file', 'dir_list', 'folder_name_root', 'folder_name_suffix',
	                          'next_ind', 'model_filename', 'data_config_filename', 'i',
	                          'kernel_input_map', 'model', 'inference']
	stored_variable_names = set(stored_variable_names).difference(set(ignored_variable_names))

	for _ in range(3):
		print('Experiment based on data in ' + os.path.split(model_filename)[0])

	for _ in range(n_eval):
		inference = Inference((x_input, output), model)
		reference = torch.min(output)[0]
		sampled_params = inference.sampling(n_sample=10, n_burnin=0, n_thin=10)

		x0_cand = optimization_candidates(x_input, output, 0, 1)
		x0 = optimization_init_points(x0_cand, inference, sampled_params, reference=reference)
		next_x_point = suggest(inference, sampled_params, x0=x0, reference=reference)

		time_list.append(time.time())
		elapse_list.append(time_list[-1] - time_list[-2])

		x_input = torch.cat([x_input, next_x_point])
		output = torch.cat([output, func(x_input[-1])])

		rect_str = '/'.join(['%+.4f' % x_input.data[-1, i] for i in range(0, x_input.size(1))])
		time_str = time.strftime('%H:%M:%S', time.gmtime(time_list[-1])) + '(' + time.strftime('%H:%M:%S', time.gmtime(elapse_list[-1])) +')  '
		print(('\n%4d : ' % (x_input.size(0))) + time_str + rect_str + '    =>' + ('%12.6f (%12.6f)' % (output.data[-1].squeeze()[0], torch.min(output.data))))

		torch.save(model, model_filename)
		stored_variable = dict()
		for key in stored_variable_names:
			stored_variable[key] = locals()[key]
		f = open(data_config_filename, 'w')
		pickle.dump(stored_variable, f)
		f.close()

	for _ in range(3):
		print('Experiment based on data in ' + os.path.split(model_filename)[0])


if __name__ == '__main__':
	run_new = False
	path, suffix = os.path.split(sys.argv[1])
	if path == '' and not ('_D' in suffix):
		run_new = True
	if run_new:
		func = locals()[sys.argv[1]]
		n_eval = int(sys.argv[3]) if len(sys.argv) > 3 else 100
		if func.dim == 0:
			cube_BO(n_eval=n_eval, func=func, dim=int(sys.argv[2]))
		else:
			cube_BO(n_eval=n_eval, func=func)
	else:
		cube_BO(n_eval=int(sys.argv[2]), path=sys.argv[1])
