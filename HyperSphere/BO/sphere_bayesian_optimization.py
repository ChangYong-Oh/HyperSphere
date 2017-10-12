import os.path
import pickle
import sys
import time

import numpy as np

from HyperSphere.BO.acquisition.acquisition_maximization import suggest, optimization_candidates, \
	optimization_init_points
from HyperSphere.GP.inference.inference import Inference
from HyperSphere.BO.utils.datafile_utils import EXPERIMENT_DIR
from HyperSphere.GP.kernels.modules.matern52 import Matern52
from HyperSphere.GP.models.gp_regression import GPRegression
from HyperSphere.coordinate.transformation import rect2spherical, spherical2rect, phi2rphi, rphi2phi
from HyperSphere.feature_map.functionals import phi_smooth
from HyperSphere.test_functions.benchmarks import *


def sphere_BO(n_eval=200, **kwargs):
	if 'path' in kwargs.keys():
		path = kwargs['path']
		if not os.path.exists(path):
			path = os.path.join(EXPERIMENT_DIR, path)
		model_filename = os.path.join(path, 'model.pt')
		data_config_filename = os.path.join(path, 'data_config.pkl')

		model = torch.load(model_filename)
		data_config_file = open(data_config_filename, 'r')
		for key, value in pickle.load(data_config_file).iteritems():
			exec(key + '=value')
		data_config_file.close()

		inference = Inference((rphi_input, output), model)
	else:
		func = kwargs['func']
		if func.dim == 0:
			ndim = kwargs['dim']
		else:
			ndim = func.dim
		dir_list = [elm for elm in os.listdir(EXPERIMENT_DIR) if os.path.isdir(os.path.join(EXPERIMENT_DIR, elm))]
		folder_name_root = func.__name__ + '_D' + str(ndim) + '_sphere'
		folder_name_suffix = [elm[len(folder_name_root):] for elm in dir_list if elm[:len(folder_name_root)] == folder_name_root]
		next_ind = 1 + np.max([int(elm) for elm in folder_name_suffix if elm.isdigit()] + [-1])
		os.makedirs(os.path.join(EXPERIMENT_DIR, folder_name_root + str(next_ind)))
		model_filename = os.path.join(EXPERIMENT_DIR, folder_name_root + str(next_ind), 'model.pt')
		data_config_filename = os.path.join(EXPERIMENT_DIR, folder_name_root + str(next_ind), 'data_config.pkl')

		search_sphere_radius = ndim ** 0.5

		x_input = Variable(torch.stack([torch.zeros(ndim), torch.ones(ndim)]))
		rphi_input = rect2spherical(x_input)
		phi_input = rphi2phi(rphi_input, search_sphere_radius)

		output = Variable(torch.zeros(x_input.size(0), 1))
		for i in range(x_input.size(0)):
			output[i] = func(x_input[i])

		kernel_input_map = phi_smooth
		model = GPRegression(kernel=Matern52(ndim=kernel_input_map.dim_change(ndim), input_map=kernel_input_map))

		time_list = [time.time()] * 2
		elapse_list = [0, 0]

		inference = Inference((phi_input, output), model)
		inference.init_parameters()
		inference.sampling(n_sample=1, n_burnin=99, n_thin=1)

	stored_variable_names = locals().keys()
	ignored_variable_names = ['kwargs', 'data_config_file', 'dir_list', 'folder_name_root', 'folder_name_suffix',
	                          'next_ind', 'model_filename', 'data_config_filename', 'i',
	                          'kernel_input_map', 'model', 'inference']
	stored_variable_names = set(stored_variable_names).difference(set(ignored_variable_names))

	for _ in range(3):
		print('Experiment based on data in ' + os.path.split(model_filename)[0])

	rotation_mat = Variable(torch.eye(ndim)).type_as(x_input)
	for _ in range(n_eval):
		rotation_mat, _ = torch.qr(torch.randn(ndim, ndim))
		rotation_mat = Variable(rotation_mat).type_as(x_input)

		rphi_input = rect2spherical(x_input, rotation_mat)
		phi_input = phi2rphi(rphi_input, radius=search_sphere_radius)
		inference = ShadowInference((phi_input, output), model)

		log_ls_data = inference.model.kernel.log_ls.data.clone()
		rotated_log_ls_data = torch.cat([log_ls_data[:1], rotation_mat.data.mv(log_ls_data[1:].exp()).abs().log()])
		inference.model.kernel.log_ls.data = rotated_log_ls_data

		reference = torch.min(output)[0]
		# gp_hyper_params = inference.learning(n_restarts=20)
		gp_hyper_params = inference.sampling(n_sample=10, n_burnin=0, n_thin=10)

		x0_cand = optimization_candidates(x_input, output, -1, 1)
		rphi0_cand = rect2spherical(x0_cand, rotation_mat)
		phi0_cand = phi2rphi(rphi0_cand, radius=search_sphere_radius)
		phi0 = optimization_init_points(phi0_cand, inference, gp_hyper_params, reference=reference)
		next_phi_point = suggest(inference, gp_hyper_params, x0=phi0, reference=reference)
		next_phi_point[0, :-1] = torch.fmod(torch.fmod(next_phi_point[0, :-1], 2) + 2, 2)
		next_phi_point[0, -1:] = torch.fmod(torch.fmod(next_phi_point[0, -1:], 1) + 1, 1)

		# only using 2pi periodicity and spherical transformation property(smooth extension)
		# kernel_input_map should only assume that it is 2 pi periodic
		next_rphi_point = rect2spherical(spherical2rect(phi2rphi(next_phi_point, radius=search_sphere_radius)))
		next_phi_point = rphi2phi(next_rphi_point, radius=search_sphere_radius)

		# using pi reflection
		# kernel_input_map assumes pi reflection
		# next_phi_point[0, :-1][next_phi_point[0, :-1] > 1] = 2 - next_phi_point[0, :-1][next_phi_point[0, :-1] > 1]
		# next_rphi_point = phi2rphi(next_phi_point, radius=search_sphere_radius)

		time_list.append(time.time())
		elapse_list.append(time_list[-1] - time_list[-2])

		phi_input = torch.cat([phi_input, Variable(next_phi_point)], 0)
		rphi_input = phi2rphi(phi_input, radius=search_sphere_radius)
		x_input = spherical2rect(rphi_input, rotation_mat)
		output = torch.cat([output, func(x_input[-1])])

		rotated_log_ls_data = inference.model.kernel.log_ls.data.clone()
		log_ls_data = torch.cat([rotated_log_ls_data[:1], rotation_mat.data.t().mv(rotated_log_ls_data[1:].exp()).abs().log()])
		inference.model.kernel.log_ls.data = log_ls_data

		sphr_str = ('%+.4f/' % rphi_input.data[-1, 0]) + '/'.join(['%+.3fpi' % (rphi_input.data[-1, i]/math.pi) for i in range(1, rphi_input.size(1))])
		rect_str = '/'.join(['%+.4f' % x_input.data[-1, i] for i in range(0, x_input.size(1))])
		time_str = time.strftime('%H:%M:%S', time.gmtime(time_list[-1])) + '(' + time.strftime('%H:%M:%S', time.gmtime(elapse_list[-1])) +')  '
		print(('\n%4d : ' % (x_input.size(0))) + time_str + rect_str + ' & ' + sphr_str + '    =>' + ('%12.6f (%12.6f)' % (output.data[-1].squeeze()[0], torch.min(output.data))))

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
		if func.dim == 0:
			n_eval = int(sys.argv[3]) if len(sys.argv) > 3 else 100
			sphere_BO(n_eval=n_eval, func=func, dim=int(sys.argv[2]))
		else:
			sphere_BO(n_eval=int(sys.argv[2]), func=func)
	else:
		sphere_BO(n_eval=int(sys.argv[2]), path=sys.argv[1])
