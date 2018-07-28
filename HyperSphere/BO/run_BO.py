import argparse
import os.path
import pickle
import sys
import time
from datetime import datetime

EXPERIMENT_DIR = '/home/coh1/Experiments/Hypersphere'

import torch
from torch.autograd import Variable
import torch.multiprocessing as multiprocessing

if os.path.realpath(__file__).rsplit('/', 3)[0] not in sys.path:
	sys.path.append(os.path.realpath(__file__).rsplit('/', 3)[0])

from HyperSphere.BO.acquisition.acquisition_maximization import suggest, optimization_candidates, optimization_init_points, deepcopy_inference, N_INIT
from HyperSphere.GP.models.gp_regression import GPRegression
from HyperSphere.test_functions.benchmarks import *

# Kernels
from HyperSphere.GP.kernels.modules.matern52 import Matern52
from HyperSphere.GP.kernels.modules.radialization import RadializationKernel

# Inferences
from HyperSphere.GP.inference.inference import Inference
from HyperSphere.BO.shadow_inference.inference_sphere_satellite import ShadowInference as satellite_ShadowInference
from HyperSphere.BO.shadow_inference.inference_sphere_origin import ShadowInference as origin_ShadowInference
from HyperSphere.BO.shadow_inference.inference_sphere_origin_satellite import ShadowInference as both_ShadowInference

# feature_map
from HyperSphere.feature_map.modules.kumaraswamy import Kumaraswamy

# boundary conditions
from HyperSphere.feature_map.functionals import sphere_bound


def BO(geometry=None, n_eval=200, path=None, func=None, ndim=None, boundary=False, ard=False, origin=False, warping=False, parallel=False):
	assert (path is None) != (func is None)

	if path is None:
		assert (func.dim == 0) != (ndim is None)
		assert geometry is not None
		if ndim is None:
			ndim = func.dim

		exp_conf_str = geometry
		if geometry == 'sphere':
			assert not ard
			exp_conf_str += 'warping' if warping else ''
			radius_input_map = Kumaraswamy(ndim=1, max_input=ndim ** 0.5) if warping else None
			model = GPRegression(kernel=RadializationKernel(max_power=3, search_radius=ndim ** 0.5, radius_input_map=radius_input_map))
			inference_method = None
			if origin and boundary:
				inference_method = both_ShadowInference
				exp_conf_str += 'both'
			elif origin:
				inference_method = origin_ShadowInference
				exp_conf_str += 'origin'
			elif boundary:
				inference_method = satellite_ShadowInference
				exp_conf_str += 'boundary'
			else:
				inference_method = Inference
				exp_conf_str += 'none'
			bnd = sphere_bound(ndim ** 0.5)
		elif geometry == 'cube':
			assert not origin
			exp_conf_str += ('ard' if ard else '') + ('boundary' if boundary else '')
			model = GPRegression(kernel=Matern52(ndim=ndim, ard=ard))
			inference_method = satellite_ShadowInference if boundary else Inference
			bnd = (-1, 1)

		if not os.path.isdir(EXPERIMENT_DIR):
			raise ValueError('In file : ' + os.path.realpath(__file__) + '\nEXPERIMENT_DIR variable is not properly assigned. Please check it.')
		dir_list = [elm for elm in os.listdir(EXPERIMENT_DIR) if os.path.isdir(os.path.join(EXPERIMENT_DIR, elm))]
		folder_name = func.__name__ + '_D' + str(ndim) + '_' + exp_conf_str + '_' + datetime.now().strftime('%Y%m%d-%H:%M:%S:%f')
		os.makedirs(os.path.join(EXPERIMENT_DIR, folder_name))
		logfile_dir = os.path.join(EXPERIMENT_DIR, folder_name, 'log')
		os.makedirs(logfile_dir)
		model_filename = os.path.join(EXPERIMENT_DIR, folder_name, 'model.pt')
		data_config_filename = os.path.join(EXPERIMENT_DIR, folder_name, 'data_config.pkl')

		x_input = Variable(torch.stack([torch.zeros(ndim), torch.FloatTensor(ndim).uniform_(-1, 1)]))
		if func.func_name == 'stochastic_depth_resnet_cifar100':
			special_init_point = torch.FloatTensor([-0.88672996375809265, -0.83845553984377363, -0.80082455589209434, -0.76868080609344613, -0.74002860499719103, -0.71384507914214379, -0.6895229479156415, -0.66666666534211871, -0.64500158781765049, -0.62432778870160499, -0.60449429448743319, -0.58538383736427368, -0.56690311453886821, -0.54897644926147593, -0.53154137077618735, -0.51454570980003023, -0.49794520561122835, -0.4817019618876005, -0.46578329447738975, -0.45016063464220946, -0.43480887900991927, -0.41970588594137237, -0.40483184457290511, -0.39016909932337462, -0.37570168000845294, -0.36141512736958714, -0.34729635533386094, -0.33333334161175654, -0.31951507564952675, -0.30583136944490208, -0.29227292909996905, -0.27883100126437665, -0.26549747264739709, -0.25226475894331168, -0.23912574658399377, -0.22607369983030123, -0.2131023835975443, -0.20020577167418563, -0.18737817967669568, -0.1746141913340078, -0.16190858934371632, -0.14925649319813961, -0.13665309066289877, -0.12409378040195429, -0.11157411163518405, -0.099089726169870107, -0.086636502479268351, -0.074210299199806373, -0.061807101474520065, -0.049422967019945307, -0.037054013082912562, -0.024696364163967699, -0.012346298973719083, 0])
			x_input = torch.cat([x_input, Variable(special_init_point)])
		n_init_eval = x_input.size(0)
		output = Variable(torch.zeros(n_init_eval, 1))
		for i in range(n_init_eval):
			output[i] = func(x_input[i])

		time_list = [time.time()] * n_init_eval
		elapse_list = [0] * n_init_eval
		pred_mean_list = [0] * n_init_eval
		pred_std_list = [0] * n_init_eval
		pred_var_list = [0] * n_init_eval
		pred_stdmax_list = [1] * n_init_eval
		pred_varmax_list = [1] * n_init_eval
		reference_list = [output.data.squeeze()[0]] * n_init_eval
		refind_list = [1] * n_init_eval
		dist_to_ref_list = [0] * n_init_eval
		sample_info_list = [(10, 0, 10)] * n_init_eval

		inference = inference_method((x_input, output), model)
		inference.init_parameters()
		inference.sampling(n_sample=1, n_burnin=99, n_thin=1)
	else:
		if not os.path.exists(path):
			path = os.path.join(EXPERIMENT_DIR, path)
		logfile_dir = os.path.join(path, 'log')
		model_filename = os.path.join(path, 'model.pt')
		data_config_filename = os.path.join(path, 'data_config.pkl')

		model = torch.load(model_filename)
		data_config_file = open(data_config_filename, 'r')
		for key, value in pickle.load(data_config_file).iteritems():
			if key != 'logfile_dir':
				exec(key + '=value')
		data_config_file.close()

	ignored_variable_names = ['n_eval', 'path', 'i', 'key', 'value', 'logfile_dir', 'n_init_eval',
	                          'data_config_file', 'dir_list', 'folder_name', 'model_filename', 'data_config_filename',
	                          'kernel', 'model', 'inference', 'parallel', 'pool']
	stored_variable_names = set(locals().keys()).difference(set(ignored_variable_names))

	if path is None:
		torch.save(model, model_filename)
		stored_variable = dict()
		for key in stored_variable_names:
			stored_variable[key] = locals()[key]
		f = open(data_config_filename, 'w')
		pickle.dump(stored_variable, f)
		f.close()

	print('Experiment based on data in %s' % os.path.split(model_filename)[0])

	# multiprocessing conflicts with pytorch linear algebra operation
	pool = multiprocessing.Pool(N_INIT) if parallel else None

	for _ in range(n_eval):
		start_time = time.time()
		logfile = open(os.path.join(logfile_dir, str(x_input.size(0) + 1).zfill(4) + '.out'), 'w')
		inference = inference_method((x_input, output), model)

		reference, ref_ind = torch.min(output, 0)
		reference = reference.data.squeeze()[0]
		gp_hyper_params = inference.sampling(n_sample=10, n_burnin=0, n_thin=1)
		inferences = deepcopy_inference(inference, gp_hyper_params)

		x0_cand = optimization_candidates(x_input, output, -1, 1)
		x0, sample_info = optimization_init_points(x0_cand, reference=reference, inferences=inferences)
		next_x_point, pred_mean, pred_std, pred_var, pred_stdmax, pred_varmax = suggest(x0=x0, reference=reference, inferences=inferences, bounds=bnd, pool=pool)

		time_list.append(time.time())
		elapse_list.append(time_list[-1] - time_list[-2])
		pred_mean_list.append(pred_mean.squeeze()[0])
		pred_std_list.append(pred_std.squeeze()[0])
		pred_var_list.append(pred_var.squeeze()[0])
		pred_stdmax_list.append(pred_stdmax.squeeze()[0])
		pred_varmax_list.append(pred_varmax.squeeze()[0])
		reference_list.append(reference)
		refind_list.append(ref_ind.data.squeeze()[0] + 1)
		dist_to_ref_list.append(torch.sum((next_x_point - x_input[ref_ind]).data ** 2) ** 0.5)
		sample_info_list.append(sample_info)

		x_input = torch.cat([x_input, next_x_point], 0)
		output = torch.cat([output, func(x_input[-1]).resize(1, 1)])

		min_ind = torch.min(output, 0)[1]
		min_loc = x_input[min_ind]
		min_val = output[min_ind]
		dist_to_suggest = torch.sum((x_input - x_input[-1]).data ** 2, 1) ** 0.5
		dist_to_min = torch.sum((x_input - min_loc).data ** 2, 1) ** 0.5
		out_of_box = torch.sum((torch.abs(x_input.data) > 1), 1)
		print('')
		for i in range(x_input.size(0)):
			time_str = time.strftime('%H:%M:%S', time.gmtime(time_list[i])) + '(' + time.strftime('%H:%M:%S', time.gmtime(elapse_list[i])) + ')  '
			data_str = ('%3d-th : %+12.4f(R:%8.4f[%4d]/ref:[%3d]%8.4f), sample([%2d] best:%2d/worst:%2d), '
			            'mean : %+.4E, std : %.4E(%5.4f), var : %.4E(%5.4f), '
			            '2ownMIN : %8.4f, 2curMIN : %8.4f, 2new : %8.4f' %
			            (i + 1, output.data.squeeze()[i], torch.sum(x_input.data[i] ** 2) ** 0.5, out_of_box[i], refind_list[i], reference_list[i],
			             sample_info_list[i][2], sample_info_list[i][0], sample_info_list[i][1],
			             pred_mean_list[i], pred_std_list[i], pred_std_list[i] / pred_stdmax_list[i], pred_var_list[i], pred_var_list[i] / pred_varmax_list[i],
			             dist_to_ref_list[i], dist_to_min[i], dist_to_suggest[i]))
			min_str = '  <========= MIN' if i == min_ind.data.squeeze()[0] else ''
			print(time_str + data_str + min_str)
			logfile.writelines(time_str + data_str + min_str + '\n')

		logfile.close()

		torch.save(model, model_filename)
		stored_variable = dict()
		for key in stored_variable_names:
			stored_variable[key] = locals()[key]
		f = open(data_config_filename, 'w')
		pickle.dump(stored_variable, f)
		f.close()

	if parallel:
		pool.close()

	print('Experiment based on data in %s' % os.path.split(model_filename)[0])

	return os.path.split(model_filename)[0]

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Bayesian Optimization runner')
	parser.add_argument('-g', '--geometry', dest='geometry', help='cube/sphere')
	parser.add_argument('-e', '--n_eval', dest='n_eval', type=int, default=1)
	parser.add_argument('-p', '--path', dest='path')
	parser.add_argument('-d', '--dim', dest='ndim', type=int)
	parser.add_argument('-f', '--func', dest='func_name')
	parser.add_argument('--boundary', dest='boundary', action='store_true', default=False)
	parser.add_argument('--origin', dest='origin', action='store_true', default=False)
	parser.add_argument('--ard', dest='ard', action='store_true', default=False)
	parser.add_argument('--warping', dest='warping', action='store_true', default=False)
	parser.add_argument('--parallel', dest='parallel', action='store_true', default=False)

	args = parser.parse_args()
	# if args.n_eval == 0:
	# 	args.n_eval = 3 if args.path is None else 1
	assert (args.path is None) != (args.func_name is None)
	args_dict = vars(args)
	if args.func_name is not None:
		exec 'func=' + args.func_name
		args_dict['func'] = func
	del args_dict['func_name']
	if args.path is None:
		assert (func.dim == 0) != (args.ndim is None)
		assert args.geometry is not None
		if args.geometry == 'sphere':
			assert not args.ard
		elif args.geometry == 'cube':
			assert not args.origin
			assert not args.warping

	print(BO(**vars(args)))
