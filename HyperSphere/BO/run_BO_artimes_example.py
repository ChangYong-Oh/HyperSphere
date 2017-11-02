import os.path
import pickle
import sys
import time
from datetime import datetime

from artemis.experiments import experiment_function
from artemis.experiments.decorators import experiment_root

from HyperSphere.BO.acquisition.acquisition_maximization import suggest, optimization_candidates, optimization_init_points, deepcopy_inference
from HyperSphere.BO.utils.datafile_utils import EXPERIMENT_DIR
from HyperSphere.GP.models.gp_regression import GPRegression
from HyperSphere.test_functions.benchmarks import *

# Kernels
from HyperSphere.GP.kernels.modules.matern52 import Matern52
from HyperSphere.GP.kernels.modules.radialization_warping import RadializationWarpingKernel
from HyperSphere.GP.kernels.modules.radialization import RadializationKernel

# Inferences
from HyperSphere.GP.inference.inference import Inference
from HyperSphere.BO.shadow_inference.inference_sphere_satellite import ShadowInference as satellite_ShadowInference
from HyperSphere.BO.shadow_inference.inference_sphere_origin import ShadowInference as origin_ShadowInference
from HyperSphere.BO.shadow_inference.inference_sphere_origin_satellite import ShadowInference as both_ShadowInference

# boundary conditions
from HyperSphere.feature_map.functionals import radial_bound

@experiment_root
def BO(geometry=None, n_eval=200, path=None, func=None, ndim=None, boundary=False, ard=None, origin=None, warping=None, seed=1234):
	assert (path is None) != (func is None)

	if isinstance(func, str):
		func = {'levy': levy}[func]

	if path is None:
		assert (func.dim == 0) != (ndim is None)
		assert geometry is not None
		if ndim is None:
			ndim = func.dim

		exp_conf_str = geometry
		if geometry == 'sphere':
			assert ard is None and origin is not None
			search_sphere_radius = ndim ** 0.5

			exp_conf_str += 'warping' if warping else ''
			kernel_method = RadializationWarpingKernel if warping else RadializationKernel
			model = GPRegression(kernel=kernel_method(max_power=3, search_radius=search_sphere_radius))
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
			bnd = radial_bound(search_sphere_radius)
		elif geometry == 'cube':
			assert origin is None and ard is not None
			exp_conf_str += ('ard' if ard else '') + ('boundary' if boundary else '')
			kernel_method = Matern52
			model = GPRegression(kernel=kernel_method(ndim=ndim, ard=ard))
			inference_method = satellite_ShadowInference if boundary else Inference
			bnd = (-1, 1)

		dir_list = [elm for elm in os.listdir(EXPERIMENT_DIR) if os.path.isdir(os.path.join(EXPERIMENT_DIR, elm))]
		folder_name = func.__name__ + '_D' + str(ndim) + '_' + exp_conf_str + '_' + datetime.now().strftime('%Y%m%d-%H:%M:%S:%f')
		os.makedirs(os.path.join(EXPERIMENT_DIR, folder_name))
		os.makedirs(os.path.join(EXPERIMENT_DIR, folder_name, 'log'))
		model_filename = os.path.join(EXPERIMENT_DIR, folder_name, 'model.pt')
		data_config_filename = os.path.join(EXPERIMENT_DIR, folder_name, 'data_config.pkl')

		x_input = Variable(torch.ger(-torch.arange(0, 2), torch.ones(ndim)))
		output = Variable(torch.zeros(x_input.size(0), 1))
		for i in range(x_input.size(0)):
			output[i] = func(x_input[i])

		time_list = [time.time()] * 2
		elapse_list = [0, 0]
		pred_mean_list = [0, 0]
		pred_std_list = [0, 0]
		pred_var_list = [0, 0]
		pred_stdmax_list = [1, 1]
		pred_varmax_list = [1, 1]
		reference_list = [output.data.squeeze()[0]] * 2
		refind_list = [1, 1]
		dist_to_ref_list = [0, 0]
		sample_info_list = [(10, 0, 10)] * 2

		inference = inference_method((x_input, output), model)
		inference.init_parameters()
		inference.sampling(n_sample=1, n_burnin=99, n_thin=1)
	else:
		if not os.path.exists(path):
			path = os.path.join(EXPERIMENT_DIR, path)
		model_filename = os.path.join(path, 'model.pt')
		data_config_filename = os.path.join(path, 'data_config.pkl')

		model = torch.load(model_filename)
		data_config_file = open(data_config_filename, 'r')
		for key, value in pickle.load(data_config_file).iteritems():
			exec(key + '=value')
		data_config_file.close()

	ignored_variable_names = ['n_eval', 'path', 'ndim', 'i',
	                          'data_config_file', 'dir_list', 'folder_name', 'model_filename', 'data_config_filename',
	                          'kernel_method', 'inference_method', 'bnd', 'model', 'inference']
	stored_variable_names = set(locals().keys()).difference(set(ignored_variable_names))

	print(('Experiment based on data in %s\n' % os.path.split(model_filename)[0]) * 3)

	for _ in range(n_eval):
		inference = inference_method((x_input, output), model)

		reference, ref_ind = torch.min(output, 0)
		reference = reference.data.squeeze()[0]
		gp_hyper_params = inference.sampling(n_sample=10, n_burnin=0, n_thin=1)
		inferences = deepcopy_inference(inference, gp_hyper_params)

		x0_cand = optimization_candidates(x_input, output, -1, 1)
		x0, sample_info = optimization_init_points(x0_cand, inferences, reference=reference)
		next_x_point, pred_mean, pred_std, pred_var, pred_stdmax, pred_varmax = suggest(inferences, x0=x0, bounds=bnd, reference=reference)

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
		output = torch.cat([output, func(x_input[-1])])

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
		print(model.kernel.__class__.__name__)

		torch.save(model, model_filename)
		stored_variable = dict()
		for key in stored_variable_names:
			stored_variable[key] = locals()[key]
		f = open(data_config_filename, 'w')
		pickle.dump(stored_variable, f)
		f.close()

	print(('Experiment based on data in %s\n' % os.path.split(model_filename)[0]) * 3)

	return os.path.split(model_filename)[0]



if __name__ == '__main__':
	run_new = False
	path, suffix = os.path.split(sys.argv[1])

	baseline = BO.add_root_variant('baseline', geometry='cube', n_eval=int(sys.argv[2]), path=None, func='levy', ndim=int(sys.argv[1]), ard=None,
	                               origin=None, warping=True)

	for seed in range(10):
		baseline.add_variant(boundary=False, seed=seed)
	for seed in range(10):
		baseline.add_variant(boundary=True, seed=seed)

	BO.browse(display_format='flat')


	# if path == '' and not ('_D' in suffix):
	# 	run_new = True
	# if run_new:
	# 	func = locals()[sys.argv[1]]
	# 	if func.dim == 0:
	# 		n_eval = int(sys.argv[3]) if len(sys.argv) > 3 else 100
	# 		BO(n_eval=n_eval, func=func, ndim=int(sys.argv[2]))
	# 	else:
	# 		BO(n_eval=int(sys.argv[2]), func=func)
	# else:
	# 	BO(n_eval=int(sys.argv[2]), path=sys.argv[1])

