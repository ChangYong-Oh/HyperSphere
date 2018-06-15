import os.path
import pickle
import sys
import time
from datetime import datetime

from HyperSphere.GP.kernels.modules.radialization_warping import RadializationWarpingKernel

# ShadowInference version should coincide with the one used in acquisition_maximization
from HyperSphere.BO.acquisition.acquisition_maximization import suggest, optimization_candidates, \
	optimization_init_points, deepcopy_inference
from HyperSphere.BO.shadow_inference.inference_sphere_origin_satellite import ShadowInference
from HyperSphere.GP.models.gp_regression import GPRegression
from HyperSphere.feature_map.functionals import radial_bound
from HyperSphere.dummy.plotting import EXPERIMENT_DIR
from HyperSphere.test_functions.benchmarks import *

exp_str = __file__.split('/')[-1].split('_')[0]


def BO(n_eval=200, path=None, func=None, ndim=None):
	assert (path is None) != (func is None)
	if path is not None:
		if not os.path.exists(path):
			path = os.path.join(EXPERIMENT_DIR, path)
		model_filename = os.path.join(path, 'model.pt')
		data_config_filename = os.path.join(path, 'data_config.pkl')

		model = torch.load(model_filename)
		data_config_file = open(data_config_filename, 'r')
		for key, value in pickle.load(data_config_file).iteritems():
			exec(key + '=value')
		data_config_file.close()

		inference = ShadowInference((x_input, output), model)
	else:
		assert (func.dim == 0) != (ndim is None)
		if ndim is None:
			ndim = func.dim
		dir_list = [elm for elm in os.listdir(EXPERIMENT_DIR) if os.path.isdir(os.path.join(EXPERIMENT_DIR, elm))]
		folder_name = func.__name__ + '_D' + str(ndim) + '_' + exp_str + '_' + datetime.now().strftime('%Y%m%d-%H:%M:%S:%f')
		os.makedirs(os.path.join(EXPERIMENT_DIR, folder_name))
		model_filename = os.path.join(EXPERIMENT_DIR, folder_name, 'model.pt')
		data_config_filename = os.path.join(EXPERIMENT_DIR, folder_name, 'data_config.pkl')

		search_sphere_radius = ndim ** 0.5

		x_input = Variable(torch.ger(-torch.arange(0, 2), torch.ones(ndim)))
		output = Variable(torch.zeros(x_input.size(0), 1))
		for i in range(x_input.size(0)):
			output[i] = func(x_input[i])

		model = GPRegression(kernel=RadializationWarpingKernel(max_power=3, search_radius=search_sphere_radius))

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

		inference = ShadowInference((x_input, output), model)
		inference.init_parameters()
		inference.sampling(n_sample=1, n_burnin=99, n_thin=1)

	stored_variable_names = locals().keys()
	ignored_variable_names = ['n_eval', 'path', 'data_config_file', 'dir_list', 'folder_name',
	                          'next_ind', 'model_filename', 'data_config_filename', 'i',
	                          'kernel_input_map', 'model', 'inference']
	stored_variable_names = set(stored_variable_names).difference(set(ignored_variable_names))

	bnd = radial_bound(search_sphere_radius)

	for _ in range(3):
		print('Experiment based on data in ' + os.path.split(model_filename)[0])

	for _ in range(n_eval):
		inference = ShadowInference((x_input, output), model)

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
		dist_to_ref_list.append(torch.sum((next_x_point - x_input[ref_ind].data) ** 2) ** 0.5)
		sample_info_list.append(sample_info)

		x_input = torch.cat([x_input, Variable(next_x_point)], 0)
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

	for _ in range(3):
		print('Experiment based on data in ' + os.path.split(model_filename)[0])

	return os.path.split(model_filename)[0]


if __name__ == '__main__':
	run_new = False
	path, suffix = os.path.split(sys.argv[1])
	if path == '' and not ('_D' in suffix):
		run_new = True
	if run_new:
		func = locals()[sys.argv[1]]
		if func.dim == 0:
			n_eval = int(sys.argv[3]) if len(sys.argv) > 3 else 100
			BO(n_eval=n_eval, func=func, dim=int(sys.argv[2]))
		else:
			BO(n_eval=int(sys.argv[2]), func=func)
	else:
		BO(n_eval=int(sys.argv[2]), path=sys.argv[1])
