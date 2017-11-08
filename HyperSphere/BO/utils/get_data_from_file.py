import os
import sys
import pickle
import numpy as np
from scipy.io import loadmat


def get_data_sphere(dir_name, algorithms, func_name, ndim):
	if isinstance(algorithms, str):
		algorithms = [algorithms]

	data_list = []
	for algorithm in algorithms:
		result_dir_name_list = [os.path.join(dir_name, elm) for elm in os.listdir(dir_name) if '_'.join(elm.split('_')[:3]) == func_name + '_D' + str(ndim) + '_' + algorithm]
		for result_dir_name in result_dir_name_list:
			result_file = open(os.path.join(result_dir_name, 'data_config.pkl'))
			unpickled_data = pickle.load(result_file)
			data = dict()
			data['algorithm'] = algorithm
			data['n_eval'] = unpickled_data['output'].numel()
			data['x'] = unpickled_data['x_input'].data.numpy()
			data['y'] = unpickled_data['output'].data.numpy()
			data['optimum'] = np.array([np.min(data['y'][:i]) for i in range(1, data['n_eval'] + 1)])
			data_list.append(data)

	return data_list


def get_data_HPOlib(dir_name, optimizer_name='spearmint_april2013_mod'):
	folder_name = [os.path.join(dir_name, elm) for elm in os.listdir(dir_name) if os.path.isdir(os.path.join(dir_name, elm)) and elm != 'Plots']
	file_name_list = [os.path.join(elm, optimizer_name + '.pkl') for elm in folder_name if os.path.exists(os.path.join(elm, optimizer_name + '.pkl'))]

	n_data = len(file_name_list)
	data_list = []
	for i in range(n_data):
		data_file = open(file_name_list[i])
		unpickled_data = pickle.load(data_file)
		data = {}
		data['algorithm'] = optimizer_name.split('_')[0]
		data['n_eval'] = len(unpickled_data['trials'])
		data['x'] = np.array([HPOlib_params2np(unpickled_data['trials'][i]['params']) for i in range(data['n_eval'])])
		data['y'] = np.array([unpickled_data['trials'][i]['result'] for i in range(data['n_eval'])])
		data['optimum'] = np.array([np.min(data['y'][:i]) for i in range(1, data['n_eval'] + 1)])
		data_list.append(data)
	return data_list


def HPOlib_params2np(params):
	x = np.array([[int(key[1:]), float(value)] for key, value in params.iteritems()])
	sort_ind = np.argsort(x[:, 0])
	return x[sort_ind, 1]


def get_data_additive(dir_name, func_name, ndim):
	data_x = loadmat(os.path.join(dir_name, func_name + '_D' + str(ndim) + '_x.mat'))['queries']
	data_y = loadmat(os.path.join(dir_name, func_name + '_D' + str(ndim) + '_y.mat'))['neg_values']
	data_optimum = loadmat(os.path.join(dir_name, func_name + '_D' + str(ndim) + '_optimum.mat'))['neg_optima']

	n_data = data_x.shape[0]
	data_list = []
	for i in range(n_data):
		data = {}
		data['algorithm'] = 'additiveBO'
		data['x'] = np.squeeze(data_x[i])
		data['y'] = np.squeeze(data_y[i])
		data['optimum'] = np.squeeze(data_optimum[i])
		data['n_eval'] = data['y'].size
		data_list.append(data)

	return data_list


def get_data(func_name, ndim):
	HPOlib_dir_name = '/home/coh1/git_repositories/HPOlib/HPOlib/benchmarks/' + func_name + str(ndim)
	additive_dir_name = '/home/coh1/Experiments/Additive_BO_mat'
	sphere_dir_name = '/home/coh1/Experiments/Hypersphere_ALL'
	data_list = get_data_additive(additive_dir_name, func_name, ndim)
	if os.path.exists(HPOlib_dir_name):
		data_list += get_data_HPOlib(HPOlib_dir_name)
	data_list += get_data_sphere(sphere_dir_name, ['spherewarpingboth', 'spherewarpingorigin'], func_name, ndim)
	return data_list


if __name__ == '__main__':
	get_data_sphere(*sys.argv[1:])