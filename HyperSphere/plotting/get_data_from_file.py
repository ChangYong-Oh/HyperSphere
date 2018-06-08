import os
import sys
import pickle
import numpy as np
from scipy.io import loadmat
import pandas as pd


def get_data_sphere(dir_name, algorithms, func_name, ndim):
	if isinstance(algorithms, str):
		algorithms = [algorithms]

	data_list = []
	for algorithm in algorithms:
		result_dir_name_list = [os.path.join(dir_name, elm) for elm in os.listdir(dir_name) if '_'.join(elm.split('_')[:-1]) == func_name + '_D' + str(ndim) + '_' + algorithm]
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


def get_data_additive(dir_name, func_name, ndim, simpler=False):
	exp_id = func_name + '_D' + str(ndim)
	file_pool = [elm for elm in os.listdir(dir_name) if elm[:len(exp_id)] == exp_id]
	additive_ids = np.unique(['_'.join(elm.split('_')[2:4]) for elm in file_pool])
	data_list = []
	for additive_id in additive_ids:
		data_id = '_'.join([exp_id, additive_id])
		data_x_list = [elm for elm in os.listdir(dir_name) if data_id == elm[:len(data_id)] and elm[-6:] == '_x.mat']
		instance_ids = [elm.split('_')[4] for elm in data_x_list]
		for instance_id in instance_ids:
			data_batch_x = loadmat(os.path.join(dir_name, '_'.join([data_id, instance_id, 'x.mat'])))['queries']
			data_batch_y = loadmat(os.path.join(dir_name, '_'.join([data_id, instance_id, 'y.mat'])))['neg_values']
			data_batch_optimum = loadmat(os.path.join(dir_name, '_'.join([data_id, instance_id, 'optimum.mat'])))['neg_optima']
			for s in range(data_batch_x.shape[0]):
				data = {}
				if simpler:
					data['algorithm'] = 'additive'
				else:
					data['algorithm'] = 'additiveBO_' + additive_id[5:]
				data['x'] = data_batch_x[s]
				data['y'] = data_batch_y[s]
				data['optimum'] = data_batch_optimum[s]
				data['n_eval'] = data['y'].size
				data_list.append(data)
	return data_list


def get_data_warping(dir_name, func_name, ndim):
	data_list = []
	folder_list = [os.path.join(dir_name, elm) for elm in os.listdir(dir_name) if func_name + '_' + str(ndim) == elm[:len(func_name + str(ndim)) + 1]]
	for folder in folder_list:
		df = pd.read_pickle(os.path.join(folder, 'query_eval_data.pkl'))
		df.sort_index(inplace=True)
		df = df.applymap(float)
		n_eval = df.shape[0]
		data = {}
		data['algorithm'] = 'spearmint_warping'
		x_data = np.empty((n_eval, 0))
		for i in range(1, ndim + 1):
			x_data = np.hstack((x_data, np.array(df['x' + str(i)]).reshape((n_eval, 1))))
		data['x'] = x_data
		data['y'] = np.array(df['value'])
		data['optimum'] = np.array([np.min(data['y'][:i]) for i in range(1, n_eval + 1)])
		data['n_eval'] = n_eval
		data_list.append(data)
	return data_list


def get_data_elastic(dir_name, func_name, ndim):
	data_list = []
	exp_id = func_name + str(ndim)
	filename_list = [os.path.join(dir_name, elm) for elm in os.listdir(dir_name) if exp_id == elm[:len(exp_id)] and '.mat' in elm]
	for filename in filename_list:
		mat_data = loadmat(filename)
		data = {}
		data['algorithm'] = 'elastic'
		data['x'] = mat_data['x']
		data['y'] = -mat_data['y'].flatten()
		data['optimum'] = -mat_data['ybest'].flatten()
		data['n_eval'] = mat_data['y'].size
		data_list.append(data)
	return data_list


def get_data(func_name, ndim, suffix='_center-random', P_setting='_P=9'):
	# suffix = '_center-corner'
	spearmint_dir_name = '/home/coh1/Experiments/spearmint_mnist_weight' + suffix + '/' + func_name + '_' + str(ndim)
	smac_dir_name = '/home/coh1/Experiments/smac_ALL' + suffix + '/' + func_name + '_' + str(ndim)
	tpe_dir_name = '/home/coh1/Experiments/tpe_ALL' + suffix + '/' + func_name + '_' + str(ndim)
	additive_dir_name = '/home/coh1/Experiments/Additive_BO_mat_ALL' + suffix + '/'
	warping_dir_name = '/home/coh1/Experiments/Warping_ALL' + suffix + '123/'
	elastic_dir_name = '/home/coh1/Experiments/elastic_BO_mat' + suffix + '/'
	sphere_dir_name = '/home/coh1/Experiments/Hypersphere_mnist_weight' + suffix + P_setting + '/'
	cube_dir_name = '/home/coh1/Experiments/Cube_mnist_weight' + suffix + '/'
	data_list = []
	try:
		data_list += get_data_HPOlib(spearmint_dir_name, 'spearmint_april2013_mod')
	except OSError:
		pass
	try:
		data_list += get_data_HPOlib(tpe_dir_name, 'hyperopt_august2013_mod')
	except OSError:
		pass
	try:
		data_list += get_data_HPOlib(smac_dir_name, 'smac_2_10_00-dev')
	except OSError:
		pass
	try:
		data_list += get_data_additive(additive_dir_name, func_name, ndim)
	except OSError:
		pass
	try:
		data_list += get_data_warping(warping_dir_name, func_name, ndim)
	except OSError:
		pass
	try:
		data_list += get_data_elastic(elastic_dir_name, func_name, ndim)
	except OSError:
		pass
	data_list += get_data_sphere(cube_dir_name, ['cube', 'cubeard'], func_name, ndim)
	data_list += get_data_sphere(sphere_dir_name, ['sphereboth', 'sphereorigin', 'spherewarpingboth', 'spherewarpingorigin'], func_name, ndim)
	return data_list


if __name__ == '__main__':
	get_data_sphere(*sys.argv[1:])