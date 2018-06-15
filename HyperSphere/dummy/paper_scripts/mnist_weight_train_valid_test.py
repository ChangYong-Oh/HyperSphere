import pickle
from datetime import datetime

import numpy as np
import torch

from HyperSphere.dummy.plotting.get_data_from_file import get_data_HPOlib, get_data_sphere
from HyperSphere.test_functions.mnist_weight import mnist_weight


def get_data_mnist_weight(ndim):
	func_name = 'mnist_weight'
	data_list = []
	spearmint_dir_name = '/home/coh1/Experiments/spearmint_mnist_weight_train_valid_test/' + func_name + '_' + str(ndim)
	data_list += get_data_HPOlib(spearmint_dir_name, 'spearmint_april2013_mod')
	sphere_dir_name = '/home/coh1/Experiments/Hypersphere_mnist_weight_train_valid_test/'
	data_list += get_data_sphere(sphere_dir_name, ['spherewarpingorigin'], func_name, ndim)
	cube_dir_name = '/home/coh1/Experiments/Cube_mnist_weight_train_valid_test/'
	data_list += get_data_sphere(cube_dir_name, ['cube'], func_name, ndim)

	for elm in data_list:
		elm['optimum_ind'] = np.argmin(elm['y'])

	return data_list


def mnist_weight_SGD_comparison(data_list):
	n_repeat = 5
	trainvalid_test_eval_list = []

	optimum_weight_list = []
	for elm in data_list:
		optimum_ind = elm['optimum_ind']
		optimum = dict()
		optimum['x'] = torch.from_numpy(elm['x'][optimum_ind].astype(np.float32))
		optimum['y'] = elm['y'][optimum_ind]
		optimum['algorithm'] = elm['algorithm']
		optimum_weight_list.append(optimum)

	for elm in optimum_weight_list:
		trainvalid_test_eval_elm = dict()
		trainvalid_test_eval_elm['algorithm'] = elm['algorithm']
		evaluation_result = []
		print('Optimum from %s algorithm is trained on train+validation and evaluated on test' % elm['algorithm'])
		print('    Due to the randomness in optimizer(SGD:ADAM) multiple training and testing is done.')
		for r in range(n_repeat):
			print('    %s %d/%d has been started. Validation score : %f' % (datetime.now().strftime('%H:%M:%S:%f'), r + 1, n_repeat, elm['y']))
			evaluation_result.append(mnist_weight(elm['x'], use_BO=True, use_validation=False).squeeze()[0])
			print(evaluation_result)
		trainvalid_test_eval_elm['trainvalid_test_eval'] = evaluation_result[:]
		trainvalid_test_eval_list.append(trainvalid_test_eval_elm)

	trainvalid_test_eval_elm = dict()
	trainvalid_test_eval_elm['algorithm'] = 'SGD'
	evaluation_result = []
	print('Optimum from %s algorithm is trained on train+validation and evaluated on test' % 'SGD')
	print('    Due to the randomness in optimizer(SGD:ADAM) multiple training and testing is done.')
	for r in range(n_repeat):
		print('    %s %d/%d has been started.' % (datetime.now().strftime('%H:%M:%S:%f'), r + 1, n_repeat))
		# random elm['x'] is fine as long as size is correct when use_BO=False
		evaluation, _ = mnist_weight(elm['x'], use_BO=False, use_validation=False)
		evaluation_result.append(evaluation.squeeze()[0])
		print(evaluation_result)
	trainvalid_test_eval_elm['trainvalid_test_eval'] = evaluation_result[:]
	trainvalid_test_eval_list.append(trainvalid_test_eval_elm)

	for elm in trainvalid_test_eval_list:
		print(elm['algorithm'])
		eval_data = elm['trainvalid_test_eval']
		print('      ' + ('Average:%f(%f)' % (np.mean(eval_data), np.std(eval_data))) + (' '.join(['%+12.6f' % e for e in eval_data])))

	return trainvalid_test_eval_list


def mnist_weight_SGD_only(ndim):
	n_repeat = 5
	trainvalid_test_eval_list = []

	trainvalid_test_eval_elm = dict()
	trainvalid_test_eval_elm['algorithm'] = 'SGD'
	evaluation_result = []
	radius_result = []
	print('Optimum from %s algorithm is trained on train+validation and evaluated on test' % 'SGD')
	print('    Due to the randomness in optimizer(SGD:ADAM) multiple training and testing is done.')
	for r in range(n_repeat):
		print('    %s %d/%d has been started.' % (datetime.now().strftime('%H:%M:%S:%f'), r + 1, n_repeat))
		dummy = torch.FloatTensor(ndim)
		evaluation, radius = mnist_weight(dummy, use_BO=False, use_validation=False)
		evaluation_result.append(evaluation.squeeze()[0])
		radius_result.append(radius)
		print(evaluation_result)
		print(radius_result)
	trainvalid_test_eval_elm['trainvalid_test_eval'] = evaluation_result[:]
	trainvalid_test_eval_elm['radius'] = radius_result[:]
	trainvalid_test_eval_list.append(trainvalid_test_eval_elm)

	for elm in trainvalid_test_eval_list:
		print(elm['algorithm'])
		eval_data = elm['trainvalid_test_eval']
		print('      ' + ('Average:%f(%f)' % (np.mean(eval_data), np.std(eval_data))) + (' '.join(['%+12.6f' % e for e in eval_data])))

	return trainvalid_test_eval_list


def print_result(filename):
	data_file = open(filename)
	data = pickle.load(data_file)
	data_file.close()
	dim = int(filename[-10:-7])

	print('----------%d----------%d----------%d----------' % (dim, dim, dim))
	for elm in data:
		algo_str = '%-19s ' % elm['algorithm']
		eval_data_list = elm['trainvalid_test_eval']
		mean_eval = np.mean(eval_data_list)
		std_eval = np.std(eval_data_list)
		eval_data_str = algo_str + ('%8.6f(%8.6f)' % (mean_eval, std_eval))
		eval_data_str += ' '.join(['%8.6f' % elm for elm in eval_data_list])
		print(eval_data_str)


if __name__ == '__main__':
	# for ndim in [200, 500]:
	# 	mnist_weight_data_list = get_data_mnist_weight(ndim=ndim)
	# 	result = mnist_weight_SGD_comparison(mnist_weight_data_list)
	# 	save_file_name = '/home/coh1/OCY_docs/publication/icml2017/trainvalid_test_eval_' + str(ndim) + 'dim.pkl'
	# 	save_file = open(save_file_name, 'w')
	# 	pickle.dump(result, save_file)
	# 	save_file.close()
	# 	print(result)
	# dim = 500
	# filename = '/home/coh1/OCY_docs/publication/ICML2018/trainvalid_test_eval_' + str(dim) + 'dim.pkl'
	# print_result(filename)

	# mnist_weight_data_list = get_data_mnist_weight(ndim=dim)
	# for elm in mnist_weight_data_list:
	# 	optimum_x = elm['x'][elm['optimum_ind']]
	# 	print(elm['algorithm'], np.sum(optimum_x ** 2) ** 0.5)
		# optimum_y = elm['y'][elm['optimum_ind']]
		# outside_of_cube = np.sum(np.abs(optimum_x) > 1)
		# print('%-20s %8.6f(%d)' % (elm['algorithm'], optimum_y, outside_of_cube))
		# if outside_of_cube > 0:
		# 	print optimum_x[np.abs(optimum_x) > 1]

	for ndim in [100, 200, 500]:
		result = mnist_weight_SGD_only(ndim)
		print(result)
