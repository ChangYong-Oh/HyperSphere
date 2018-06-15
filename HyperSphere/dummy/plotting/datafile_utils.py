import os
import os.path
import pickle
import sys

import numpy as np
from torch.autograd import Variable


file_path = os.path.realpath(__file__)
if 'anaconda' in file_path.lower():
	EXPERIMENT_DIR = os.path.join('/'.join(file_path.split('/')[:-7]), 'Experiments/Hypersphere')
else:
	EXPERIMENT_DIR = os.path.join('/'.join(file_path.split('/')[:-6]), 'Experiments/Hypersphere')


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


def folder_name_list(path):
	parent_dir, prefix = os.path.split(path)
	if not os.path.exists(parent_dir):
		parent_dir = os.path.join(EXPERIMENT_DIR, parent_dir)
	search_sub_dir = not np.any([os.path.isfile(os.path.join(parent_dir, elm)) for elm in os.listdir(parent_dir)])
	if search_sub_dir:
		experiment_type = set()
		sub_dir_list = [elm for elm in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, elm))]
		for sub_folder in sub_dir_list:
			sub_folder_folder_list = [elm for elm in os.listdir(os.path.join(parent_dir, sub_folder)) if os.path.isdir(os.path.join(parent_dir, sub_folder, elm)) and prefix == elm[:len(prefix)]]
			experiment_type = experiment_type.union([remove_entailing_number(elm.split('_')[-1]) for elm in sub_folder_folder_list])
		result_dict = dict()
		for exp_type in experiment_type:
			result_dict[exp_type] = []
		for sub_folder in sub_dir_list:
			sub_folder_folder_list = [elm for elm in os.listdir(os.path.join(parent_dir, sub_folder)) if os.path.isdir(os.path.join(parent_dir, sub_folder, elm)) and prefix == elm[:len(prefix)]]
			for exp_type in experiment_type:
				result_dict[exp_type] += [os.path.join(parent_dir, sub_folder, elm) for elm in sub_folder_folder_list if exp_type == elm.split('_')[-1][:len(exp_type)]]
	else:
		folder_list = [elm for elm in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, elm)) and prefix == elm[:len(prefix)]]
		experiment_type = set([remove_entailing_number(elm.split('_')[-1]) for elm in folder_list])
		if len(folder_list) == 0:
			print('No experimental result')
			return
		result_dict = dict()
		for exp_type in experiment_type:
			result_dict[exp_type] = []
		for exp_type in experiment_type:
			result_dict[exp_type] += [os.path.join(parent_dir, elm) for elm in folder_list if exp_type in elm]
	return result_dict


def remove_entailing_number(str):
	result = str[:]
	for i in range(len(result)-1, -1, -1):
		if not result[i].isdigit():
			result = result[:i+1]
			break
	return result


def how_many_evaluations(path):
	folder_category = folder_name_list(path)
	grassmanian_folder_list = folder_category['grassmanian']
	sphere_folder_list = folder_category['sphere']
	cube_folder_list = folder_category['cube']
	for folder in grassmanian_folder_list + sphere_folder_list + cube_folder_list:
		f = open(os.path.join(folder, 'data_config.pkl'))
		n_eval = pickle.load(f)['output'].numel()
		f.close()
		print('%4d evaluation in Experiment %s' % (n_eval, folder))


if __name__ == '__main__':
	print('  1: The number of evaluations')
	print('  2: The nan value check(remove last evaluation)')
	sys.stdout.write('Choose command : ')
	command = int(sys.stdin.readline()[:-1])
	if command == 1:
		path_info = '/'.join(os.path.abspath(__file__).split('/')[:3] + ['Experiments'])
		sys.stdout.write('path(%s) : ' % path_info)
		path = sys.stdin.readline()[:-1]
		if not os.path.exists(os.path.split(path)[0]):
			path = os.path.join(path_info, path)
		how_many_evaluations(path)
	elif command == 2:
		path_info = '/'.join(os.path.abspath(__file__).split('/')[:3] + ['Experiments'])
		sys.stdout.write('path(%s) : ' % path_info)
		path = sys.stdin.readline()[:-1]
		if not os.path.exists(os.path.split(path)[0]):
			path = os.path.join(path_info, path)
		remove_last_evaluation(path)

