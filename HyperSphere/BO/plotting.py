import os
import os.path
import pickle
import sys

import numpy as np
import torch

import matplotlib.pyplot as plt

from HyperSphere.BO.bayesian_optimization_utils import EXPERIMENT_DIR


def folder_name_list(path):
	parent_dir, prefix = os.path.split(path)
	if not os.path.exists(parent_dir):
		parent_dir = os.path.join(EXPERIMENT_DIR, parent_dir)
	search_sub_dir = not np.any([os.path.isfile(os.path.join(parent_dir, elm)) for elm in os.listdir(parent_dir)])
	if search_sub_dir:
		sphere_folder_list = []
		cube_folder_list = []
		for sub_folder in [elm for elm in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, elm))]:
			sub_folder_folder_list = [elm for elm in os.listdir(os.path.join(parent_dir, sub_folder)) if os.path.isdir(os.path.join(parent_dir, sub_folder, elm)) and prefix == elm[:len(prefix)]]
			sphere_folder_list += [os.path.join(parent_dir, sub_folder, elm) for elm in sub_folder_folder_list if 'sphere' in elm]
			cube_folder_list += [os.path.join(parent_dir, sub_folder, elm) for elm in sub_folder_folder_list if 'cube' in elm]
	else:
		folder_list = [elm for elm in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, elm)) and prefix == elm[:len(prefix)]]
		if len(folder_list) == 0:
			print('No experimental result')
			return
		sphere_folder_list = [os.path.join(parent_dir, elm) for elm in folder_list if 'sphere' in elm]
		cube_folder_list = [os.path.join(parent_dir, elm) for elm in folder_list if 'cube' in elm]
	return sphere_folder_list, cube_folder_list


def plot(path):
	sphere_folder_list, cube_folder_list = folder_name_list(path)

	sphere_output_list = []
	for folder in sphere_folder_list:
		f = open(os.path.join(folder, 'data_config.pkl'))
		output = pickle.load(f)['output'].squeeze(1)
		sphere_output_list.append(output)
		f.close()
	sphere_n_eval = np.min([elm.size(0) for elm in sphere_output_list])
	sphere_output_tensor = torch.stack([elm[:sphere_n_eval] for elm in sphere_output_list], 1)
	sphere_best_history = sphere_output_tensor.clone()
	for i in range(1, sphere_best_history.size(0)):
		sphere_best_history[i], _ = torch.min(sphere_output_tensor[:i+1], 0)

	cube_output_list = []
	for folder in cube_folder_list:
		f = open(os.path.join(folder, 'data_config.pkl'))
		output = pickle.load(f)['output'].squeeze(1)
		cube_output_list.append(output)
		f.close()
	cube_n_eval = np.min([elm.size(0) for elm in cube_output_list])
	cube_output_tensor = torch.stack([elm[:cube_n_eval] for elm in cube_output_list], 1)
	cube_best_history = cube_output_tensor.clone()
	for i in range(1, cube_best_history.size(0)):
		cube_best_history[i], _ = torch.min(cube_output_tensor[:i + 1], 0)

	sphere_data = sphere_best_history.data if hasattr(sphere_best_history, 'data') else sphere_best_history
	sphere_data = (sphere_data.cpu() if sphere_data.is_cuda else sphere_data).numpy()
	sphere_mean = np.mean(sphere_data, 1)
	sphere_std = np.std(sphere_data, 1)

	cube_data = cube_best_history.data if hasattr(cube_best_history, 'data') else cube_best_history
	cube_data = (cube_data.cpu() if cube_data.is_cuda else cube_data).numpy()
	cube_mean = np.mean(cube_data, 1)
	cube_std = np.std(cube_data, 1)

	fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True, sharey=True)
	axes[1].plot(np.arange(sphere_n_eval), sphere_mean, 'r', label='sphere(' + str(sphere_data.shape[1]) + ')')
	axes[1].fill_between(np.arange(sphere_n_eval), sphere_mean - sphere_std, sphere_mean + sphere_std, color='r', alpha=0.25)
	axes[1].plot(np.arange(cube_n_eval), cube_mean, 'b', label='cube(' + str(cube_data.shape[1]) + ')')
	axes[1].fill_between(np.arange(cube_n_eval), cube_mean - cube_std, cube_mean + cube_std, color='b', alpha=0.25)
	axes[1].legend()

	for i in range(sphere_data.shape[1]):
		axes[0].plot(np.arange(sphere_n_eval), sphere_data[:, i])
	axes[0].set_title('Spherical')

	for i in range(cube_data.shape[1]):
		axes[2].plot(np.arange(cube_n_eval), cube_data[:, i])
	axes[2].set_title('Rectangular')

	plt.show()


if __name__ == '__main__':
	plot('/home/coh1/Experiments/Hypersphere_ALL/levy_D20')
