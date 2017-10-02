import os
import os.path
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch

from HyperSphere.BO.utils.datafile_utils import folder_name_list

np.random.seed(123)
color_matrix = np.random.uniform(0, 1, [10, 3])
np.random.seed()


def optimum_plot(path):
	title = os.path.split(path)[1]
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
	sphere_best_history_list = []
	for elm in sphere_output_list:
		sphere_best_history_list.append(np.array([torch.min(elm.data[:d]) for d in range(1, elm.numel()+1)]))

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
	cube_best_history_list = []
	for elm in cube_output_list:
		cube_best_history_list.append(np.array([torch.min(elm.data[:d]) for d in range(1, elm.numel()+1)]))

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
	axes[1].set_title('Comparison', fontsize=10)
	axes[1].legend()

	for i, best_history in enumerate(sphere_best_history_list):
		axes[0].plot(np.arange(best_history.size), best_history, color=color_matrix[i])
	axes[0].set_title('Spherical', fontsize=10)

	for i, best_history in enumerate(cube_best_history_list):
		axes[2].plot(np.arange(best_history.size), best_history, color=color_matrix[i])
	axes[2].set_title('Rectangular', fontsize=10)

	plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=2.0)
	plt.suptitle(title)
	plt.show()


def radius_plot(path):
	title = os.path.split(path)[1]
	sphere_folder_list, cube_folder_list = folder_name_list(path)

	sphere_radius_list = []
	for folder in sphere_folder_list:
		f = open(os.path.join(folder, 'data_config.pkl'))
		input = pickle.load(f)['x_input'].squeeze(1)
		sphere_radius_list.append(torch.sqrt(torch.sum(input ** 2, 1)))
		f.close()
	sphere_n_eval = np.min([elm.size(0) for elm in sphere_radius_list])
	sphere_radius_tensor = torch.stack([elm[:sphere_n_eval] for elm in sphere_radius_list], 1)
	sphere_radius_list = [(elm.data if hasattr(elm, 'data') else elm).numpy() for elm in sphere_radius_list]

	cube_radius_list = []
	for folder in cube_folder_list:
		f = open(os.path.join(folder, 'data_config.pkl'))
		input = pickle.load(f)['x_input'].squeeze(1)
		cube_radius_list.append(torch.sqrt(torch.sum(input ** 2, 1)))
		f.close()
	cube_n_eval = np.min([elm.size(0) for elm in cube_radius_list])
	cube_radius_tensor = torch.stack([elm[:cube_n_eval] for elm in cube_radius_list], 1)
	cube_radius_list = [(elm.data if hasattr(elm, 'data') else elm).numpy() for elm in cube_radius_list]

	sphere_data = sphere_radius_tensor.data if hasattr(sphere_radius_tensor, 'data') else sphere_radius_tensor
	sphere_data = (sphere_data.cpu() if sphere_data.is_cuda else sphere_data).numpy()
	sphere_mean = np.mean(sphere_data, 1)
	sphere_std = np.std(sphere_data, 1)

	cube_data = cube_radius_tensor.data if hasattr(cube_radius_tensor, 'data') else cube_radius_tensor
	cube_data = (cube_data.cpu() if cube_data.is_cuda else cube_data).numpy()
	cube_mean = np.mean(cube_data, 1)
	cube_std = np.std(cube_data, 1)

	fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True, sharey=True)
	axes[1].plot(np.arange(sphere_n_eval), sphere_mean, 'r', label='sphere(' + str(sphere_data.shape[1]) + ')')
	axes[1].fill_between(np.arange(sphere_n_eval), sphere_mean - sphere_std, sphere_mean + sphere_std, color='r',
	                     alpha=0.25)
	axes[1].plot(np.arange(cube_n_eval), cube_mean, 'b', label='cube(' + str(cube_data.shape[1]) + ')')
	axes[1].fill_between(np.arange(cube_n_eval), cube_mean - cube_std, cube_mean + cube_std, color='b', alpha=0.25)
	axes[1].set_title('Comparison', fontsize=10)
	axes[1].legend()

	for i, radius in enumerate(sphere_radius_list):
		axes[0].plot(np.arange(radius.size), radius, color=color_matrix[i])
	axes[0].set_title('Spherical', fontsize=10)

	for i, radius in enumerate(cube_radius_list):
		axes[2].plot(np.arange(radius.size), radius, color=color_matrix[i])
	axes[2].set_title('Rectangular', fontsize=10)

	plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=2.0)
	plt.suptitle(title)
	plt.show()

if __name__ == '__main__':
	radius_plot('/home/coh1/Experiments/Hypersphere_ALL/styblinskitang_D20')
	# rosenbrock
	# levy
	# styblinskitang

