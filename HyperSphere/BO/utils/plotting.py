import os
import os.path
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch

from HyperSphere.BO.utils.datafile_utils import folder_name_list

color_matrix = np.random.uniform(0, 1, [10, 3])


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

	sphere_mean, sphere_std = mean_std(sphere_best_history)
	cube_mean, cube_std = mean_std(cube_best_history)

	fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True, sharey=True)
	axes[1].plot(np.arange(sphere_n_eval), sphere_mean, 'r', label='sphere(' + str(len(sphere_output_list)) + ')')
	axes[1].fill_between(np.arange(sphere_n_eval), sphere_mean - sphere_std, sphere_mean + sphere_std, color='r', alpha=0.25)
	axes[1].plot(np.arange(cube_n_eval), cube_mean, 'b', label='cube(' + str(len(cube_output_list)) + ')')
	axes[1].fill_between(np.arange(cube_n_eval), cube_mean - cube_std, cube_mean + cube_std, color='b', alpha=0.25)
	axes[1].set_title('Comparison', fontsize=10)
	axes[1].legend()

	plot_samples(axes[0], sphere_best_history_list, color_matrix, 'Spherical')
	plot_samples(axes[2], cube_best_history_list, color_matrix, 'Rectangular')

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

	sphere_mean, sphere_std = mean_std(sphere_radius_tensor)
	cube_mean, cube_std = mean_std(cube_radius_tensor)

	fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True, sharey=True)
	axes[1].plot(np.arange(sphere_n_eval), sphere_mean, 'r', label='sphere(' + str(len(sphere_radius_list)) + ')')
	axes[1].fill_between(np.arange(sphere_n_eval), sphere_mean - sphere_std, sphere_mean + sphere_std, color='r', alpha=0.25)
	axes[1].plot(np.arange(cube_n_eval), cube_mean, 'b', label='cube(' + str(len(cube_radius_list)) + ')')
	axes[1].fill_between(np.arange(cube_n_eval), cube_mean - cube_std, cube_mean + cube_std, color='b', alpha=0.25)
	axes[1].set_title('Comparison', fontsize=10)
	axes[1].legend()

	plot_samples(axes[0], sphere_radius_list, color_matrix, 'Spherical')
	plot_samples(axes[2], cube_radius_list, color_matrix, 'Rectangular')

	plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=2.0)
	plt.suptitle(title)
	plt.show()


def plot_samples(ax, sample_list, color_matrix, title_str):
	sample_len = [elm.size for elm in sample_list]
	max_len = int(np.max(sample_len) * 1.1)
	for i, sample in enumerate(sample_list):
		ax.plot(np.arange(sample.size), sample, color=color_matrix[i])
	ax.set_title(title_str, fontsize=10)
	ax.set_xticks(np.arange(0, max_len, 50))
	ax.set_xticks(np.arange(0, max_len, 10), minor=True)
	ax.grid(which='both')


def mean_std(sample_tensor):
	sample_data = sample_tensor.data if hasattr(sample_tensor, 'data') else sample_tensor
	sample_data = (sample_data.cpu() if sample_data.is_cuda else sample_data).numpy()
	return np.mean(sample_data, 1), np.std(sample_data, 1)


if __name__ == '__main__':
	optimum_plot('/home/coh1/Experiments/Hypersphere_ALL/styblinskitang_D20')
	# rosenbrock
	# levy
	# styblinskitang

