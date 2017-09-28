import os
import os.path
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch

from HyperSphere.BO.utils.datafile_utils import folder_name_list


def plot(path):
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
	axes[1].set_title('Comparison', fontsize=10)
	axes[1].legend()

	for i in range(sphere_data.shape[1]):
		axes[0].plot(np.arange(sphere_n_eval), sphere_data[:sphere_n_eval, i])
	axes[0].set_title('Spherical', fontsize=10)

	for i in range(cube_data.shape[1]):
		axes[2].plot(np.arange(cube_n_eval), cube_data[:cube_n_eval, i])
	axes[2].set_title('Rectangular', fontsize=10)

	plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=2.0)
	plt.suptitle(title)
	plt.show()


if __name__ == '__main__':
	plot('/home/coh1/Experiments/Hypersphere_ALL/levy_D20')
