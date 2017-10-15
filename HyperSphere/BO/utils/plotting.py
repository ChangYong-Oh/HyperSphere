import os
import os.path
import pickle

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch

from HyperSphere.BO.utils.datafile_utils import folder_name_list

color_list = ['b', 'g', 'r', 'tab:brown', 'm', 'p', 'k', 'w']


def optimum_plot(path):
	title = os.path.split(path)[1]
	grouped_folder_list = folder_name_list(path)
	group =  grouped_folder_list.keys()
	n_group = len(group)

	output_list_dict = dict()
	n_eval_dict = dict()
	output_tensor_dict = dict()
	best_history_dict = dict()
	best_history_list_dict = dict()

	for g in group:
		output_list_dict[g] = []
		for folder in grouped_folder_list[g]:
			f = open(os.path.join(folder, 'data_config.pkl'))
			output = pickle.load(f)['output'].squeeze(1)
			output_list_dict[g].append(output)
			f.close()
		n_eval_dict[g] = np.min([elm.size(0) for elm in output_list_dict[g]])
		output_tensor_dict[g] = torch.stack([elm[:n_eval_dict[g]] for elm in output_list_dict[g]], 1)
		best_history_dict[g] = output_tensor_dict[g].clone()
		for i in range(1, best_history_dict[g].size(0)):
			best_history_dict[g][i], _ = torch.min(output_tensor_dict[g][:i + 1], 0)
		best_history_list_dict[g] = []
		for elm in output_list_dict[g]:
			best_history_list_dict[g].append(np.array([torch.min(elm.data[:d]) for d in range(1, elm.numel() + 1)]))

	mean_dict = dict()
	std_dict = dict()
	for g in group:
		mean_dict[g], std_dict[g] = mean_std(best_history_dict[g])

	gs = gridspec.GridSpec(n_group + 3, 1)

	ax_big = plt.subplot(gs[n_group:])
	for i, g in enumerate(group):
		ax_big.plot(np.arange(n_eval_dict[g]), mean_dict[g], color=color_list[i], label=g + '(' + str(len(output_list_dict[g])) + ')')
		ax_big.fill_between(np.arange(n_eval_dict[g]), mean_dict[g] - std_dict[g], mean_dict[g] + std_dict[g], color=color_list[i], alpha=0.25)
	ax_big.set_ylabel('Comparison', rotation=0, fontsize=8)
	ax_big.yaxis.set_label_coords(-0.06, 0.85)
	ax_big.legend()

	for i, g in enumerate(group):
		ax = plt.subplot(gs[i], sharex=ax_big, sharey=ax_big)
		plot_samples(ax, best_history_list_dict[g], color_list, g)

	plt.subplots_adjust(hspace=0.02)

	plt.suptitle(title)
	plt.show()


def radius_plot(path):
	title = os.path.split(path)[1]
	grouped_folder_list = folder_name_list(path)
	group = grouped_folder_list.keys()
	n_group = len(group)

	radius_list_dict = dict()
	n_eval_dict = dict()
	radius_tensor_dict = dict()

	for g in group:
		radius_list_dict[g] = []
		for folder in grouped_folder_list[g]:
			f = open(os.path.join(folder, 'data_config.pkl'))
			input = pickle.load(f)['x_input'].squeeze(1)
			radius_list_dict[g].append(torch.sqrt(torch.sum(input ** 2, 1)))
			f.close()
		n_eval_dict[g] = np.min([elm.size(0) for elm in radius_list_dict[g]])
		radius_tensor_dict[g] = torch.stack([elm[:n_eval_dict[g]] for elm in radius_list_dict[g]], 1)
		radius_list_dict[g] = [(elm.data if hasattr(elm, 'data') else elm).numpy() for elm in radius_list_dict[g]]

	mean_dict = dict()
	std_dict = dict()
	for g in group:
		mean_dict[g], std_dict[g] = mean_std(radius_tensor_dict[g])

	fig, axes = plt.subplots(n_group, 2, sharex='col')

	for i, g in enumerate(group):
		plot_samples(axes[i, 0], radius_list_dict[g], color_list, g)
		hist_samples(axes[i, 1], radius_list_dict[g], color_list)

	plt.setp([axes[-1, 0].get_xticklabels()], visible=True)
	plt.setp([axes[-1, 1].get_xticklabels()], visible=True)

	plt.subplots_adjust(hspace=0.02)

	plt.suptitle(title)
	plt.show()


def hist_samples(ax, sample_list, color_list):
	for i, sample in enumerate(sample_list):
		ax.hist(sample, color=color_list[i], alpha=0.25)
	ax.yaxis.set_label_coords(-0.06, 0.5)
	plt.setp([ax.get_xticklabels()], visible=False)


def plot_samples(ax, sample_list, color_list, title_str):
	sample_len = [elm.size for elm in sample_list]
	max_len = int(np.max(sample_len) * 1.1)
	for i, sample in enumerate(sample_list):
		ax.plot(np.arange(sample.size), sample, color=color_list[i])
	ax.set_ylabel(title_str, rotation=0, fontsize=8)
	ax.yaxis.set_label_coords(-0.06, 0.5)
	# ax.set_xticks(np.arange(0, max_len, 50))
	# ax.set_xticks(np.arange(0, max_len, 10), minor=True)
	plt.setp([ax.get_xticklabels()], visible=False)

	ax.grid(which='both')


def mean_std(sample_tensor):
	sample_data = sample_tensor.data if hasattr(sample_tensor, 'data') else sample_tensor
	sample_data = (sample_data.cpu() if sample_data.is_cuda else sample_data).numpy()
	return np.mean(sample_data, 1), np.std(sample_data, 1)


if __name__ == '__main__':
	path = '/home/coh1/Experiments/Hypersphere_ALL/levy_D20'
	optimum_plot(path)
	radius_plot(path)
	# rosenbrock
	# levy
	# styblinskitang

