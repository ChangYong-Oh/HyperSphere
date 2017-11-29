import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from HyperSphere.BO.utils.get_data_from_file import get_data

color_list = ['b', 'g', 'r', 'tab:brown', 'm', 'p', 'k', 'w']


def algorithm_color(algorithm):
	if algorithm == 'hyperopt':
		return 'rosybrown'
	if algorithm == 'smac':
		return 'darkkhaki'
	if algorithm == 'spearmint':
		return 'maroon'
	if algorithm == 'spearmint_warping':
		return 'yellow'
	if algorithm == 'cubeard':
		return 'orange'
	if algorithm[:10] == 'additiveBO':
		return 'g'
	if algorithm == 'elasticGP':
		return 'coral'
	if algorithm == 'sphereboth':
		return 'royalblue'
	if algorithm == 'sphereorigin':
		return 'blueviolet'
	if algorithm == 'spherewarpingboth':
		return 'fuchsia'
	if algorithm == 'spherewarpingorigin':
		return 'red'


def optimum_plot(func_name, ndim, type='avg'):
	data_list = get_data(func_name, ndim)
	title = func_name + '_D' + str(ndim)
	algorithms = np.unique([elm['algorithm'] for elm in data_list])
	n_algorithms = algorithms.size

	y_min = np.inf
	y_max = np.max(np.min([data['optimum'][:2] for data in data_list]))
	norm_z = 0.2
	plot_data = {}
	for algorithm in algorithms:
		plot_data[algorithm] = {}
		plot_data[algorithm]['sample'] = []
		plot_data[algorithm]['n_samples'] = 0
		min_n_eval = np.min([data['n_eval'] for data in data_list if data['algorithm'] == algorithm])
		min_std_data = np.empty((0, min_n_eval))
		for data in data_list:
			if data['algorithm'] == algorithm:
				plot_data[algorithm]['sample'].append(data['optimum'])
				plot_data[algorithm]['n_samples'] += 1
				min_std_data = np.vstack((min_std_data, data['optimum'][:min_n_eval]))
				y_min = min(y_min, np.min(data['optimum']))
		plot_data[algorithm]['mean'] = np.mean(min_std_data, 0)
		plot_data[algorithm]['std'] = np.std(min_std_data, 0)
		plot_data[algorithm]['plot_x'] = np.arange(min_n_eval)
		y_min = min(y_min, np.min(plot_data[algorithm]['mean'] - norm_z * plot_data[algorithm]['std'] * 0.2))

	if type == 'avg':
		gs = gridspec.GridSpec(1, 1)

		ax_big = plt.subplot(gs[0])
		for key in sorted(plot_data.keys()):
			data = plot_data[key]
			color = algorithm_color(key)
			ax_big.plot(data['plot_x'], data['mean'], color=color, label=key + '(' + str(data['n_samples']) + ')')
			ax_big.fill_between(data['plot_x'], data['mean'] - norm_z * data['std'], data['mean'] + norm_z * data['std'], color=color, alpha=0.25)
		ax_big.set_ylabel('Comparison', rotation=0, fontsize=8)
		ax_big.yaxis.set_label_coords(-0.06, 0.85)
		# ax_big.set_ylim(y_min, y_max)
		ax_big.legend()
	elif type == 'sample':
		gs = gridspec.GridSpec(n_algorithms, 1)
		for i, key in enumerate(plot_data.keys()):
			if i == 0:
				ax = plt.subplot(gs[i])
			else:
				ax = plt.subplot(gs[i], sharex=ax)
			plot_samples(ax, plot_data[key]['sample'], color_list, key)
			ax.set_ylim(y_min, y_max)

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


if __name__ == '__main__':
	optimum_plot('michalewicz', 20, type='avg')
	# schwefel
	# rotatedschwefel
	# michalewicz
	# rosenbrock
	# levy
	# styblinskitang
	# rotatedstyblinskitang