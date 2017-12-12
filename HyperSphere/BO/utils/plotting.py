import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from HyperSphere.BO.utils.get_data_from_file import get_data

color_list = ['b', 'g', 'r', 'tab:brown', 'm', 'fuchsia', 'k', 'w']


def algorithm_color(algorithm):
	if algorithm == 'hyperopt':
		return 'rosybrown'
	if algorithm == 'smac':
		return 'darkkhaki'
	if algorithm == 'spearmint':
		return 'maroon'
	if algorithm == 'spearmint_warping':
		return 'yellow'
	if algorithm == 'cube':
		return 'salmon'
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
	y_max = np.min([data['optimum'][1] for data in data_list])
	norm_z = 1.0
	plot_data = {}
	for algorithm in algorithms:
		plot_data[algorithm] = {}
		plot_data[algorithm]['sample'] = []
		plot_data[algorithm]['n_samples'] = 0
		plot_data[algorithm]['best'] = None
		min_n_eval = np.min([data['n_eval'] for data in data_list if data['algorithm'] == algorithm])
		min_std_data = np.empty((0, min_n_eval))
		for data in data_list:
			if data['algorithm'] == algorithm:
				plot_data[algorithm]['sample'].append(data['optimum'])
				plot_data[algorithm]['n_samples'] += 1
				min_std_data = np.vstack((min_std_data, data['optimum'][:min_n_eval]))
				y_min = min(y_min, np.min(data['optimum']))
				if plot_data[algorithm]['best'] is None:
					plot_data[algorithm]['best'] = data['optimum'][:min_n_eval]
				else:
					if plot_data[algorithm]['best'][min_n_eval - 1] > data['optimum'][min_n_eval - 1]:
						plot_data[algorithm]['best'] = data['optimum'][:min_n_eval]
		plot_data[algorithm]['mean'] = np.mean(min_std_data, 0)
		plot_data[algorithm]['std'] = np.std(min_std_data, 0)
		plot_data[algorithm]['plot_x'] = np.arange(min_n_eval)
		y_min = min(y_min, np.min(plot_data[algorithm]['mean'] - norm_z * plot_data[algorithm]['std'] * 0.2))
	y_rng = y_max - y_min
	y_min = y_min - 0.1 * y_rng
	y_max = y_max + 0.1 * y_rng

	if type == 'avg':
		gs = gridspec.GridSpec(1, 1)

		ax = plt.subplot(gs[0])
		for key in sorted(plot_data.keys()):
			data = plot_data[key]
			color = algorithm_color(key)
			ax.plot(data['plot_x'], data['mean'], color=color, label=key + '(' + str(data['n_samples']) + ')')
			ax.fill_between(data['plot_x'], data['mean'] - norm_z * data['std'], data['mean'] + norm_z * data['std'], color=color, alpha=0.25)
		ax.set_ylabel('Comparison', rotation=0, fontsize=8)
		ax.yaxis.set_label_coords(-0.06, 0.85)
		ax.set_ylim(y_min, y_max)
		ax.legend()
	elif type == 'sample':
		gs = gridspec.GridSpec(n_algorithms, 1)
		for i, key in enumerate(sorted(plot_data.keys())):
			if i == 0:
				ax = plt.subplot(gs[i])
			else:
				ax = plt.subplot(gs[i], sharex=ax)
			color = algorithm_color(key)
			plot_samples(ax, plot_data[key]['sample'], color, key)
			ax.set_ylim(y_min, y_max)
	elif type == 'best':
		gs = gridspec.GridSpec(1, 1)

		ax = plt.subplot(gs[0])
		for key in sorted(plot_data.keys()):
			data = plot_data[key]
			color = algorithm_color(key)
			ax.plot(data['plot_x'], data['best'], color=color, label=key + '(' + str(data['n_samples']) + ')')
		ax.set_ylabel('Comparison', rotation=0, fontsize=8)
		ax.yaxis.set_label_coords(-0.06, 0.85)
		ax.set_ylim(y_min, y_max)
		ax.legend()
	elif type == 'all':
		gs = gridspec.GridSpec(2, 2)

		ax_avg = plt.subplot(gs[0, 0])
		ax_mean = plt.subplot(gs[1, 0])
		ax_sample = plt.subplot(gs[0, 1])
		ax_best = plt.subplot(gs[1, 1])
		for key in sorted(plot_data.keys()):
			data = plot_data[key]
			color = algorithm_color(key)
			ax_avg.plot(data['plot_x'], data['mean'], color=color, label=key + '(' + str(data['n_samples']) + ')')
			ax_avg.fill_between(data['plot_x'], data['mean'] - norm_z * data['std'], data['mean'] + norm_z * data['std'], color=color, alpha=0.25)
			ax_mean.plot(data['plot_x'], data['mean'], color=color, label=key + '(' + str(data['n_samples']) + ')')
			plot_samples(ax_sample, plot_data[key]['sample'], color)
			ax_best.plot(data['plot_x'], data['best'], color=color, label=key + '(' + str(data['n_samples']) + ')')
		ax_avg.set_title('Mean' + u"\u00B1" + str(norm_z) + u"\u2022" + 'Std')
		ax_avg.set_ylim(y_min, y_max)
		ax_mean.set_title('Mean')
		ax_mean.set_ylim(y_min, y_max)
		ax_sample.set_title('Samples')
		ax_sample.set_ylim(y_min, y_max)
		ax_best.set_title('Best run')
		ax_best.set_ylim(y_min, y_max)
		ax_mean.legend()
	elif type == 'custom':
		gs = gridspec.GridSpec(1, 3)

		ax_mean = plt.subplot(gs[0])
		ax_best = plt.subplot(gs[1])
		ax_sample = plt.subplot(gs[2])
		for key in sorted(plot_data.keys()):
			data = plot_data[key]
			color = algorithm_color(key)
			ax_mean.plot(data['plot_x'], data['mean'], color=color, label=key + '(' + str(data['n_samples']) + ')')
			plot_samples(ax_sample, plot_data[key]['sample'], color)
			ax_best.plot(data['plot_x'], data['best'], color=color, label=key + '(' + str(data['n_samples']) + ')')
		ax_mean.set_title('Mean')
		ax_mean.set_ylim(y_min, y_max)
		ax_sample.set_title('Samples')
		ax_sample.set_ylim(y_min, y_max)
		ax_best.set_title('Best run')
		ax_best.set_ylim(y_min, y_max)
		ax_mean.legend()
		ax_sample.grid()

	plt.subplots_adjust(hspace=0.02)

	plt.suptitle(title)
	plt.show()


def hist_samples(ax, sample_list, color_list):
	for i, sample in enumerate(sample_list):
		ax.hist(sample, color=color_list[i], alpha=0.25)
	ax.yaxis.set_label_coords(-0.06, 0.5)
	plt.setp([ax.get_xticklabels()], visible=False)


def plot_samples(ax, sample_list, color, title_str=None):
	for i, sample in enumerate(sample_list):
		ax.plot(np.arange(sample.size), sample, color=color)
	if title_str is not None:
		ax.set_ylabel(title_str, rotation=0, fontsize=8)


if __name__ == '__main__':
	optimum_plot('levy', 100, type='custom')
	# schwefel
	# rotatedschwefel
	# michalewicz
	# rosenbrock
	# levy
	# styblinskitang
	# rotatedstyblinskitang