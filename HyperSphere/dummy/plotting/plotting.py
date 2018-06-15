import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from HyperSphere.dummy.plotting.get_data_from_file import get_data
from HyperSphere.test_functions.mnist_weight import mnist_weight_baseline
from HyperSphere.dummy.plotting.plot_color import algorithm_color


def optimum_plot(func_name, ndim, type='avg', suffix='_center-random', P_setting='_P=9'):
	data_list = get_data(func_name, ndim, suffix, P_setting)
	title = func_name + '_D' + str(ndim)
	algorithms = np.unique([elm['algorithm'] for elm in data_list])
	n_algorithms = algorithms.size

	plt.rc('pdf', fonttype=42)
	plt.rc('ps', fonttype=42)

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

	if func_name == 'mnist_weight':
		plt.figure(figsize=(10, 8))
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
	elif type[:6] == 'custom':
		gs = gridspec.GridSpec(1, 1)

		# ax_mean = plt.subplot(gs[0])
		# ax_best = plt.subplot(gs[1])
		ax_3rd = plt.subplot(gs[0])
		if func_name == 'mnist_weight':
			baseline_sample = mnist_weight_baseline(ndim=ndim)
			baseline_mean = np.mean(baseline_sample)
			baseline_std = np.std(baseline_sample)
			# ax_mean.axhline(baseline_mean, c='k', label='SGD')
			# ax_3rd.axhline(baseline_mean, c='k', label='SGD')
			# ax_3rd.axhspan(baseline_mean - norm_z * baseline_std, baseline_mean + norm_z * baseline_std, facecolor='gray', alpha=0.5)
			# txt_str = 'n mean std\n'
			# dim_eval = {100: 400, 200: 600, 500: 800}
			# for e in range(dim_eval[ndim]):
			# 	txt_str += ('%d %.4f %.4f\n' % (e + 1, baseline_mean, baseline_std))
			# txt_file = open('/home/coh1/Publications/mnist_result_data/SGD_' + str(ndim) + '.txt', 'wt')
			# txt_file.write(txt_str)
			# txt_file.close()
		for i, key in enumerate(sorted(plot_data.keys())):
			data = plot_data[key]
			color = algorithm_color(key)

			if key == 'sphereboth':
				continue
				label = 'BOCK B'
			elif key == 'sphereorigin':
				continue
				label = 'BOCK-W'
			elif key == 'spherewarpingboth':
				continue
				label = 'BOCK+B'
			elif key == 'spherewarpingorigin':
				label = 'BOCK'
			elif key == 'cube':
				label = 'Matern'
			elif key == 'cubeard':
				continue
				label = 'MaternARD'
			elif key == 'spearmint':
				label = 'Spearmint'
			elif key == 'spearmint_warping':
				label = 'Warping'
			elif 'additiveBO_' in key:
				if '_5_' in key:
					label = 'AdditiveBO'
				else:
					continue
			elif 'hyperopt' == key:
				label = 'TPE'
			elif 'smac' == key:
				label = 'SMAC'
			elif 'elastic' == key:
				label = 'Elastic BO'

			# ax_mean.plot(data['plot_x'], data['mean'], color=color, label=label)
			if type[7:] == 'avg':
				ax_3rd.plot(data['plot_x'], data['mean'], color=color, label=label)
				ax_3rd.fill_between(data['plot_x'], data['mean'] - norm_z * data['std'], data['mean'] + norm_z * data['std'], color=color, alpha=0.25)
				print(('%-30s $\stackbin[%3d,%1d]{}{%4.2f$\pm$%4.2f}$') % (key, data['mean'].size, len(data['sample']), data['mean'][-1], data['std'][-1]))
			elif type[7:] == 'sample':
				plot_samples(ax_3rd, data['sample'], color)
			# ax_best.plot(data['plot_x'], data['best'], color=color, label=key + '(' + str(data['n_samples']) + ')')
		# ax_mean.set_title('Mean')
		# ax_mean.set_ylim(y_min, y_max)
		# if type[7:] == 'avg':
			# ax_3rd.set_title('Mean ' + u"\u00B1" + ' {:.2f}'.format(norm_z) + u"\u2022" + 'Std')
		# elif type[7:] == 'sample':
		# 	ax_3rd.set_title('Samples')
		ax_3rd.set_ylim(y_min, y_max)
		# ax_best.set_title('Best run')
		# ax_best.set_ylim(y_min, y_max)
		# ax_mean.legend(fontsize=12)
		ax_3rd.legend(fontsize=24, loc=3)
		# ax_3rd.xaxis.set_minor_locator(MultipleLocator(50))
		# ax_sample.yaxis.set_minor_locator(MultipleLocator(0.5))
		ax_3rd.grid(which='minor')

	plt.subplots_adjust(hspace=0.02)

	if 'mnist_weight' in title:
		plt.tick_params(axis='both', which='major', labelsize=36)
		# plt.xticks([], [])
		# plt.yticks([], [])
		plt.tight_layout(rect=[0.015, 0.015, 1, 1.0])
	else:
		plt.tight_layout()
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
	# optimum_plot('mnist_weight', 100, type='custom_avg', suffix='_center-random', P_setting='_P=3')
	optimum_plot('mnist_weight', 500, type='custom_avg', suffix='_train_valid_test', P_setting='')
	# schwefel
	# rotatedschwefel
	# michalewicz
	# rosenbrock
	# levy
	# styblinskitang
	# rotatedstyblinskitang
	# mnist_weight
	# cifar10_weight 1930 or 1920