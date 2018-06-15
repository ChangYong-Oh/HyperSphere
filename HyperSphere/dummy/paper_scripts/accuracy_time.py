import matplotlib.pyplot as plt
import numpy as np
from HyperSphere.dummy.plotting.plot_color import algorithm_color

from HyperSphere.dummy.plotting.get_data_from_file import get_data_sphere, get_data_HPOlib, get_data_elastic, \
	get_data_additive, get_data_warping
from run_time_data import run_time_hyper, run_time_spearmint, run_time_elastic, run_time_additive, run_time_warping, \
	run_time_smac, run_time_tpe

FUNC_NAME = 'rosenbrock'


def accuarcy_vs_time(dims, suffix='_center-random', P_setting='_P=3'):
	sphere_dir = '/home/coh1/Experiments/Hypersphere_ALL' + suffix + P_setting + '/'
	cube_dir = '/home/coh1/Experiments/Cube_ALL' + suffix + '/'

	result = {}
	result['smac'] = run_time_smac()
	result['hyperopt'] = run_time_tpe()
	result['additive'] = run_time_additive()
	result['elastic'] = run_time_elastic()
	result['spearmint_warping'] = run_time_warping()
	result['spearmint'] = run_time_spearmint()
	result['cube'] = run_time_hyper('cube', cube_dir)
	result['sphereboth'] = run_time_hyper('sphereboth', sphere_dir)
	result['sphereorigin'] = run_time_hyper('sphereorigin', sphere_dir)
	result['spherewarpingboth'] = run_time_hyper('spherewarpingboth', sphere_dir)
	result['spherewarpingorigin'] = run_time_hyper('spherewarpingorigin', sphere_dir)

	for algo, all_func in result.iteritems():
		for exp in all_func.keys():
			if FUNC_NAME + '_' + str(dims) != exp:
				del all_func[exp]
	for algo, data in result.iteritems():
		try:
			result[algo] = data[FUNC_NAME + '_' + str(dims)][0]
		except KeyError:
			print(algo, FUNC_NAME, dims)
			print(data.keys())
			assert 1 == 0
	runtime_dict = result

	data_list = []
	data_list += get_data_HPOlib('/home/coh1/Experiments/spearmint_ALL' + suffix + '/' + FUNC_NAME + '_' + str(dims))
	data_list += get_data_HPOlib('/home/coh1/Experiments/smac_ALL' + suffix + '/' + FUNC_NAME + '_' + str(dims), optimizer_name='smac_2_10_00-dev')
	data_list += get_data_HPOlib('/home/coh1/Experiments/tpe_ALL' + suffix + '/' + FUNC_NAME + '_' + str(dims), optimizer_name='hyperopt_august2013_mod')
	data_list += get_data_warping('/home/coh1/Experiments/Warping_ALL' + suffix + '/', FUNC_NAME, dims)
	multiple_additive_date = get_data_additive('/home/coh1/Experiments/Additive_BO_mat_ALL' + suffix + '/', FUNC_NAME, dims)
	additive_optimum_dict = {}
	for elm in multiple_additive_date:
		try:
			additive_optimum_dict[elm['algorithm']] += elm['optimum'][-1]
		except KeyError:
			additive_optimum_dict[elm['algorithm']] = elm['optimum'][-1]
	additive_data = [elm for elm in multiple_additive_date if elm['algorithm']==min(additive_optimum_dict, key=additive_optimum_dict.get)]
	for elm in additive_data:
		elm['algorithm'] = 'additive'
	data_list += additive_data
	data_list += get_data_elastic('/home/coh1/Experiments/elastic_BO_mat_center-random/', FUNC_NAME, dims)
	data_list += get_data_sphere(cube_dir, ['cube', 'cubeard'], FUNC_NAME, dims)
	data_list += get_data_sphere(sphere_dir, ['sphereboth', 'sphereorigin', 'spherewarpingboth', 'spherewarpingorigin'], FUNC_NAME, dims)


	optimum_dict = {}
	for elm in data_list:
		try:
			optimum_dict[elm['algorithm']] += [elm['optimum'][-1]]
		except KeyError:
			optimum_dict[elm['algorithm']] = [elm['optimum'][-1]]

	plt.rc('pdf', fonttype=42)
	plt.rc('ps', fonttype=42)

	plt.figure()
	for key in runtime_dict.keys():
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
			label = 'Spearmint+'
		elif key == 'additive':
			label = 'AdditiveBO'
		elif key == 'elastic':
			continue
			label = 'ElasticGP'
		elif key == 'hyperopt':
			label = 'TPE'
		elif key == 'smac':
			label = 'SMAC'

		x_mean = np.mean(runtime_dict[key]) / 3600.0
		x_std = np.std(runtime_dict[key]) / 3600.0
		y_mean = np.mean(optimum_dict[key])
		y_std = np.std(optimum_dict[key])
		mew = 4
		if label=='SMAC':
			mew = 3.0
		elif label=='TPE':
			mew = 6.0

		plt.plot([x_mean], [y_mean], '+' if label!='SMAC' else 'x', color=algorithm_color(key) if label!='SMAC' else 'k', label=label, markersize=16, mew=mew)
		# plt.axvline(x=x_mean, ymin=y_mean - y_std, ymax=y_mean + y_std, color=algorithm_color(key))
		# plt.axhline(y=y_mean, xmin=x_mean - x_std, xmax=x_mean + x_std, color=algorithm_color(key))
		print('%20s %10.4f %10.4f %10.4fvs %10.4f %10.4f %10.4f' % (key, x_mean + x_std, x_mean, x_mean + x_std, y_mean - y_std, y_mean, y_mean + y_std))
	plt.legend(fontsize=16, loc=1 if dims == 20 else 2)
	plt.ylabel('Discovered minimum', fontsize=16)
	plt.xlabel('Run time(hours)', fontsize=16)
	plt.yticks(fontsize=16)
	plt.xticks(fontsize=16)
	plt.tight_layout(rect=[0.0, 0.0, 1.0, 1.0])
	plt.show()


if __name__ == '__main__':
	accuarcy_vs_time(dims=20)