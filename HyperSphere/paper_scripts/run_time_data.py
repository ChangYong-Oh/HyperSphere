import os
import sys
import pickle
import time
import numpy as np
import pandas as pd
from datetime import datetime
import torch
import matplotlib.pyplot as plt
from scipy.io import loadmat
from matplotlib.ticker import MultipleLocator
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FormatStrFormatter


FUNC_NAME = ['branin', 'hartmann6', 'levy', 'rosenbrock']
DIMS = [20, 100]


def algorithm_color(algorithm):
	if algorithm == 'spearmint':
		return 'darkorchid'
	if algorithm == 'spearmint_warping':
		return 'indigo'
	if algorithm == 'cube':
		return 'salmon'
	if algorithm == 'cubeard':
		return 'r'
	if algorithm == 'sphereboth':
		return 'green'
	if algorithm == 'sphereorigin':
		return 'limegreen'
	if algorithm == 'spherewarpingboth':
		return 'olive'
	if algorithm == 'spherewarpingorigin':
		return 'darkcyan'


def run_time_smac(smac_dir='/home/coh1/Experiments/smac_ALL_center-random'):
	result = {}
	for elm in sorted(os.listdir(smac_dir)):
		if os.path.isdir(os.path.join(smac_dir, elm)) and '_' in elm and elm.split('_')[0] in FUNC_NAME and int(elm.split('_')[1]) in DIMS:
			exp_dir = os.path.join(smac_dir, elm)
			run_time_list = []
			n_eval_list = []
			for run_instance in os.listdir(exp_dir):
				output_filename = [e for e in os.listdir(os.path.join(exp_dir, run_instance)) if e[-3:]=='out'][0]
				runtime_sentence = open(os.path.join(exp_dir, run_instance, output_filename)).readlines()[-9]
				run_time_list.append(float(runtime_sentence.split(' ')[-2]))
				n_eval_list.append(None)
			result[elm] = (run_time_list, n_eval_list)
	return result


def run_time_tpe(tpe_dir='/home/coh1/Experiments/tpe_ALL_center-random'):
	result = {}
	for elm in sorted(os.listdir(tpe_dir)):
		if os.path.isdir(os.path.join(tpe_dir, elm)) and '_' in elm and elm.split('_')[0] in FUNC_NAME and int(
				elm.split('_')[1]) in DIMS:
			exp_dir = os.path.join(tpe_dir, elm)
			run_time_list = []
			n_eval_list = []
			for run_instance in os.listdir(exp_dir):
				output_filename = [e for e in os.listdir(os.path.join(exp_dir, run_instance)) if e[-3:] == 'out'][0]
				output_fileread = open(os.path.join(exp_dir, run_instance, output_filename)).readlines()
				begin_time = output_fileread[2].split(' ')[1][1:-9].split(':')
				begin_seconds = int(begin_time[0]) * 3600 + int(begin_time[1]) * 60 + int(begin_time[2])
				try:
					end_time = output_fileread[-12].split(' ')[1][1:-9].split(':')
					end_seconds = int(end_time[0]) * 3600 + int(end_time[1]) * 60 + int(end_time[2])
				except ValueError:
					end_time = output_fileread[-11].split(' ')[1][1:-9].split(':')
					end_seconds = int(end_time[0]) * 3600 + int(end_time[1]) * 60 + int(end_time[2])
				if end_seconds < begin_seconds:
					end_seconds += 3600 * 24
				run_time_list.append(float(end_seconds - begin_seconds))
				n_eval_list.append(None)
			result[elm] = (run_time_list, n_eval_list)
	return result


def run_time_spearmint(spearmint_dir='/home/coh1/Experiments/spearmint_ALL_center-random'):
	result = {}
	for elm in sorted(os.listdir(spearmint_dir)):
		if os.path.isdir(os.path.join(spearmint_dir, elm)) and '_' in elm and elm.split('_')[0] in FUNC_NAME and int(elm.split('_')[1]) in DIMS:
			exp_dir = os.path.join(spearmint_dir, elm)
			run_time_list = []
			n_eval_list = []
			for run_instance in os.listdir(exp_dir):
				output_dir = os.path.join(spearmint_dir, elm, run_instance, 'output')
				last_modified_time_list = np.sort([os.stat(os.path.join(output_dir, output_file)).st_mtime for output_file in os.listdir(output_dir)])
				durations = last_modified_time_list[1:] - last_modified_time_list[:-1]
				last_duration = durations[-1]
				durations[1:-1][durations[1:-1] > 2 * last_duration] = 0.5 * (durations[2:] + durations[:-2])[durations[1:-1] > 2 * last_duration]
				run_time_list.append(np.sum(durations))
				n_eval_list.append(durations.size)
			result[elm] = (run_time_list, n_eval_list)
	return result


def run_time_warping(warping_dir='/home/coh1/Experiments/Warping_ALL_center-random'):
	result = {}

	exp_list = []
	for elm in sorted(os.listdir(warping_dir)):
		if os.path.isdir(os.path.join(warping_dir, elm)) and '_' in elm and elm.split('_')[0] in FUNC_NAME and int(elm.split('_')[1]) in DIMS:
			exp_list.append(elm.split('_')[0] + '_' + elm.split('_')[1])
	exp_list = list(np.unique(exp_list))

	for exp in exp_list:
		run_time_list = []
		n_eval_list = []
		for run_instance in os.listdir(warping_dir):
			if exp in run_instance and os.path.isdir(os.path.join(warping_dir, run_instance)):
				output_dir = os.path.join(warping_dir, run_instance, 'output')
				last_modified_time_list = np.sort([os.stat(os.path.join(output_dir, output_file)).st_mtime for output_file in os.listdir(output_dir)])
				durations = last_modified_time_list[1:] - last_modified_time_list[:-1]
				last_duration = durations[-1]
				durations[1:-1][durations[1:-1] > 2 * last_duration] = 0.5 * (durations[2:] + durations[:-2])[durations[1:-1] > 2 * last_duration]
				run_time_list.append(np.sum(durations))
				n_eval_list.append(durations.size)
		result[exp] = (run_time_list, n_eval_list)
	return result


def run_time_additive(additive_dir='/home/coh1/Experiments/Additive_BO_mat_ALL_center-random'):
	result = {}

	exp_list = os.listdir(additive_dir)
	exp_end_info_dict = {}
	for elm in exp_list:
		if elm.split('_')[-1] == 'x.mat':
			time_milliseconds = (datetime.strptime(elm.split('_')[-2], '%Y%m%d-%H:%M:%S:%f') - datetime(1970, 1, 1)).total_seconds()
			exp_end_info_dict[time_milliseconds] = '_'.join(elm.split('_')[:-3])

	time_list = []
	corresponding_exp_list = []
	for key in sorted(exp_end_info_dict.keys()):
		time_list.append(key)
		corresponding_exp_list.append(exp_end_info_dict[key].split('_')[0] + '_' + exp_end_info_dict[key].split('_')[1][1:] + '_' + exp_end_info_dict[key].split('_')[2])

	elapsed_seconds = np.array(time_list)[1:] - np.array(time_list)[:-1]
	corresponding_exp_list = corresponding_exp_list[1:]

	summary = dict(zip(corresponding_exp_list, elapsed_seconds))

	for key, elm in summary.iteritems():
		result_key = '_'.join(key.split('_')[:2])
		result_info = key.split('_')[2]
		try:
			result[result_key][0].append(elm / 5.0)
			result[result_key][1].append(result_info)
		except KeyError:
			result[result_key] = ([], [])

	return result


def run_time_elastic(elastic_dir = '/home/coh1/Experiments/elastic_BO_mat_center-random'):
	result = {}
	filename_list = [os.path.join(elastic_dir, elm) for elm in os.listdir(elastic_dir) if elm[-3:]=='log']
	for elm in os.listdir(elastic_dir):
		if elm[-3:] == 'log':
			if 'branin' in elm:
				result_key = elm[:6] + '_' + elm.split('_')[0][6:]
			elif 'hartmann6' in elm:
				result_key = elm[:9] + '_' + elm.split('_')[0][9:]
			elif 'rosenbrock' in elm:
				result_key = elm[:10] + '_' + elm.split('_')[0][10:]
			elif 'levy' in elm:
				result_key = elm[:4] + '_' + elm.split('_')[0][4:]

			begin_time = (datetime.strptime(elm.split('_')[1][:-4], '%Y%m%d-%H:%M:%S:%f') - datetime(1970, 1, 1)).total_seconds()
			end_time = (datetime.fromtimestamp(os.path.getmtime(os.path.join(elastic_dir, elm))) - datetime(1970, 1, 1)).total_seconds()
			elapsed_time = end_time - begin_time
			try:
				result[result_key][0].append(elapsed_time)
				result[result_key][1].append(None)
			except KeyError:
				result[result_key] = ([], [])

	return result


def run_time_hyper(algo_type, hyper_dir='/home/coh1/Experiments/Hypersphere_ALL_center-random_P=9'):
	result = {}

	exp_list = []
	for elm in sorted(os.listdir(hyper_dir)):
		if os.path.isdir(os.path.join(hyper_dir, elm)) and '_' in elm and algo_type == elm.split('_')[2] and elm.split('_')[0] in FUNC_NAME and int(elm.split('_')[1][1:]) in DIMS:
			exp_list.append(elm.split('_')[0] + '_' + elm.split('_')[1])
	exp_list = list(np.unique(exp_list))

	for exp in exp_list:
		run_time_list = []
		n_eval_list = []
		for run_instance in os.listdir(hyper_dir):
			if '_'.join([exp, algo_type]) == '_'.join(run_instance.split('_')[:3]) and os.path.isdir(os.path.join(hyper_dir, run_instance)):
				# output_dir = os.path.join(hyper_dir, run_instance, 'log')
				# last_modified_time_list = np.sort(
				# 	[os.stat(os.path.join(output_dir, output_file)).st_mtime for output_file in os.listdir(output_dir)])
				# durations = last_modified_time_list[1:] - last_modified_time_list[:-1]
				data_file = open(os.path.join(hyper_dir, run_instance, 'data_config.pkl'))
				durations = np.array(pickle.load(data_file)['elapse_list'])
				data_file.close()
				last_duration = durations[-1]
				durations[1:-1][durations[1:-1] > 2 * last_duration] = 0.5 * (durations[2:] + durations[:-2])[durations[1:-1] > 2 * last_duration]
				run_time_list.append(np.sum(durations))
				n_eval_list.append(durations.size)
		result[exp.split('_')[0] + '_' + exp.split('_')[1][1:]] = (run_time_list, n_eval_list)
	return result


def microseconds_convert(seconds):
	minutes = seconds / 60
	second_remain = seconds % 60
	hours = minutes / 60
	minute_remain = minutes % 60
	days = hours / 24
	hour_remain = hours % 24
	return ('%2d-%02d:%02d:%02d' % (days, hour_remain, minute_remain, second_remain))


if __name__ == '__main__':
	exp = 'branin'

	cube_dir = '/home/coh1/Experiments/Cube_ALL_center-random'
	sphere_dir = '/home/coh1/Experiments/Hypersphere_ALL_center-random_P=3'
	dims = [20, 100]

	result = {}
	# result['elastic'] = run_time_elastic()
	# result['additve'] = run_time_additive()
	result['spearmint'] = run_time_spearmint()
	# result['warping'] = run_time_warping()
	result['cube'] = run_time_hyper('cube', cube_dir)
	# result['cubeard'] = run_time_hyper('cubeard', cube_dir)
	result['sphereboth'] = run_time_hyper('sphereboth', sphere_dir)
	# result['sphereorigin'] = run_time_hyper('sphereorigin', sphere_dir)
	result['spherewarpingboth'] = run_time_hyper('spherewarpingboth', sphere_dir)
	result['spherewarpingorigin'] = run_time_hyper('spherewarpingorigin', sphere_dir)

	plot_data = {}
	for key, data in result.iteritems():
		mean_data = np.empty(len(dims))
		std_data = np.empty(len(dims))
		neval_data = np.empty(len(dims))
		for i, dim in enumerate(dims):
			if exp + '_' + str(dim) in data.keys():
				mean_data[i] = np.mean(data[exp + '_' + str(dim)][0])
				std_data[i] = np.std(data[exp + '_' + str(dim)][0])
				neval_data[i] = np.mean(data[exp + '_' + str(dim)][1])
			else:
				mean_data[i] = 0
				std_data[i] = 0
				neval_data[i] = 0
		plot_data[key] = (mean_data, std_data, neval_data)

	fig, ax = plt.subplots()
	ind = np.arange(len(dims))
	width = 0.15

	norm_z = 1.96
	component_list = []

	plot_type = 'line'

	i = 0
	n_data = len(plot_data)
	legend_name = []
	for key in sorted(plot_data.keys()):
		if key == 'cube':
			legend_name.append('Matern')
		if key == 'sphereboth':
			continue
			legend_name.append('BOCK-W')
		# if key == 'sphereorigin':
		# 	legend_name.append('BOCK')
		if key == 'spherewarpingboth':
			legend_name.append('BOCK+B')
			continue
		if key == 'spherewarpingorigin':
			legend_name.append('BOCK')
		if key == 'spearmint':
			legend_name.append('Spearmint')
		data = plot_data[key]
		color = algorithm_color(key)
		print(key)
		print('dim mean_hour std_hour')
		for i in range(len(dims)):
			print('%s %f %f' % (dims[i], data[0][i]/3600.0, data[1][i]/3600.0))
		if plot_type == 'bar':
			component_list.append(ax.bar(ind + i * width, data[0]/3600.0, width, color=color, bottom=0, yerr=data[1]/3600.0))
			i = i + 1
		else:
			ax.plot(dims, data[0] / 3600.0, color=color, label=legend_name[-1])
			ax.fill_between(dims, data[0] / 3600.0 - data[1] / 3600.0, data[0] / 3600.0 + data[1] / 3600.0, color=color, alpha=0.25)
	if plot_type == 'bar':
		plt.legend(component_list, legend_name, fontsize=16, loc=2)
		ax.set_xticks(ind + width / 2.0 * float(n_data - 1))
		ax.set_xticklabels([str(elm) + ' dim' for elm in dims], fontsize=16)
	else:
		ax.set_xticklabels([str(elm) for elm in range(10, 111, 10)], fontsize=16)
		plt.legend(fontsize=16, loc=2)
	ax.set_xlabel('Dimension', fontsize=20)
	ax.set_ylabel('Hours', fontsize=20, rotation=0)
	ax.yaxis.set_label_coords(0, 1)
	# plt.subplots_adjust(left=0.0, right=0.98, top=0.98, bottom=0.05)
	plt.yticks(fontsize=16)
	plt.tight_layout()
	plt.show()
