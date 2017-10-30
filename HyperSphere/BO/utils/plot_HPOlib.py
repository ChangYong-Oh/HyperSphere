import os
import sys
import pickle
import numpy as np

import matplotlib.pyplot as plt


def plot(filename_list):
	n_runs = len(filename_list)
	evaluations_list = []
	evaluations_length = []
	for filename in filename_list:
		file = open(filename)
		data = pickle.load(file)
		evaluations = [elm['result'] for elm in data['trials']]
		evaluations_list.append(evaluations)
		evaluations_length.append(len(evaluations))
		file.close()

	optimum_list = []
	for evaluations, length in zip(evaluations_list, evaluations_length):
		optimum = [np.min(evaluations[:i]) for i in range(1, length + 1)]
		optimum_list.append(np.array(optimum))

	min_length = np.min(evaluations_length)
	optimum_array = np.array([elm[:min_length] for elm in optimum_list])
	y_min = np.min(optimum_array)
	y_max = np.max(np.min(optimum_array[:, :2], 1))
	y_len = y_max - y_min

	steps = np.arange(min_length)
	fig, axes = plt.subplots(2, 1, sharex=True, sharey=True)
	for i in range(n_runs):
		axes[0].plot(steps, optimum_array[i])

	mean = np.mean(optimum_array, 0)
	std = np.std(optimum_array, 0)
	axes[1].plot(steps, mean)
	axes[1].fill_between(steps, mean - 1.96 * std, mean + 1.96 * std, alpha=0.25)

	axes[0].set_ylim(y_min - 0.1 * y_len, y_max + 0.1 * y_len)
	axes[1].set_ylim(y_min - 0.1 * y_len, y_max + 0.1 * y_len)

	title_str = filename_list[0].split('/')[-3]
	plt.suptitle(title_str)

	plt.show()


def generate_filename_list(dir_name, optimizer_name='spearmint_april2013_mod'):
	folder_name = [os.path.join(dir_name, elm) for elm in os.listdir(dir_name) if os.path.isdir(os.path.join(dir_name, elm)) and elm != 'Plots']
	file_name = [os.path.join(elm, optimizer_name + '.pkl') for elm in folder_name if os.path.exists(os.path.join(elm, optimizer_name + '.pkl'))]
	return file_name


if __name__ == '__main__':
	if len(sys.argv) == 3:
		plot(generate_filename_list(sys.argv[1], sys.argv[2]))
	elif len(sys.argv) == 2:
		plot(generate_filename_list(sys.argv[1]))