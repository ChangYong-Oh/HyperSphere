import os
import sys
import pickle
import time
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FormatStrFormatter


EXPS = ['rosenbrock_D20']
ALGORITHMS = ['sphereboth', 'sphereorigin', 'spherewarpingboth', 'spherewarpingorigin']
ALGO_NAMES = {}
ALGO_NAMES['sphereboth'] = 'BOCK B'
ALGO_NAMES['sphereorigin'] = 'BOCK'
ALGO_NAMES['spherewarpingboth'] = 'BOCK W B'
ALGO_NAMES['spherewarpingorigin'] = 'BOCK W'
DATA_DIR = '/home/coh1/Experiments/Hypersphere_P_comparison'
P = [3, 5, 7, 9]


def algorithm_color(algorithm):
	if algorithm == 'sphereboth':
		return 'green'
	if algorithm == 'sphereorigin':
		return 'limegreen'
	if algorithm == 'spherewarpingboth':
		return 'olive'
	if algorithm == 'spherewarpingorigin':
		return 'darkcyan'


def get_data():
	result_dict = {}
	for exp in EXPS:
		for algo in ALGORITHMS:
			comparison_dict = {}
			for p in P:
				p_dir = os.path.join(DATA_DIR, 'P=' + str(p))
				optimum_list = []
				for instance in os.listdir(p_dir):
					if '_'.join([exp, algo]) in instance:
						instance_dir = os.path.join(p_dir, instance)
						kernel = torch.load(os.path.join(instance_dir, 'model.pt')).kernel
						assert int(str(kernel)[-2:-1]) == p
						data_file = open(os.path.join(instance_dir, 'data_config.pkl'))
						output = pickle.load(data_file)['output']
						optimum = np.array([torch.min(output[:i].data) for i in range(1, output.numel() + 1)])
						data_file.close()
						optimum_list.append(optimum)
				comparison_dict[p] = optimum_list
			result_dict['_'.join([exp, algo])] = comparison_dict
	return result_dict


def plot_comparison():
	comparison_data = get_data()
	summary_data = {}
	for key in comparison_data.keys():
		p_data = comparison_data[key]
		stacked_data = {}
		for p in P:
			n_min_eval = np.min([len(elm) for elm in p_data[p]])
			stacked_data[p] = np.stack([elm[:n_min_eval] for elm in p_data[p]])
		summary_data[key] = stacked_data
	for key in sorted(summary_data.keys()):
		exp = '_'.join(key.split('_')[:2])
		algo = key.split('_')[-1]
		print(exp, ALGO_NAMES[algo])
		for p in sorted(P):
			optimum_data = summary_data[key][p]
			result_samples = '/'.join(['%12.4f' % elm for elm in optimum_data[:, -1]])
			middle_three = sorted(optimum_data[:, -1])[1:-1]
			middle_three_str = '%12.4f(%10.4f)   %8.2f$\pm$%-8.2f' % (np.mean(middle_three), np.std(middle_three), np.mean(middle_three), np.std(middle_three))
			print(('%d %12.4f(%10.4f)   %8.2f$\pm$%-8.2f' % (p, np.mean(optimum_data[:, -1]), np.std(optimum_data[:, -1]), np.mean(optimum_data[:, -1]), np.std(optimum_data[:, -1]))) + result_samples + middle_three_str)


if __name__ == '__main__':
	plot_comparison()


