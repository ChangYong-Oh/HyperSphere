import os
import torch


def model_info():
	exp_data_dir = '/home/coh1/Experiments/Hypersphere_ALL'
	dir_list = sorted(os.listdir(exp_data_dir))
	for elm in dir_list:
		sub_dir = os.path.join(exp_data_dir, elm)
		if os.path.isdir(sub_dir) and 'sphere' in elm and 'cifar' in elm:
			try:
				kernel = torch.load(os.path.join(sub_dir, 'model.pt')).kernel
				print kernel.sphere_kernel.max_power, sub_dir
			except ImportError:
				print('Import Error %s' % (sub_dir))
			except EOFError:
				print('EOF Error %s' % (sub_dir))
			except IOError:
				print('IO Error %s' % (sub_dir))

if __name__ == '__main__':
	model_info()