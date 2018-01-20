import os
import math
from datetime import datetime
import pickle
import subprocess
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import numpy as np
import GPUtil
import torch
from torch.autograd import Variable
NDIM = 54


def stochastic_depth_resnet_cifar10(probability_tensor):
	return _stochastic_depth_resnet(probability_tensor, 'cifar10+')

stochastic_depth_resnet_cifar10.dim = NDIM


def stochastic_depth_resnet_cifar100(probability_tensor):
	return _stochastic_depth_resnet(probability_tensor, 'cifar100+')

stochastic_depth_resnet_cifar100.dim = NDIM


def transform_with_center(x, center_probability=0.5):
	if isinstance(center_probability, (float, int)):
		center_probability = x.data.clone() * 0 + center_probability
	assert x.numel() == center_probability.numel()
	assert torch.sum(center_probability > 1.0) == 0
	assert torch.sum(center_probability < 0.0) == 0

	shift = []
	for i in range(center_probability.numel()):
		poly_d = center_probability.squeeze()[i]
		if poly_d == 0:
			shift.append(-1.0)
		elif poly_d == 1:
			shift.append(1.0)
		elif 0 < poly_d < 1:
			poly_zeros = np.roots([-0.25, 0, 0.75, 0.5 - poly_d])
			shift.append(poly_zeros[np.argmin(np.abs(poly_zeros))])
	if hasattr(x, 'data'):
		shift = Variable(torch.FloatTensor(shift)).type_as(x).resize_as(x)
	else:
		shift = torch.FloatTensor(shift).type_as(x).view_as(x)

	x = ((x + 1 + shift) * 0.5).clamp(min=0, max=1)
	y = 3 * x ** 2 - 2 * x ** 3

	return y


def _stochastic_depth_resnet(probability_tensor, data_type):
	time_tag = datetime.now().strftime('%Y%m%d-%H:%M:%S:%f')

	stochastic_depth_dir = os.path.join(os.path.abspath(os.path.join(os.path.split(__file__)[0], '../../../')), 'img_classification_pk_pytorch')
	pretrain_dir = os.path.join(stochastic_depth_dir, 'save', 'cifar100+_warm-start')

	save_dir = os.path.join(stochastic_depth_dir, 'save', data_type + '_' + time_tag)

	probability_filename = os.path.join(stochastic_depth_dir, 'save', 'stochastic_depth_death_rate_' + data_type + '_' + time_tag + '.pkl')
	probability_file = open(probability_filename, 'w')
	# Use information given in stochastic resnet paper setting as the center point
	probability_list = transform_with_center(probability_tensor, 0.5)
	pickle.dump(list(probability_list.data if hasattr(probability_list, 'data') else probability_list), probability_file)
	probability_file.close()

	gpu_device = str(GPUtil.getAvailable(order='random', limit=1)[0])

	cmd_str = 'cd ' + stochastic_depth_dir + ';'
	cmd_str += 'CUDA_VISIBLE_DEVICES=' + gpu_device + ' python main.py'
	cmd_str += ' --data ' + data_type + '  --normalized'
	cmd_str += ' --resume ' + os.path.join(pretrain_dir, 'model_best.pth.tar    ') + ' --save ' + save_dir
	cmd_str += ' --death-mode chosen --death-rate-filename ' + probability_filename
	cmd_str += ' --decay_rate 0.1 --decay_epoch_ratio 0.5 --learning-rate 0.01 --epoch 250'
	print(('=' * 20) + 'COMMAND' + ('=' * 20))
	print(cmd_str)
	process = subprocess.Popen(cmd_str, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
	lastline = ''
	while True:
		nextline = process.stdout.readline()
		if nextline == '':
			if process.poll() is not None:
				break
		else:
			lastline = nextline
		sys.stdout.write(nextline)
		sys.stdout.flush()
	return torch.FloatTensor([[float(lastline)]])


if __name__ == '__main__':
	print('Return value : ', _stochastic_depth_resnet(torch.rand(54), 'cifar100+'))



# CUDA_VISIBLE_DEVICES=0 python main.py --data cifar100+ --arch resnet --depth 110 --save save/cifar100+-resnet-110-batch256 --batch-size 256 --epochs 50