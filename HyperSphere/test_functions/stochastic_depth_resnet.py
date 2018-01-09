import os
import math
from datetime import datetime
import pickle
import subprocess
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import GPUtil
import torch


def stochastic_depth_resnet_cifar10(probability_tensor):
	return _stochastic_depth_resnet(probability_tensor, 'cifar10+')

stochastic_depth_resnet_cifar10.dim = 54


def stochastic_depth_resnet_cifar100(probability_tensor):
	return _stochastic_depth_resnet(probability_tensor, 'cifar100+')

stochastic_depth_resnet_cifar100.dim = 54


def _stochastic_depth_resnet(probability_tensor, data_type):
	time_tag = datetime.now().strftime('%Y%m%d-%H:%M:%S:%f')

	stochastic_depth_dir = os.path.join(os.path.abspath(os.path.join(os.path.split(__file__)[0], '../../../')), 'img_classification_pk_pytorch')
	save_dir = os.path.join(stochastic_depth_dir, 'save', data_type + '_' + time_tag)

	probability_filename = os.path.join(stochastic_depth_dir, 'stochastic_depth_death_rate_' + data_type + '_' + time_tag + '.pkl')
	probability_file = open(probability_filename, 'w')
	probability_list = 1.0 / (1.0 + torch.exp(-4.0 * probability_tensor + math.log(3)))
	pickle.dump(list(probability_list.data if hasattr(probability_list, 'data') else probability_list), probability_file)
	probability_file.close()

	gpu_device = str(GPUtil.getFirstAvailable()[0])

	cmd_str = 'cd ' + stochastic_depth_dir + ';'
	cmd_str += 'CUDA_VISIBLE_DEVICES=' + gpu_device + ' python main.py'
	cmd_str += ' --data ' + data_type
	cmd_str += ' --arch resnet --depth 110 --death-mode chosen --death-rate-filename ' + probability_filename
	cmd_str += ' --save ' + save_dir
	cmd_str += ' --batch-size 256 --epoch 500'
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