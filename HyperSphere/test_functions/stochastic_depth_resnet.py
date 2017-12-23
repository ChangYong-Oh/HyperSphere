import os
import time
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
	time_tag = time.strftime('%H:%M:%S', time.gmtime())
	probability_filename = '/tmp/stochastic_depth_death_rate_' + data_type + '_' + time_tag + '.pkl'
	probability_file = open(probability_filename, 'w')
	probability_list = 1.0 / (1.0 + torch.exp(-4.0 * probability_tensor))
	pickle.dump(list(probability_list), probability_file)
	probability_file.close()

	stochastic_depth_dir = os.path.join(os.path.abspath(os.path.join(os.path.split(__file__)[0], '../../../')), 'img_classification_pk_pytorch')
	save_dir = os.path.join(stochastic_depth_dir, 'save', data_type + '_' + time_tag)
	gpu_device = str(GPUtil.getFirstAvailable()[0])

	cmd_str = 'cd ' + stochastic_depth_dir + ';'
	cmd_str += 'CUDA_VISIBLE_DEVICES=' + gpu_device + ' python main.py'
	cmd_str += ' --data ' + data_type
	cmd_str += ' --arch resnet --depth 110 --death-mode chosen --death-rate-filename ' + probability_filename
	cmd_str += ' --save ' + save_dir
	cmd_str += ' --batch-size 256 --epoch 500'
	subprocess_stdout = subprocess.check_output(cmd_str, shell=True)
	if subprocess_stdout[-1] == '\n':
		subprocess_stdout = subprocess_stdout[:-1]
	output = torch.FloatTensor([[float(subprocess_stdout.split('\n')[-1])]])
	print(output)


if __name__ == '__main__':
	_stochastic_depth_resnet(torch.rand(54), 'cifar100+')



# CUDA_VISIBLE_DEVICES=0 python main.py --data cifar100+ --arch resnet --depth 110 --save save/cifar100+-resnet-110-batch256 --batch-size 256 --epochs 50