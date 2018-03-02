import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

p_2490_2536_filename = os.path.join('/home/coh1/Experiments/StochasticDepthP', 'stochastic_depth_death_rate_cifar100+_20180125-11:42:54:345544_2490&2536.pkl')
p_2494_2489_filename = os.path.join('/home/coh1/Experiments/StochasticDepthP', 'stochastic_depth_death_rate_cifar100+_20180125-11:51:59:879674_2494&2489.pkl')
p_2474_2497_filename = os.path.join('/home/coh1/Experiments/StochasticDepthP', 'stochastic_depth_death_rate_cifar100+_20180130-13:15:35:724102_2474&2497.pkl')


def death_rate_plot(p_filename):
	valid_err = float((os.path.split(p_filename)[1])[-13:-9]) / 100
	test_err = float((os.path.split(p_filename)[1])[-8:-4]) / 100
	f = open(p_filename)
	p_data = pickle.load(f)
	f.close()
	plt.figure(figsize=(10, 8))
	plt.bar(range(1, len(p_data) + 1), p_data)
	plt.xticks(size=24)
	plt.xlabel('n-th residual block', fontsize=24)
	plt.yticks(size=24)
	plt.ylabel('Death rate', fontsize=24)
	plt.title('Validation Error : %5.2f%% / Test Error : %5.2f%%' % (valid_err, test_err), fontsize=24)
	plt.subplots_adjust(left=0.12, right=0.98, top=0.95, bottom=0.1)
	print(np.sum(p_data))
	plt.show()


if __name__ == '__main__':
	death_rate_plot(p_2490_2536_filename)