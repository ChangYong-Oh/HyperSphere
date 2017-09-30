import numpy as np
import scipy.stats as stats
import sampyl as smp

import torch
from torch.nn.parameter import Parameter
from HyperSphere.GP.modules.gp_modules import GPModule
from HyperSphere.feature_map.functions import reduce_threshold


def sigmoid_numpy(x):
	return 1.0 / (1.0 + np.exp(-x))


def sigmoid_inv_numpy(x):
	return 1.0 - np.log(1.0/x - 1)


def sigmoid_inv(x):
	return 1.0 - torch.log(1.0/x - 1)


class ReduceThreshold(GPModule):

	def __init__(self):
		super(ReduceThreshold, self).__init__()
		self.sigmoid_inv_threshold = Parameter(torch.FloatTensor(1))
		self.alpha = 2.0
		self.beta = 10.0

	def reset_parameters(self):
		self.sigmoid_inv_threshold.data.fill_(sigmoid_inv_numpy(stats.beta.rvs(a=self.alpha, b=self.beta)))

	def model_param_init(self):
		self.sigmoid_inv_threshold.data.fill_(sigmoid_inv_numpy((self.alpha - 1.0)/(self.alpha + self.beta - 2.0)))

	def out_of_bounds(self, vec=None):
		if vec is None:
			return (self.sigmoid_inv_threshold.data < -10).any()
		else:
			return (vec < -10).any()

	def n_params(self):
		return 1

	def param_to_vec(self):
		return self.sigmoid_inv_threshold.data.clone()

	def vec_to_param(self, vec):
		self.sigmoid_inv_threshold.data = vec

	def prior(self, vec):
		return smp.beta(sigmoid_numpy(vec), alpha=self.alpha, beta=self.beta)

	def forward(self, input):
		return reduce_threshold.ReduceThreshold.apply(input, torch.sigmoid(self.sigmoid_inv_threshold))

