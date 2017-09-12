import math
import sampyl as smp

import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from HyperSphere.GP.means.functions import constant


class ConstantMean(Module):

	def __init__(self):
		super(ConstantMean, self).__init__()
		self.const_mean = Parameter(torch.FloatTensor(1))
		self.reset_parameters()

	def reset_parameters(self):
		self.const_mean.data.normal_(std=2.0)

	def n_params(self):
		return 1

	def param_to_vec(self):
		return self.const_mean.data.clone()

	def vec_to_param(self, vec):
		self.const_mean.data = vec

	def prior(self, vec):
		return smp.normal(vec, mu=0.0, sig=2.0)

	def forward(self, input):
		return constant.ConstantMean.apply(input, self.const_mean)

	def __repr__(self):
		return self.__class__.__name__


if __name__ == '__main__':
	likelihood = ConstantMean()
	print(list(likelihood.parameters()))