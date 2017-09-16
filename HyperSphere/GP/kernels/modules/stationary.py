import math
import sampyl as smp

import torch
from torch.nn.parameter import Parameter
from HyperSphere.GP.kernels.modules.kernel import Kernel


class Stationary(Kernel):

	def __init__(self, ndim):
		super(Stationary, self).__init__(ndim)
		self.log_ls = Parameter(torch.FloatTensor(ndim))

	def reset_parameters(self):
		super(Stationary, self).reset_parameters()
		self.log_ls.data.uniform_(0, 2).log()

	def out_of_bounds(self, vec=None):
		if vec is None:
			if not super(Stationary, self).out_of_bounds(self.log_amp):
				return (self.log_ls.data > math.log(10)).any() or (self.log_ls.data < math.log(0.0001)).any()
		else:
			if not super(Stationary, self).out_of_bounds(vec[:1]):
				return (vec[1:] > math.log(10)).any() or (vec < math.log(0.0001)).any()
		return True

	def n_params(self):
		return super(Stationary, self).n_params() + self.ndim

	def param_to_vec(self):
		return torch.cat([self.log_amp.data, self.log_ls.data])

	def vec_to_param(self, vec):
		self.log_amp.data = vec[0:1]
		self.log_ls.data = vec[1:]

	def elastic_vec_to_param(self, vec, func):
		self.log_amp.data = vec[0:1]
		self.log_ls.data = func(vec[1:])

	def prior(self, vec):
		return super(Stationary, self).prior(vec[:1]) + smp.normal(vec[1:], mu=0.0, sig=2.0)

	def __repr__(self):
		return self.__class__.__name__ + ' (' + 'dim=' + str(self.ndim) + ')'
