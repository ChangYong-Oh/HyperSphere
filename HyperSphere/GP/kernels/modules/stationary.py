import math
import sampyl as smp

import torch
from torch.nn.parameter import Parameter
from HyperSphere.GP.kernels.modules.kernel import GPModule, Kernel


class Stationary(Kernel):

	def __init__(self, ndim, input_map=None):
		super(Stationary, self).__init__(ndim, input_map)
		self.log_ls = Parameter(torch.FloatTensor(ndim))

	def reset_parameters(self):
		super(Stationary, self).reset_parameters()
		self.log_ls.data.uniform_(0, 2).log()

	def out_of_bounds(self, vec=None):
		if vec is None:
			if not super(Stationary, self).out_of_bounds():
				return (self.log_ls.data > 0.5 * math.log(2.0 * self.ndim)).any() or (self.log_ls.data < -12).any()
		else:
			if not super(Stationary, self).out_of_bounds(vec[:super(Stationary, self).n_params()]):
				return (vec[1:] > 0.5 * math.log(2.0 * self.ndim)).any() or (vec < -12).any()
		return True

	def n_params(self):
		cnt = super(Stationary, self).n_params() + self.ndim
		return cnt

	def param_to_vec(self):
		return torch.cat([super(Stationary, self).param_to_vec(), self.log_ls.data])

	def vec_to_param(self, vec):
		n_param_super = super(Stationary, self).n_params()
		super(Stationary, self).vec_to_param(vec[:n_param_super])
		self.log_ls.data = vec[n_param_super:]

	def elastic_vec_to_param(self, vec, func):
		n_param_super = super(Stationary, self).n_params()
		super(Stationary, self).vec_to_param(vec[:n_param_super])
		self.log_ls.data = func(vec[n_param_super:])

	def prior(self, vec):
		n_param_super = super(Stationary, self).n_params()
		return super(Stationary, self).prior(vec[:n_param_super]) + smp.normal(vec[n_param_super:], mu=0.0, sig=2.0)

	def __repr__(self):
		return self.__class__.__name__ + ' (' + 'dim=' + str(self.ndim) + ')'
