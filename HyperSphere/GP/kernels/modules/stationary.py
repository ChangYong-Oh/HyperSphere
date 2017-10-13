import math
import numpy as np
import sampyl as smp

import torch
from torch.nn.parameter import Parameter
from HyperSphere.GP.kernels.modules.kernel import Kernel, log_lower_bnd


class Stationary(Kernel):

	def __init__(self, ndim, input_map=None):
		super(Stationary, self).__init__(input_map)
		self.ndim = ndim
		self.log_ls = Parameter(torch.FloatTensor(ndim))

	def reset_parameters(self):
		super(Stationary, self).reset_parameters()
		self.log_ls.data.uniform_(0, 2).log()

	def out_of_bounds(self, vec=None):
		if vec is None:
			if not super(Stationary, self).out_of_bounds():
				return (self.log_ls.data > math.log(2.0 * self.ndim ** 0.5)).any() or (self.log_ls.data < log_lower_bnd).any()
		else:
			if not super(Stationary, self).out_of_bounds(vec[:super(Stationary, self).n_params()]):
				return (vec[1:] > math.log(2.0 * self.ndim ** 0.5)).any() or (vec < log_lower_bnd).any()
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

	def prior(self, vec, ls_upper_bound=None):
		if ls_upper_bound is None:
			ls_upper_bound = 2.0 * self.ndim ** 0.5
		n_param_super = super(Stationary, self).n_params()
		return super(Stationary, self).prior(vec[:n_param_super]) + smp.uniform(np.exp(vec[n_param_super:]), lower=0.0, upper=ls_upper_bound)

	def __repr__(self):
		return self.__class__.__name__ + ' (' + 'dim=' + str(self.ndim) + ')'
