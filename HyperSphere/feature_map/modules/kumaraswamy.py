import numpy as np
import sampyl as smp

import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from HyperSphere.GP.modules.gp_modules import GPModule, log_lower_bnd, log_upper_bnd


def id_dim_change(x):
	return x


class Kumaraswamy(GPModule):

	def __init__(self, ndim, max_input=None):
		super(Kumaraswamy, self).__init__()
		self.dim_change = id_dim_change
		self.ndim = ndim
		if max_input is None:
			max_input = Variable(torch.ones(ndim))
		elif isinstance(max_input, float):
			max_input = Variable(torch.ones(ndim) * max_input)
		self.max_input = max_input
		self.log_a = Parameter(torch.FloatTensor(ndim))
		self.log_b = Parameter(torch.FloatTensor(ndim))

	def reset_parameters(self):
		self.log_a.data.exponential_(lambd=1.5).neg_()
		self.log_b.data.exponential_(lambd=1.5)

	def init_parameters(self):
		self.log_a.data.fill_(0.0)
		self.log_b.data.fill_(0.0)

	def out_of_bounds(self, vec=None):
		max_param_mag = 1.5
		if vec is None:
			return (self.log_a.data > 0).any() or (self.log_a.data < -max_param_mag).any() or (self.log_b.data < 0).any() or (self.log_b.data > max_param_mag).any()
		else:
			return (vec[:1] > 0).any() or (vec[:1] < -max_param_mag).any() or (vec[1:] < 0).any() or (vec[1:] > max_param_mag).any()

	def n_params(self):
		return 2

	def param_to_vec(self):
		return torch.cat([self.log_a.data, self.log_b.data])

	def vec_to_param(self, vec):
		self.log_a.data = vec[:1]
		self.log_b.data = vec[1:]

	def prior(self, vec):
		# return smp.normal(vec[:1], 0.0, 2.0) + smp.normal(vec[1:], 0.0, 2.0)
		return smp.exponential(x=-vec[:1], rate=1.5) + smp.exponential(x=vec[1:], rate=1.5)

	def forward(self, input):
		a = torch.exp(self.log_a)
		b = torch.exp(self.log_b)
		max_value = self.max_input.type_as(input)
		return max_value * (1 - (1 - (input / max_value).clamp(min=0, max=1) ** a) ** b)
