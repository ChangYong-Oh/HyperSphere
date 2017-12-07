import math

import torch
from torch.autograd import Variable
from HyperSphere.GP.modules.gp_modules import GPModule


def id_dim_change(x):
	return x


class RadiusPeriodize(GPModule):

	def __init__(self, ndim, max_input=None):
		super(RadiusPeriodize, self).__init__()
		self.dim_change = id_dim_change
		self.ndim = ndim
		if max_input is None:
			max_input = Variable(torch.ones(ndim))
		elif isinstance(max_input, float):
			max_input = Variable(torch.ones(ndim) * max_input)
		self.max_input = max_input

	def reset_parameters(self):
		pass

	def init_parameters(self):
		pass

	def out_of_bounds(self, vec=None):
		False

	def n_params(self):
		return 0

	def param_to_vec(self):
		return torch.FloatTensor(0)

	def vec_to_param(self, vec=None):
		pass

	def prior(self, vec=None):
		return 0

	def forward(self, input):
		max_value = self.max_input.type_as(input)
		return max_value * (1 - torch.cos(input / max_value * math.pi)) * 0.5
