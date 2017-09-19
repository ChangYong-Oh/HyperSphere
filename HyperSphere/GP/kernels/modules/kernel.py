import math
import sampyl as smp

import torch
from torch.nn.parameter import Parameter
from HyperSphere.GP.modules.gp_modules import Module, GPModule


class Kernel(GPModule):

	def __init__(self, ndim, input_map=lambda x: x):
		super(Kernel, self).__init__()
		self.ndim = ndim
		self.log_amp = Parameter(torch.FloatTensor(1))
		self.input_map = input_map

	def reset_parameters(self):
		self.log_amp.data.normal_()
		if isinstance(self.input_map, Module):
			self.input_map.reset_parameters()

	def out_of_bounds(self, vec=None):
		if not (vec[-1] < math.log(0.0001)).any():
			if isinstance(self.input_map, GPModule):
				return self.input_map.out_of_bounds(vec[:-1])
			return False
		return True

	def n_params(self):
		cnt = 1
		if isinstance(self.input_map, Module):
			for p in self.input_map.parameters():
				cnt += p.numel()
		return cnt

	def prior(self, vec):
		likelihood = smp.normal(vec[-1:], mu=0.0, sig=1.0)
		if isinstance(self.input_map, GPModule):
			likelihood += self.input_map.prior(vec[:-1])
		return likelihood
