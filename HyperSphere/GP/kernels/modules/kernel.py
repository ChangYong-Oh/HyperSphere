import math
import sampyl as smp

import torch
from torch.nn.parameter import Parameter
from HyperSphere.GP.modules.gp_modules import GPModule


class Kernel(GPModule):

	def __init__(self, ndim):
		super(Kernel, self).__init__()
		self.ndim = ndim
		self.log_amp = Parameter(torch.FloatTensor(1))

	def reset_parameters(self):
		self.log_amp.data.normal_()

	def out_of_bounds(self, vec=None):
		return (vec < math.log(0.0001)).any()

	def n_params(self):
		return 1

	def prior(self, vec):
		return smp.normal(vec, mu=0.0, sig=1.0)
