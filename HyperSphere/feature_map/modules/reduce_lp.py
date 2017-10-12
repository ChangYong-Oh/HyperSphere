import sampyl as smp

import torch
from torch.nn.parameter import Parameter
from HyperSphere.GP.modules.gp_modules import GPModule
from HyperSphere.feature_map.functions import reduce_lp


class ReduceLp(GPModule):

	def __init__(self):
		super(ReduceLp, self).__init__()
		self.log_p_minus_one = Parameter(torch.FloatTensor(1))

	def reset_parameters(self):
		self.log_p_minus_one.data.normal_()

	def init_parameters(self):
		self.log_p_minus_one.data.fill_(0.0)

	def out_of_bounds(self, vec=None):
		if vec is None:
			return (self.log_p_minus_one.data < -10).any() or (self.log_p_minus_one.data > 5).any()
		else:
			return (vec < -10).any() or (vec > 5).any()

	def n_params(self):
		return 1

	def param_to_vec(self):
		return self.log_p_minus_one.data.clone()

	def vec_to_param(self, vec):
		self.log_p_minus_one.data = vec

	def prior(self, vec):
		return smp.normal(vec)

	def forward(self, input):
		return reduce_lp.ReduceLp.apply(input, 1 + torch.exp(self.log_p_minus_one))

