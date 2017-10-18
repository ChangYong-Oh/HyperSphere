import numpy as np

import torch
from torch.nn.parameter import Parameter
from HyperSphere.GP.modules.gp_modules import GPModule, log_lower_bnd, log_upper_bnd
from HyperSphere.feature_map.functionals import id_transform


class Kernel(GPModule):

	def __init__(self, input_map=None):
		super(Kernel, self).__init__()
		self.log_amp = Parameter(torch.FloatTensor(1))
		if input_map is not None:
			self.input_map = input_map
		else:
			self.input_map = id_transform

	def reset_parameters(self):
		self.log_amp.data.normal_()
		if isinstance(self.input_map, GPModule):
			self.input_map.reset_parameters()

	def init_parameters(self, amp):
		self.log_amp.data.fill_(np.log(amp))
		if isinstance(self.input_map, GPModule):
			self.input_map.init_parameters()

	def log_kernel_amp(self):
		return self.log_amp

	def out_of_bounds(self, vec=None):
		if vec is None:
			if (log_lower_bnd <= self.log_amp.data).all() and (self.log_amp.data <= log_upper_bnd).all():
				if isinstance(self.input_map, GPModule):
					return self.input_map.out_of_bounds()
				return False
		else:
			if log_lower_bnd <= vec[0] and vec[0] <= log_upper_bnd:
				if isinstance(self.input_map, GPModule):
					return self.input_map.out_of_bounds(vec[1:])
				return False
		return True

	def n_params(self):
		cnt = 1
		if isinstance(self.input_map, GPModule):
			for p in self.input_map.parameters():
				cnt += p.numel()
		return cnt

	def param_to_vec(self):
		flat_param_list = [self.log_amp.data.clone()]
		if isinstance(self.input_map, GPModule):
			flat_param_list.append(self.input_map.param_to_vec())
		return torch.cat(flat_param_list)

	def vec_to_param(self, vec):
		self.log_amp.data = vec[:1]
		if isinstance(self.input_map, GPModule):
			self.input_map.vec_to_param(vec[1:])

	def prior(self, vec):
		if isinstance(self.input_map, GPModule):
			return self.input_map.prior(vec[1:])
		return 0

	def forward(self, input1, input2=None):
		raise NotImplementedError
