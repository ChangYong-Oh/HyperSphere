import math
import numpy as np
import sampyl as smp

import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from HyperSphere.GP.kernels.modules.kernel import Kernel, log_lower_bnd, log_upper_bnd


class SphereRadialKernel(Kernel):

	def __init__(self, max_power):
		super(SphereRadialKernel, self).__init__(input_map=None)
		self.max_power = max_power
		self.log_amp_const = Parameter(torch.FloatTensor(1))
		self.log_amp_power = Parameter(torch.FloatTensor(max_power))

	def reset_parameters(self):
		super(SphereRadialKernel, self).reset_parameters()
		self.log_amp_const.data.normal_().mul_(2).sub_(2)
		self.log_amp_power.data.normal_()

	def init_parameters(self, amp):
		super(SphereRadialKernel, self).init_parameters(amp)
		self.log_amp_const.data.fill_(-2)
		self.log_amp_power.data.fill_(0)

	def out_of_bounds(self, vec=None):
		if vec is None:
			if not super(SphereRadialKernel, self).out_of_bounds(self.log_amp):
				if (self.log_amp_power.data < log_lower_bnd).any() or (self.log_amp_power.data > log_upper_bnd).any():
					return True
				return (self.log_amp_const.data < log_lower_bnd).any() or (self.log_amp_const.data > log_upper_bnd).any()
			return True
		else:
			n_super_param = super(SphereRadialKernel, self).n_params()
			if not super(SphereRadialKernel, self).out_of_bounds(vec[:n_super_param]):
				if (vec[n_super_param + 1:] < log_lower_bnd).any() or (vec[n_super_param + 1:] > log_upper_bnd).any():
					return True
				return (vec[n_super_param:n_super_param + 1] < log_lower_bnd).any() or (vec[n_super_param:n_super_param + 1] > log_upper_bnd).any()
			return True

	def n_params(self):
		return super(SphereRadialKernel, self).n_params() + self.max_power + 1

	def param_to_vec(self):
		return torch.cat([super(SphereRadialKernel, self).param_to_vec(), self.log_amp_const.data, self.log_amp_power.data])

	def vec_to_param(self, vec):
		n_super_param = super(SphereRadialKernel, self).n_params()
		super(SphereRadialKernel, self).vec_to_param(vec[:n_super_param])
		self.log_amp_const.data = vec[n_super_param:n_super_param + 1]
		self.log_amp_power.data = vec[n_super_param + 1:]

	def prior(self, vec):
		n_super_param = super(SphereRadialKernel, self).n_params()
		return super(SphereRadialKernel, self).prior(vec[:n_super_param]) + smp.normal(vec[n_super_param:n_super_param + 1], -2, 2) + smp.normal(vec[n_super_param + 1:], 0, 2)

	def forward_on_identity(self):
		value = torch.exp(self.log_amp_const.data)[0] + torch.sum(torch.exp(self.log_amp_power.data))
		return value * torch.exp(self.log_amp.data)[0]

	def forward(self, input1, input2=None):
		stabilizer = 0
		if input2 is None:
			input2 = input1
			stabilizer = Variable(torch.diag(input1.data.new(input1.size(0)).fill_(1e-6 * self.forward_on_identity())))
		inner_prod = input1.mm(input2.t())
		sum_exp = torch.exp(self.log_amp_const) + torch.sum(torch.exp(self.log_amp_power))
		gram_mat = (torch.exp(self.log_amp_const) / sum_exp) + (torch.exp(self.log_amp_power[0]) / sum_exp) * inner_prod
		for p in range(1, self.max_power):
			gram_mat += (torch.exp(self.log_amp_power[p]) / sum_exp) * inner_prod ** (p + 1)
		return torch.exp(self.log_amp) * gram_mat + stabilizer

	def __repr__(self):
		return self.__class__.__name__ + ' (max_power=' + str(self.max_power) + ')'


if __name__ == '__main__':
	kernel = SphereRadialKernel(max_power=3)
	vec = torch.ones(4)
	vec[2:] *= 80
	print(kernel.out_of_bounds(vec))