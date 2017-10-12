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
		self.log_amp0 = Parameter(torch.FloatTensor(1))
		self.max_power = max_power
		if max_power > 1:
			self.log_amps = Parameter(torch.FloatTensor(max_power - 1))

	def reset_parameters(self):
		super(SphereRadialKernel, self).reset_parameters()
		self.log_amp0.data.normal_().mul_(2).sub_(2)
		if self.max_power > 1:
			self.log_amps.data.normal_()

	def init_parameters(self, amp):

		self.log_amp0.data.fill_(-3)
		self.log_amp.data.fill_(np.log(amp / self.max_power))
		self.log_amps.data.fill_(np.log(amp / self.max_power))
		# self.log_amp.data.fill_(np.log(amp * 2 / (self.max_power + 1)))
		# self.log_amps.data = torch.log(amp * torch.arange(self.max_power - 1, 0, -1).type_as(self.log_amps.data) * 2 / (self.max_power * (self.max_power + 1)))

	def out_of_bounds(self, vec=None):
		if vec is None:
			if not super(SphereRadialKernel, self).out_of_bounds(self.log_amp):
				if self.max_power > 1:
					if (self.log_amps.data < log_lower_bnd).any() or (self.log_amps.data > log_upper_bnd).any():
						return True
				return (self.log_amp0.data < log_lower_bnd).any() or (self.log_amp0.data > log_upper_bnd).any()
			return True
		else:
			if not super(SphereRadialKernel, self).out_of_bounds(vec[1:2]):
				if self.max_power > 1:
					if (vec[2:] < log_lower_bnd).any() or (vec[2:] > log_upper_bnd).any():
						return True
				return (vec[:1] < log_lower_bnd).any() or (vec[:1] > log_upper_bnd).any()
			return True

	def n_params(self):
		return self.max_power + 1

	def param_to_vec(self):
		vec = torch.cat([self.log_amp0.data, self.log_amp.data])
		if self.max_power > 1:
			vec = torch.cat([vec, self.log_amps.data])
		return vec

	def vec_to_param(self, vec):
		self.log_amp0.data = vec[:1]
		self.log_amp.data = vec[1:2]
		if self.max_power > 1:
			self.log_amps.data = vec[2:]

	def prior(self, vec):
		likelihood = super(SphereRadialKernel, self).prior(vec[1:2]) + smp.normal(vec[:1], -2, 2)
		if self.max_power > 1:
			likelihood += smp.normal(vec[2:], -2, 2)
		return likelihood

	def forward_on_identity(self):
		value = torch.exp(self.log_amp0.data)[0] + torch.exp(self.log_amp.data)[0]
		if self.max_power > 1:
			value += torch.sum(torch.exp(self.log_amps.data))
		return value

	def forward(self, input1, input2=None):
		stabilizer = 0
		if input2 is None:
			input2 = input1
			stabilizer = Variable(torch.diag(input1.data.new(input1.size(0)).fill_(1e-6 * self.forward_on_identity())))
		inner_prod = input1.mm(input2.t())
		gram_mat = torch.exp(self.log_amp0) + torch.exp(self.log_amp) * inner_prod
		if self.max_power > 1:
			for p in range(2, self.max_power + 1):
				gram_mat += torch.exp(self.log_amps[p - 2]) * inner_prod ** p
		return gram_mat + stabilizer

	def __repr__(self):
		return self.__class__.__name__ + ' (max_power=' + str(self.max_power) + ')'


if __name__ == '__main__':
	kernel = SphereRadialKernel(max_power=3)
	vec = torch.ones(4)
	vec[2:] *= 80
	print(kernel.out_of_bounds(vec))