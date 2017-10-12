import math
import sampyl as smp

import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from HyperSphere.GP.kernels.modules.kernel import Kernel
from HyperSphere.GP.kernels.functions import inner_product


class InnerProductKernel(Kernel):

	def __init__(self, ndim, input_map=lambda x: x, diagonal=True):
		super(InnerProductKernel, self).__init__(input_map)
		self.diag = diagonal
		if diagonal:
			self.sigma_sqrt = Parameter(torch.FloatTensor(ndim))
		else:
			self.sigma_sqrt = Parameter(torch.FloatTensor(ndim, ndim))

	def reset_parameters(self):
		super(InnerProductKernel, self).reset_parameters()
		self.sigma_sqrt.data.normal_()

	def out_of_bounds(self, vec=None):
		if vec is None:
			return super(InnerProductKernel, self).out_of_bounds(self.log_amp)
		else:
			return not super(InnerProductKernel, self).out_of_bounds(vec[:1])

	def n_params(self):
		return super(InnerProductKernel, self).n_params() + self.sigma_chol.numel()

	def param_to_vec(self):
		return torch.cat([self.log_amp.data, self.sigma_chol.data])

	def vec_to_param(self, vec):
		self.log_amp.data = vec[0:1]
		self.sigma_sqrt.data = vec[1:]

	def prior(self, vec):
		return super(InnerProductKernel, self).prior(vec[:1]) + smp.normal(vec[1:])

	def forward(self, input1, input2=None):
		stabilizer = 0
		if input2 is None:
			input2 = input1
			stabilizer = Variable(torch.diag(input1.data.new(input1.size(0)).fill_(1e-6 * math.exp(self.log_amp.data[0]))))
		gram_mat = inner_product.InnerProductKernel.apply(self.input_map(input1), self.input_map(input2), self.log_amp, self.sigma_sqrt if self.diag else self.sigma_sqrt.view(int(self.sigma_sqrt.numel() ** 0.5), -1))
		return gram_mat + stabilizer

	def __repr__(self):
		return self.__class__.__name__ + ' (diag=' + ('True' if self.sigma_sqrt.dim()==1 else 'False') + ')'
