import math
import sampyl as smp

import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from HyperSphere.GP.kernels.modules.kernel import Kernel
from HyperSphere.GP.kernels.functions import inner_product


class InnerProductKernel(Kernel):

	def __init__(self, ndim, diagonal=True):
		super(InnerProductKernel, self).__init__(ndim)
		if diagonal:
			self.sigma_chol_L = Parameter(torch.FloatTensor(ndim))
		else:
			self.sigma_chol_L = Parameter(torch.FloatTensor(ndim, ndim))
		self.reset_parameters()

	def reset_parameters(self):
		super(InnerProductKernel, self).reset_parameters()
		self.sigma_chol_L.data.normal_()

	def out_of_bounds(self, vec=None):
		if vec is None:
			return super(InnerProductKernel, self).out_of_bounds(self.log_amp)
		else:
			return not super(InnerProductKernel, self).out_of_bounds(vec[:1])

	def n_params(self):
		return super(InnerProductKernel, self).n_params() + self.sigma_chol.numel()

	def param_to_vec(self):
		return torch.cat([self.log_amp.data, self.sigma_chol.data.view(-1)])

	def vec_to_param(self, vec):
		self.log_amp.data = vec[0:1]
		self.sigma_chol.data = vec[1:]

	def prior(self, vec):
		return super(InnerProductKernel, self).prior(vec[:1]) + smp.normal(vec[1:])

	def forward(self, input1, input2=None):
		stabilizer = 0
		if input2 is None:
			input2 = input1
			stabilizer = Variable(torch.diag(input1.data.new(input1.size(0)).fill_(1e-6 * math.exp(self.log_amp.data[0]))))
		gram_mat = inner_product.SquaredExponentialKernel.apply(input1, input2, self.log_amp, self.sigma_chol_L)
		return gram_mat + stabilizer

	def __repr__(self):
		return self.__class__.__name__ + ' (' + 'dim=' + str(self.ndim) + ', diag=' + ('True' if self.sigma_chol_L.dim()==1 else 'False') + ')'
