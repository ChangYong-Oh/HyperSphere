import math
import sampyl as smp

import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from HyperSphere.GP.kernels.functions import squared_exponential


class SquaredExponentialKernel(Module):

	def __init__(self, ndim):
		super(SquaredExponentialKernel, self).__init__()
		self.ndim = ndim
		self.log_amp = Parameter(torch.FloatTensor(1))
		self.log_ls = Parameter(torch.FloatTensor(ndim))
		self.reset_parameters()

	def reset_parameters(self):
		self.log_amp.data.normal_(std=2.0)
		self.log_ls.data.normal_(std=2.0)

	def n_params(self):
		return 1 + self.ndim

	def param_to_vec(self):
		return torch.cat([self.log_amp.data, self.log_ls.data])

	def vec_to_param(self, vec):
		self.log_amp.data = vec[0:1]
		self.log_ls.data = vec[1:]

	def prior(self, vec):
		return smp.normal(vec[0:1], mu=0.0, sig=2.0) + smp.normal(vec[1:], mu=0.0, sig=2.0)

	def forward(self, input1, input2=None):
		stabilizer = 0
		if input2 is None:
			input2 = input1
			stabilizer = Variable(torch.diag(input1.data.new(input1.size(0)).fill_(1e-6 * math.exp(self.log_amp.data[0]))))
		gram_mat = squared_exponential.SquaredExponentialKernel.apply(input1, input2, self.log_amp, self.log_ls)
		return gram_mat + stabilizer

	def __repr__(self):
		return self.__class__.__name__ + ' (' + 'dim=' + str(self.ndim) + ')'


if __name__ == '__main__':
	kernel = SquaredExponentialKernel(ndim=5)
	print(list(kernel.parameters()))
