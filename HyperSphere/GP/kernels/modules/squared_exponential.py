import math

import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from HyperSphere.GP.kernels.functions import squared_exponential


class SquaredExponentialKernel(Module):

	def __init__(self, ndim):
		super(SquaredExponentialKernel, self).__init__()
		self.ndim = ndim
		self.amp = Parameter(torch.FloatTensor(1))
		self.ls = Parameter(torch.FloatTensor(ndim))
		self.reset_parameters()

	def reset_parameters(self):
		self.amp.data.fill_(1)
		self.ls.data.fill_(1)

	def forward(self, input1, input2=None):
		if input2 is None:
			input2 = input1
		return squared_exponential.SquaredExponentialKernel.apply(input1, input2, self.amp, self.ls)

	def __repr__(self):
		return self.__class__.__name__ + ' (' + 'dim=' + str(self.ndim) + ')'


if __name__ == '__main__':
	kernel = SquaredExponentialKernel(ndim=5)
	print(list(kernel.parameters()))