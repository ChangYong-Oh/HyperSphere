import math
import sampyl as smp

import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from HyperSphere.GP.kernels.modules.kernel import Kernel, log_lower_bnd, log_upper_bnd
from HyperSphere.GP.kernels.functions import neural_network


class NeuralNetworkKernel(Kernel):

	def __init__(self, ndim, input_map=None, diagonal=True, trainable_amp=True):
		super(NeuralNetworkKernel, self).__init__(input_map=input_map, trainable_amp=trainable_amp)
		self.ndim = ndim
		self.diag = diagonal
		self.sigma_sqrt = Parameter(torch.FloatTensor(ndim)) if diagonal else Parameter(torch.FloatTensor(ndim, ndim))

	def reset_parameters(self):
		super(NeuralNetworkKernel, self).reset_parameters()
		self.sigma_sqrt.data.normal_()
		self.sigma_sqrt.data += torch.ones(self.sigma_sqrt.size()) if self.diag else torch.eye(self.ndim)

	def init_parameters(self, amp=None):
		super(NeuralNetworkKernel, self).init_parameters(amp)
		self.sigma_sqrt.data.zero_()
		self.sigma_sqrt.data += torch.ones(self.sigma_sqrt.size()) if self.diag else torch.eye(self.ndim)

	def out_of_bounds(self, vec=None):
		if vec is None:
			return super(NeuralNetworkKernel, self).out_of_bounds()
		else:
			n_param_super = super(NeuralNetworkKernel, self).n_params()
			return super(NeuralNetworkKernel, self).out_of_bounds(vec[:n_param_super])

	def n_params(self):
		return super(NeuralNetworkKernel, self).n_params() + self.sigma_sqrt.numel()

	def param_to_vec(self):
		return torch.cat([super(NeuralNetworkKernel, self).param_to_vec(), self.sigma_sqrt.data])

	def vec_to_param(self, vec):
		n_param_super = super(NeuralNetworkKernel, self).n_params()
		super(NeuralNetworkKernel, self).vec_to_param(vec[:n_param_super])
		self.sigma_sqrt.data = vec[n_param_super:]

	def prior(self, vec):
		n_param_super = super(NeuralNetworkKernel, self).n_params()
		return super(NeuralNetworkKernel, self).prior(vec[:n_param_super]) + smp.normal(vec[n_param_super:], mu=1.0)

	def forward(self, input1, input2=None):
		stabilizer = 0
		if input2 is None:
			input2 = input1
			stabilizer = Variable(torch.diag(input1.data.new(input1.size(0)).fill_(1e-6)))
		sigma_sqrt = self.sigma_sqrt if self.diag else self.sigma_sqrt.view(int(self.sigma_sqrt.numel() ** 0.5), -1)
		gram_mat = neural_network.neuralNetworkKernel(self.input_map(input1), self.input_map(input2), sigma_sqrt)
		return (gram_mat + stabilizer) * self.kernel_amp()

	def forward_on_identical(self):
		return self.kernel_amp() * (1 + 1e-6)

	def __repr__(self):
		return self.__class__.__name__ + ' (diag=' + ('True' if self.sigma_sqrt.dim()==1 else 'False') + ')'
