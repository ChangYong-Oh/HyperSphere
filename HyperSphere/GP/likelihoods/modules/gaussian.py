import math
import sampyl as smp

import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from HyperSphere.GP.likelihoods.functions import gaussian


class GaussianLikelihood(Module):

	def __init__(self):
		super(GaussianLikelihood, self).__init__()
		self.log_noise_var = Parameter(torch.FloatTensor(1))
		self.reset_parameters()

	def reset_parameters(self):
		self.log_noise_var.data.normal_(std=2.0)

	def n_params(self):
		return 1

	def param_to_vec(self):
		return self.log_noise_var.data.clone()

	def vec_to_param(self, vec):
		self.log_noise_var.data = vec

	def prior(self, vec):
		return smp.normal(vec, mu=0.0, sig=2.0)

	def forward(self, input):
		return gaussian.GaussianLikelihood.apply(input, self.log_noise_var)

	def __repr__(self):
		return self.__class__.__name__


if __name__ == '__main__':
	likelihood = GaussianLikelihood()
	print(list(likelihood.parameters()))
