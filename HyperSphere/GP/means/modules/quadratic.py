import sampyl as smp

import torch
from torch.nn.parameter import Parameter
from HyperSphere.GP.means.modules.mean import Mean
from HyperSphere.GP.means.functions import quadratic


class QuadraticMean(Mean):

	def __init__(self):
		super(QuadraticMean, self).__init__()
		self.log_a = Parameter(torch.FloatTensor(1))
		self.c = Parameter(torch.FloatTensor(1))

	def reset_parameters(self):
		self.log_a.data.normal_() # a => roughly to 0
		self.c.data.normal_(std=10.0) # approximation to uniform

	def out_of_bounds(self, vec=None):
		if vec is None:
			return False
		else:
			return False

	def n_params(self):
		return 2

	def param_to_vec(self):
		return torch.cat([self.log_a.data, self.c.data])

	def vec_to_param(self, vec):
		self.log_a.data = vec[0]
		self.c.data = vec[1]

	def prior(self, vec):
		return smp.normal(torch.exp(self.log_a).data.squeeze()[0])

	def forward(self, input):
		return quadratic.QuadraticMean.apply(input, torch.exp(self.log_a), self.c)

	def __repr__(self):
		return self.__class__.__name__


if __name__ == '__main__':
	likelihood = ConstantMean()
	print(list(likelihood.parameters()))