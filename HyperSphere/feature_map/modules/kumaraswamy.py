import numpy as np
from scipy import stats

import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from HyperSphere.GP.modules.gp_modules import GPModule, log_lower_bnd, log_upper_bnd


def id_dim_change(x):
	return x


class Kumaraswamy(GPModule):

	def __init__(self, ndim, max_input=None):
		super(Kumaraswamy, self).__init__()
		self.dim_change = id_dim_change
		self.ndim = ndim
		if max_input is None:
			max_input = Variable(torch.ones(ndim))
		elif isinstance(max_input, float):
			max_input = Variable(torch.ones(ndim) * max_input)
		self.max_input = max_input
		self.log_a = Parameter(torch.FloatTensor(ndim))
		self.log_b = Parameter(torch.FloatTensor(ndim))

	def reset_parameters(self):
		if np.random.uniform() > 0.5:
			self.log_a.data.normal_(mean=0, std=0.1)
		else:
			self.log_a.data.normal_(mean=0, std=2.0)
		if np.random.uniform() > 0.5:
			self.log_b.data.normal_(mean=0, std=0.1).abs_()
		else:
			self.log_b.data.normal_(mean=0, std=2.0).abs_()

	def init_parameters(self):
		self.log_a.data.fill_(0)
		self.log_b.data.fill_(0)

	def out_of_bounds(self, vec=None):
		if vec is None:
			return (self.log_a.data > log_upper_bnd).any() or (self.log_a.data < log_lower_bnd).any() or (self.log_b.data < 0).any() or (self.log_b.data > log_upper_bnd).any()
		else:
			return (vec[:1] > log_upper_bnd).any() or (vec[:1] < log_lower_bnd).any() or (vec[1:] < 0).any() or (vec[1:] > log_upper_bnd).any()

	def n_params(self):
		return 2

	def param_to_vec(self):
		return torch.cat([self.log_a.data, self.log_b.data])

	def vec_to_param(self, vec):
		self.log_a.data = vec[:1]
		self.log_b.data = vec[1:]

	def prior(self, vec):
		# return smp.normal(vec[:1], 0, 0.25) + smp.normal(vec[1:], 0, 0.25)
		return np.sum(np.log(stats.norm.pdf(vec, 0, 0.1) + stats.norm.pdf(vec, 0, 2.0)))

	def forward(self, input):
		a = torch.exp(self.log_a)
		b = torch.exp(self.log_b)
		max_value = self.max_input.type_as(input)
		return self.max_input.type_as(input) * (1 - (1 - (input / max_value).clamp(min=0, max=1) ** a) ** b)


if __name__ == '__main__':
	from HyperSphere.feature_map.functionals import phi_reflection_lp
	n = 10
	dim = 10
	input = Variable(torch.FloatTensor(n, dim).uniform_(-1, 1))
	feature_map = Kumaraswamy()
	feature_map.reset_parameters()
	print(torch.exp(feature_map.log_p_minus_one.data)[0] + 1)
	output1 = feature_map(input)
	output2 = phi_reflection_lp(input, torch.exp(feature_map.log_p_minus_one.data)[0] + 1)
	print(torch.dist(output1, output2))