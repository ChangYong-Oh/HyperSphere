from HyperSphere.feature_map.functionals import phi_reflection
from HyperSphere.feature_map.modules.reduce_lp import ReduceLp


def reflection_lp_dim_change(x):
	return x + 1


class ReflectionLp(ReduceLp):

	def __init__(self):
		super(ReflectionLp, self).__init__()
		self.dim_change = reflection_lp_dim_change

	def forward(self, input):
		return super(ReflectionLp, self).forward(phi_reflection(input))


if __name__ == '__main__':
	import torch
	from torch.autograd import Variable
	from HyperSphere.feature_map.functionals import phi_reflection_threshold
	n = 10
	dim = 10
	input = Variable(torch.FloatTensor(n, dim).uniform_(-1, 1))
	feature_map = ReflectionLp()
	feature_map.reset_parameters()
	print(torch.sigmoid(feature_map.sigmoid_inv_threshold.data)[0])
	output1 = feature_map(input)
	output2 = phi_reflection_threshold(input, torch.sigmoid(feature_map.sigmoid_inv_threshold.data)[0])
	print(torch.dist(output1, output2))