from HyperSphere.feature_map.functionals import phi_smooth
from HyperSphere.feature_map.modules.reduce_lp import ReduceLp


def smooth_lp_dim_change(x):
	return x + 1


class SmoothLp(ReduceLp):

	def __init__(self):
		super(SmoothLp, self).__init__()
		self.dim_change = smooth_lp_dim_change

	def forward(self, input):
		return super(SmoothLp, self).forward(phi_smooth(input))


if __name__ == '__main__':
	import torch
	from torch.autograd import Variable
	from HyperSphere.feature_map.functionals import phi_reflection_lp
	n = 10
	dim = 10
	input = Variable(torch.FloatTensor(n, dim).uniform_(-1, 1))
	feature_map = SmoothLp()
	feature_map.reset_parameters()
	print(torch.exp(feature_map.log_p_minus_one.data)[0] + 1)
	output1 = feature_map(input)
	output2 = phi_reflection_lp(input, torch.exp(feature_map.log_p_minus_one.data)[0] + 1)
	print(torch.dist(output1, output2))