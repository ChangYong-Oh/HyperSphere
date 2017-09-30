from HyperSphere.feature_map.functionals import phi_periodize
from HyperSphere.feature_map.modules.reduce_threshold import ReduceThreshold


def periodize_threshold_dim_change(x):
	return x + 1


class PeriodizeThreshold(ReduceThreshold):

	def __init__(self):
		super(PeriodizeThreshold, self).__init__()
		self.dim_change = periodize_threshold_dim_change

	def forward(self, input):
		return super(PeriodizeThreshold, self).forward(phi_periodize(input))


if __name__ == '__main__':
	import torch
	from torch.autograd import Variable
	from HyperSphere.feature_map.functionals import phi_periodize_threshold
	n = 10
	dim = 10
	input = Variable(torch.FloatTensor(n, dim).uniform_(-1, 1))
	feature_map = PeriodizeThreshold()
	feature_map.reset_parameters()
	print(torch.sigmoid(feature_map.sigmoid_inv_threshold.data)[0])
	output1 = feature_map(input)
	output2 = phi_periodize_threshold(input, torch.sigmoid(feature_map.sigmoid_inv_threshold.data)[0])
	print(torch.dist(output1, output2))