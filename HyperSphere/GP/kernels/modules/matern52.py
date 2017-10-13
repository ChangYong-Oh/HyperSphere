import math

import torch
from torch.autograd import Variable
from HyperSphere.GP.kernels.modules.stationary import Stationary
from HyperSphere.GP.kernels.functions import matern52


class Matern52(Stationary):

	def __init__(self, ndim, input_map=None, ls_upper_bound=None):
		super(Matern52, self).__init__(ndim, input_map, ls_upper_bound)

	def forward(self, input1, input2=None):
		stabilizer = 0
		if input2 is None:
			input2 = input1
			stabilizer = Variable(torch.diag(input1.data.new(input1.size(0)).fill_(1e-6 * math.exp(self.log_amp.data[0]))))
		gram_mat = matern52.Matern52.apply(self.input_map(input1), self.input_map(input2), self.log_amp, self.log_ls)
		return gram_mat + stabilizer

	def __repr__(self):
		return self.__class__.__name__ + ' (' + 'dim=' + str(self.ndim) + ')'


if __name__ == '__main__':
	kernel = Matern52(ndim=5)
	print(list(kernel.parameters()))
