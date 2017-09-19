import torch
from torch.autograd import Variable

from HyperSphere.GP.models.gp import GP
from HyperSphere.GP.likelihoods.modules.gaussian import GaussianLikelihood
from HyperSphere.GP.means.modules.constant import ConstantMean


class GPRegression(GP):

	def __init__(self, kernel, mean=ConstantMean()):
		super(GPRegression, self).__init__()
		self.kernel = kernel
		self.mean = mean
		self.likelihood = GaussianLikelihood()


if __name__ == '__main__':
	from HyperSphere.GP.kernels.modules.squared_exponential import SquaredExponentialKernel
	GP = GPRegression(kernel=SquaredExponentialKernel(5))
	print(list(GP.named_parameters()))
