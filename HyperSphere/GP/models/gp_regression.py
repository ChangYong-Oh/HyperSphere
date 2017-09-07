import torch
from torch.autograd import Variable
import torch.nn as nn

from HyperSphere.GP.likelihoods.modules.gaussian import GaussianLikelihood
from HyperSphere.GP.means.modules.constant import ConstantMean
from HyperSphere.GP.kernels.modules.squared_exponential import SquaredExponentialKernel


class GPRegression(nn.Module):

	def __init__(self, kernel=SquaredExponentialKernel(ndim=5), mean=ConstantMean()):
		super(GPRegression, self).__init__()
		self.kernel = kernel
		self.mean = mean
		self.likelihood = GaussianLikelihood()


if __name__ == '__main__':
	GP = GPRegression()
	print(list(GP.parameters()))
