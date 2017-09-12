import math

import torch
from torch.autograd import Function, Variable, gradcheck


class NormCDF(Function):

	@staticmethod
	def forward(ctx, input, mu, std):
		ctx.save_for_backward(input, mu, std)
		z = (input - mu) / std
		return 1.0 / (1.0 + torch.exp(-math.pi ** 0.5 * (-0.0004406 * z ** 5 + 0.0418198 * z ** 3 + 0.9 * z)))

	@staticmethod
	def backward(ctx, grad_output):
		input, mu, std = ctx.saved_variables
		grad_input = grad_mu = grad_std = None

		z = (input - mu) / std
		denom = (1.0 + torch.exp(-math.pi ** 0.5 * (-0.0004406 * z ** 5 + 0.0418198 * z ** 3 + 0.9 * z))) ** 2
		numer = -torch.exp(-math.pi ** 0.5 * (-0.0004406 * z ** 5 + 0.0418198 * z ** 3 + 0.9 * z)) * (-math.pi ** 0.5 * (-0.0004406 * 5 * z ** 4 + 0.0418198 * 3 * z ** 2 + 0.9))

		if ctx.needs_input_grad[0]:
			normcdf_grad_input = numer / denom / std
			grad_input = grad_output * normcdf_grad_input
		if ctx.needs_input_grad[1]:
			normcdf_grad_mu = numer / denom / -std
			grad_mu = grad_output * normcdf_grad_mu
		if ctx.needs_input_grad[2]:
			normcdf_grad_std = numer / denom / -std**2
			grad_std = grad_output * normcdf_grad_std
		return grad_input, grad_mu, grad_std


if __name__ == '__main__':
	ndata = 10
	input = Variable(torch.randn(ndata), requires_grad=True)
	mu = Variable(torch.randn(ndata), requires_grad=True)
	std = Variable(torch.randn(ndata).exp_(), requires_grad=True)
	test = gradcheck(NormCDF.apply, (input, mu, std), eps=1e-4, atol=1e-3, rtol=1e-2)
	print(test)
