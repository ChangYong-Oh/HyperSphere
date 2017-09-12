import numpy as np

import torch
from torch.autograd import Function, Variable, gradcheck


class LogDeterminant(Function):

	@staticmethod
	def forward(ctx, matrix):
		ctx.save_for_backward(matrix)
		# To make Cholesky decomposition consider upper and lower parts together
		return 2.0 * torch.sum(torch.log(0.5 * torch.diag(torch.potrf(matrix, False)) + 0.5 * torch.diag(torch.potrf(matrix, True))), 0, True).view(1, 1)

	@staticmethod
	def backward(ctx, grad_output):
		"""
		:param ctx: 
		:param grad_output: grad_output is assumed to be d[Scalar]/dK and size() is n1 x n2 
		:return: 
		"""
		matrix, = ctx.saved_variables
		grad_matrix = None

		if ctx.needs_input_grad[0]:
			grad_matrix = grad_output * matrix.inverse()

		return grad_matrix


if __name__ == '__main__':
	ndim = 2
	A = torch.randn(ndim, ndim)
	matrix = Variable(A.mm(A.t()) + 2.0 * torch.eye(ndim), requires_grad=True)

	eps = 1e-4

	# gradcheck doesn't have to pass all the time.
	test = gradcheck(LogDeterminant.apply, (matrix, ), eps=eps, atol=1e-3, rtol=1e-2)
	print(test)
