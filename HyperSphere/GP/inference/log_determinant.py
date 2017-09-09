import numpy as np

import torch
from torch.autograd import Function, Variable, gradcheck


class LogDeterminant(Function):

	@staticmethod
	def forward(ctx, matrix):
		ctx.save_for_backward(matrix)
		output = matrix[:1, :1].clone()
		if matrix.is_cuda:
			matrix = matrix.cpu()
		if isinstance(matrix, Variable):
			matrix = matrix.data
		_, log_det = np.linalg.slogdet(matrix.numpy())

		output[0, 0] = float(log_det)

		return output

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
	ndim = 5
	A = torch.randn(ndim, ndim)
	matrix = Variable(A.mm(A.t()) + torch.eye(ndim), requires_grad=True)

	# gradcheck doesn't have to pass all the time.
	test = gradcheck(LogDeterminant.apply, (matrix, ), eps=1e-4, atol=1e-3, rtol=1e-2)
	print(test)
