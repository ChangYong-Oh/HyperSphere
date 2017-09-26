import sys

import numpy as np

import torch
from torch.autograd import Function, Variable, gradcheck


class LogDeterminant(Function):

	@staticmethod
	def forward(ctx, matrix):
		ctx.save_for_backward(matrix)

		# matrix_tensor = matrix.data if hasattr(matrix, 'data') else matrix
		# matrix_numpy = (matrix_tensor.cpu() if matrix_tensor.is_cuda else matrix_tensor).numpy()
		# _, logdet = np.linalg.slogdet(matrix_numpy)
		# result = matrix_tensor.new(1, 1).fill_(float(logdet))
		# return Variable(result) if hasattr(matrix, 'data') else result

		# one_vector = (matrix.data if hasattr(matrix, 'data') else matrix).new(matrix.size(0)).fill_(1)
		# id_matrix = torch.diag(Variable(one_vector) if hasattr(matrix, 'data') else one_vector)
		# stabilizer = 0
		# while True:
		# 	try:
		# 		chol_from_upper = torch.potrf(matrix + id_matrix * stabilizer, True)
		# 		chol_from_lower = torch.potrf(matrix + id_matrix * stabilizer, False)
		# 		break
		# 	except RuntimeError:
		# 		if stabilizer == 0:
		# 			stabilizer = np.abs(torch.min(torch.symeig(matrix)[0]))
		# 		stabilizer *= 1 + 1e-4
		#
		# return (torch.sum(torch.log(torch.diag(chol_from_upper)), 0, keepdim=True) + torch.sum(torch.log(torch.diag(chol_from_lower)), 0, keepdim=True)).view(1, 1)

		try:
			chol_from_upper = torch.potrf(matrix, True)
			chol_from_lower = torch.potrf(matrix, False)
			return (torch.sum(torch.log(torch.diag(chol_from_upper)), 0, keepdim=True) + torch.sum(torch.log(torch.diag(chol_from_lower)), 0, keepdim=True)).view(1, 1)
		except RuntimeError:
			matrix_tensor = matrix.data if hasattr(matrix, 'data') else matrix
			matrix_numpy = (matrix_tensor.cpu() if matrix_tensor.is_cuda else matrix_tensor).numpy()
			_, logdet = np.linalg.slogdet(matrix_numpy)
			result = matrix_tensor.new(1, 1).fill_(float(logdet))
			return Variable(result) if hasattr(matrix, 'data') else result

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
	matrix = Variable(A.mm(A.t()) + np.random.uniform(0, 0.2) * torch.eye(ndim), requires_grad=True)

	eps = 1e-4

	# gradcheck doesn't have to pass all the time.
	test = gradcheck(LogDeterminant.apply, (matrix, ), eps=eps, atol=1e-3, rtol=1e-2)
	print(test)
