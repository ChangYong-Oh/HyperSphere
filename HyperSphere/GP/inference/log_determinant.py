import numpy as np

import torch
from torch.autograd import Function, Variable, gradcheck


class LogDeterminant(Function):

	@staticmethod
	def forward(ctx, matrix):
		ctx.save_for_backward(matrix)

		try:
			chol_from_upper = torch.potrf(matrix, True)
			chol_from_lower = torch.potrf(matrix, False)
			return (torch.sum(torch.log(torch.diag(chol_from_upper)), 0, keepdim=True) + torch.sum(torch.log(torch.diag(chol_from_lower)), 0, keepdim=True)).view(1, 1)
		except RuntimeError:
			eigvals = torch.symeig(matrix)[0]
			return torch.sum(torch.log(eigvals[eigvals > 0]), 0, keepdim=True)

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
			logdet_grad = torch.gesv(torch.diag(Variable(matrix.data.new(matrix.size(0)).fill_(1))), matrix)[0]
			grad_matrix = grad_output * logdet_grad

		return grad_matrix


if __name__ == '__main__':
	ndim = 10
	A = torch.randn(ndim, ndim)
	A = A.mm(A.t())
	eig, _ = torch.symeig(A)
	print(torch.min(eig))
	matrix = Variable(A + (1e-4 - torch.min(eig)) * torch.eye(ndim), requires_grad=True)
	_, logdet = np.linalg.slogdet(matrix.data.numpy())

	eps = 1e-4

	fdm_deriv = torch.zeros(ndim, ndim)
	for i in range(ndim):
		for j in range(ndim):
			matrix_perturb = Variable(matrix.data).clone()
			matrix_perturb.data[i, j] += eps
			_, logdet_perturb = np.linalg.slogdet(matrix_perturb.data.numpy())
			fdm_deriv[i, j] = (logdet_perturb - logdet) / eps

	output = LogDeterminant.apply(matrix)
	output.backward()
	print(torch.abs((matrix.grad.data - fdm_deriv)/matrix.grad.data))


	# gradcheck doesn't have to pass all the time.
	test = gradcheck(LogDeterminant.apply, (matrix, ), eps=eps, atol=1e-3, rtol=1e-2)
	print(test)
