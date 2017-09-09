import torch
from torch.autograd import Function, Variable, gradcheck


class InverseBilinearForm(Function):

	@staticmethod
	def forward(ctx, vec_left, matrix, vec_right=None):
		if vec_right is None:
			vec_right = vec_left

		ctx.save_for_backward(vec_left, matrix, vec_right)

		return torch.mm(vec_left.t(), matrix.inverse()).mm(vec_right)

	@staticmethod
	def backward(ctx, grad_output):
		"""
		:param ctx: 
		:param grad_output: grad_output is assumed to be d[Scalar]/dK and size() is n1 x n2 
		:return: 
		"""
		vec_left, matrix, vec_right = ctx.saved_variables
		grad_vec_left = grad_matrix = grad_vec_right = None

		inverse_matrix = matrix.inverse()
		if ctx.needs_input_grad[0]:
			grad_vec_left = grad_output * torch.mm(inverse_matrix, vec_right)
		if ctx.needs_input_grad[1]:
			grad_matrix = grad_output * -torch.mm(inverse_matrix, vec_left).mm(vec_right.t()).mm(inverse_matrix)
		if ctx.needs_input_grad[2]:
			grad_vec_right = grad_output * torch.mm(inverse_matrix, vec_left)

		return grad_vec_left, grad_matrix, grad_vec_right


if __name__ == '__main__':
	ndim = 5
	vec_left = Variable(torch.randn(ndim, 1), requires_grad=True)
	A = torch.randn(ndim, ndim)
	matrix = Variable(A.mm(A.t()) + torch.eye(ndim), requires_grad=True)
	vec_right = Variable(torch.randn(ndim, 1), requires_grad=True)

	# gradcheck doesn't have to pass all the time.
	test = gradcheck(InverseBilinearForm.apply, (vec_left, matrix, vec_right), eps=1e-4, atol=1e-3, rtol=1e-2)
	print(test)
