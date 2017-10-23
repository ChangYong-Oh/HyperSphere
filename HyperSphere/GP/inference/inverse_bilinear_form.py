import torch
from torch.autograd import Function, Variable, gradcheck


class InverseBilinearForm(Function):

	@staticmethod
	def forward(ctx, vec_left, matrix, vec_right=None):
		if vec_right is None:
			vec_right = vec_left

		ctx.save_for_backward(vec_left, matrix, vec_right)

		jitter = 0
		eye_mat = torch.diag(matrix[:1, :1].clone().repeat(matrix.size(1), 1).view(-1) * 0 + 1)
		while True:
			try:
				vec_right_sol = torch.gesv(vec_right, matrix + eye_mat * jitter)[0]
				break
			except RuntimeError:
				jitter = torch.mean(torch.diag(matrix)) * 1e-6 if jitter == 0 else jitter * 10

		return torch.mm(vec_left.t(), vec_right_sol)

	@staticmethod
	def backward(ctx, grad_output):
		"""
		:param ctx: 
		:param grad_output: grad_output is assumed to be d[Scalar]/dK and size() is n1 x n2 
		:return: 
		"""
		vec_left, matrix, vec_right = ctx.saved_variables
		grad_vec_left = grad_matrix = grad_vec_right = None

		linear_solver = torch.gesv(torch.cat([vec_left, vec_right], 1), matrix)[0]
		vec_left_sol = linear_solver[:, :vec_left.size(1)]
		vec_right_sol = linear_solver[:, vec_left.size(1):]
		if ctx.needs_input_grad[0]:
			grad_vec_left = grad_output * vec_right_sol
		if ctx.needs_input_grad[1]:
			grad_matrix = grad_output * -torch.mm(vec_left_sol, vec_right_sol.t())
		if ctx.needs_input_grad[2]:
			grad_vec_right = grad_output * vec_left_sol

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
