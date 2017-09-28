import torch
from torch.autograd import Variable
import time
import numpy as np
import scipy.linalg as linalg


def test_speed_inverse_gesv(ndim=10):
	A = torch.randn(ndim, ndim)
	A = A.mm(A.t())
	eig, _ = torch.symeig(A)
	print(torch.min(eig))
	matrix = Variable(A + 1e0 * torch.eye(ndim), requires_grad=True)
	n_rep = 10000

	inv1 = None
	start_time = time.time()
	for _ in range(n_rep):
		inv1, _ = torch.gesv(torch.diag(Variable(matrix.data.new(matrix.size(0)).fill_(1))), matrix)
	print(time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time)))

	inv2 = None
	start_time = time.time()
	for _ in range(n_rep):
		inv2 = matrix.inverse()
	print(time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time)))

	print(torch.sum((inv1.data - inv2.data)**2))


def test_speed_scipy_torch(ndim=10):
	A = torch.randn(ndim, ndim)
	A = A.mm(A.t())
	eig, _ = torch.symeig(A)
	print(torch.min(eig))
	matrix = Variable(A + 1e0 * torch.eye(ndim), requires_grad=True)
	n_rep = 10000

	start_time = time.time()
	for _ in range(n_rep):
		_, lu = torch.gesv(Variable(matrix.data.new(ndim).fill_(1)).view(-1), matrix)
	print(time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time)))
	print(torch.prod(torch.diag(lu)))

	start_time = time.time()
	for _ in range(n_rep):
		_, _, u = linalg.lu(matrix.data.numpy())
	print(time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time)))
	print(np.prod(np.diag(u)))


def test_speed_gesv_potrf(ndim=10):
	A = torch.randn(ndim, ndim)
	A = A.mm(A.t())
	A += 1e-6 * torch.eye(ndim)
	eig, _ = torch.symeig(A)
	if torch.min(eig) < 0:
		A += (1e-6 - eig) * torch.eye(ndim)
	eig, _ = torch.symeig(A)
	print(torch.min(eig))
	print(torch.sum(torch.log(eig)))
	matrix = A
	n_rep = 100

	start_time = time.time()
	for _ in range(n_rep):
		_, lu = torch.gesv(matrix.new(ndim).fill_(1).view(-1), matrix)
		result = torch.sum(torch.log(torch.diag(lu)))
	print('GESV')
	print(time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time)))
	print(result)
	diag = torch.diag(lu)
	print(torch.sum(torch.log(diag[diag > 0])))
	print(torch.min(torch.diag(lu)), torch.max(torch.diag(lu)))

	start_time = time.time()
	for _ in range(n_rep):
		chol_from_upper = torch.potrf(matrix, True)
		chol_from_lower = torch.potrf(matrix, False)
		result = (torch.sum(torch.log(torch.diag(chol_from_upper)), 0, keepdim=True) + torch.sum(torch.log(torch.diag(chol_from_lower)), 0, keepdim=True)).view(1, 1)
	print('POTRF')
	print(time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time)))
	print(result)
	print(torch.min(torch.diag(chol_from_upper)), torch.max(torch.diag(chol_from_upper)))
	print(torch.min(torch.diag(chol_from_lower)), torch.max(torch.diag(chol_from_lower)))

if __name__ == '__main__':
	test_speed_gesv_potrf(200)