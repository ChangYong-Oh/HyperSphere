import torch
from torch.autograd import Variable
import torch.optim


def error_reproduce():
	n_data = 500
	ndim = 1000

	b = Variable(torch.randn(ndim, n_data))
	A = Variable(torch.randn(ndim, ndim))

	A_sym = A.mm(A.t())

	pool = torch.multiprocessing.Pool(1)
	res = pool.apply_async(torch.gesv, args=(b, A))
	pool.close()
	pool.join()
	return res.get()


def no_error_without_blas():
	n_data = 500
	ndim = 1000

	b = Variable(torch.randn(ndim, n_data))
	A = Variable(torch.randn(ndim, ndim))

	pool = torch.multiprocessing.Pool(1)
	res = pool.apply_async(torch.gesv, args=(b, A))
	pool.close()
	pool.join()
	return res.get()


def no_error_by_calling_pool_first():
	n_data = 500
	ndim = 1000

	b = Variable(torch.randn(ndim, n_data))
	A = Variable(torch.randn(ndim, ndim))

	pool = torch.multiprocessing.Pool(1)

	A_sym = A.mm(A.t())

	res = pool.apply_async(torch.gesv, args=(b, A))
	pool.close()
	pool.join()
	return res.get()


if __name__ == '__main__':
	print(no_error_by_calling_pool_first())
