import torch
from torch.autograd import Variable
import torch.optim


def reproduce():
	n_data = 100
	ndim = 100

	b = Variable(torch.randn(ndim, n_data))
	A = Variable(torch.randn(ndim, ndim))

	pool = torch.multiprocessing.Pool(5)

	A_sym = A.mm(A.t())

	res = pool.apply_async(torch.gesv, args=(b, A))
	pool.close()
	pool.join()
	return res.get()


if __name__ == '__main__':
	print(reproduce())
