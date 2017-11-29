import torch
from torch.autograd import Variable, grad
import torch.optim as optim
import time


def optimize(max_step, x0, b, A):
	x = Variable(x0.clone(), requires_grad=True)
	optimizer = optim.Adam([x], lr=0.01)
	for s in range(max_step):
		optimizer.zero_grad()
		loss = torch.mean((A.mm(x) - b) ** 2)
		x.grad = grad([loss], [x], retain_graph=True)[0]
		optimizer.step()
	optimum_loc = x.clone()
	optimum_value = torch.sum((A.mm(x) - b) ** 2).data.squeeze()[0]
	return optimum_loc, optimum_value


def error_reproduce_pool():
	ndim = 1000
	n_init = 20

	# Calling pool BEFORE any pytorch linear algebra operation does not cause any problem
	# pool = torch.multiprocessing.Pool(n_init)

	b = Variable(torch.randn(ndim, 1))
	A = Variable(torch.randn(ndim, ndim))
	A = A.mm(A.t()) # pytorch Linear Algebra operation
	x0 = torch.randn(ndim, n_init)

	# Calling pool AFTER any pytorch linear algebra operation causes hanging
	pool = torch.multiprocessing.Pool(n_init)

	result_list = []
	for i in range(n_init):
		result_list.append(pool.apply_async(optimize, args=(100, x0[:, i:i+1], b, A)))
	while [not p.ready() for p in result_list].count(True) > 0:
		time.sleep(1)
	pool.close()
	return [r.get() for r in result_list]


def error_reproduce_process():
	n_data = 500
	ndim = 1000

	b = Variable(torch.randn(ndim, n_data))
	A = Variable(torch.randn(ndim, ndim))

	def gesv_wrapper(return_dict, i, *args):
		return_dict[i] = torch.gesv(*args)[0]

	return_dict = torch.multiprocessing.Manager().dict()

	process = torch.multiprocessing.Process(target=gesv_wrapper, args=(return_dict, 0, b, A))
	process.start()
	process.join()
	return return_dict


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

	res = pool.apply_async(torch.gesv, args=(b, A))
	pool.close()
	pool.join()
	return res.get()


def pool_reuse_check():
	n_batch = 5
	pool = torch.multiprocessing.Pool(n_batch)
	x = range(20)

	for i in range(4):
		result_list = []
		for j in range(n_batch):
			result_list.append(pool.apply_async(test_func, args=(x[n_batch * i + j],)))
		return_values = [res.get() for res in result_list]
		print(return_values)


def test_func(x):
	return x ** 2


if __name__ == '__main__':
	print(pool_reuse_check())
