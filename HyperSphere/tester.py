import torch
from torch.autograd import Variable
import sys
import time
from datetime import datetime
import numpy as np
import scipy as sp
import scipy.linalg as linalg
import torch.multiprocessing
import matplotlib.pyplot as plt


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


def object_comparison():
	from HyperSphere.GP.models.gp_regression import GPRegression
	from HyperSphere.GP.kernels.modules.matern52 import Matern52
	from HyperSphere.GP.kernels.modules.squared_exponential import SquaredExponentialKernel
	from HyperSphere.BO.shadow_inference.inference_slide_both import ShadowInference as si1
	from HyperSphere.BO.shadow_inference.inference_slide_origin import ShadowInference as si2
	from HyperSphere.BO.shadow_inference.inference_slide_origin import ShadowInference as si3
	model1 = GPRegression(Matern52(3))
	model2 = GPRegression(SquaredExponentialKernel(3))
	a = si1((Variable(torch.randn(10, 3)), Variable(torch.randn(10, 3))), model1)
	b = a.__class__((a.train_x, a.train_y), model1)
	print(a.__class__ is b.__class__)
	print(a.__class__)
	print(b.__class__)


def kumaraswamy():
	import matplotlib.pyplot as plt
	n_value = 10
	a_pool = np.linspace(0.5, 5, n_value)
	b_pool = np.linspace(0.5, 5, n_value)
	x = np.linspace(0, 1, 100)
	for i in range(n_value):
		# y = 1 - (1 - x ** 1.0) ** b_pool[i]
		# label = '%.4f' % b_pool[i]
		y = 1 - (1 - x ** a_pool[i]) ** 2.0
		label = '%.4f' % a_pool[i]
		plt.plot(x, y, label=label)
	plt.legend()
	plt.show()


def test_func(i):
	p = torch.multiprocessing.current_process()
	print(p.pid)
	sys.stdout.flush()
	ndim = 500
	A = Variable(torch.randn(ndim, ndim))
	b = Variable(torch.randn(ndim, 4))
	x = torch.gesv(b, A)[0]
	return x


def multiprocessor_test():
	n = 5

	pool = torch.multiprocessing.Pool(n)
	result_list = []
	for i in range(n):
		time.sleep(0.1)
		result_list.append(pool.apply_async(test_func, args=(i,)))
		print('At %s, running %d process' % (datetime.now().strftime('%Y%m%d-%H:%M:%S:%f'), [p.ready() for p in result_list].count(False)))
		sys.stdout.flush()
	print("all processes are running")
	sys.stdout.flush()
	while [p.ready() for p in result_list].count(False) > 0:
		time.sleep(0.1)
		print('At %s, running %d process' % (datetime.now().strftime('%Y%m%d-%H:%M:%S:%f'), [p.ready() for p in result_list].count(False)))
	result = [p.get() for p in result_list]
	for res in result:
		print(res.size())

	# process_list = [torch.multiprocessing.Process(target=test_func, args=(i, )) for i in range(n)]
	# print("Before start n_running : %d" % [p.is_alive() for p in process_list].count(True))
	# sys.stdout.flush()
	# for p in process_list:
	# 	p.start()
	# while [p.is_alive() for p in process_list].count(True) > 0:
	# 	time.sleep(0.1)
	# 	print(datetime.now().strftime('%Y%m%d-%H:%M:%S:%f'))
	# 	sys.stdout.flush()

	print 'Done'


def inversion_time(n_data):
	start_time = time.time()
	A = sp.randn(n_data, n_data)
	print('random generation' + time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
	start_time = time.time()
	B = linalg.inv(A)
	print('matrix inversion' + time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))


def centering_check():
	x = np.linspace(0, 1, 100)
	zeros_list = []
	zeros = []
	for s in x:
		poly_zeros = np.roots([-0.25, 0, 0.75, 0.5 - s])
		zeros_list.append(poly_zeros)
		assert np.sum(np.imag(poly_zeros)) == 0
		smallest_abs = np.argmin(np.abs(poly_zeros))
		zeros.append(poly_zeros[smallest_abs])
	print(zeros_list)
	plt.plot(x, zeros)
	plt.show()


def inverse_with_center(center_probability=0.5):
	if isinstance(center_probability, (float, int)):
		center_probability = torch.zeros(54) + center_probability

	shift = []
	for i in range(center_probability.numel()):
		poly_d = center_probability.squeeze()[i]
		if poly_d == 0:
			shift.append(-1.0)
		elif poly_d == 1:
			shift.append(1.0)
		elif 0 < poly_d < 1:
			poly_zeros = np.roots([-0.25, 0, 0.75, 0.5 - poly_d])
			shift.append(poly_zeros[np.argmin(np.abs(poly_zeros))])
	shift = torch.FloatTensor(shift)

	target = torch.linspace(0, 0.5, 55)[1:]
	target_np = target.numpy()
	zero_list = []
	for i in range(54):
		zeros = np.roots([2, -3, 0, target_np[i]])
		zero = zeros[np.logical_and(zeros >= 0, zeros <= 1)][0] * 2.0 - 1.0 - shift[i]
		zero_list.append(zero)
	zero_list[-1] = 0
	print zero_list

if __name__ == '__main__':
	inverse_with_center()
