import multiprocessing
import sys
from functools import partial

from HyperSphere.BO.spherewarpingboundary_BO import BO as BO_boundary
from HyperSphere.BO.spherewarpingnone_BO import BO as BO_none
from HyperSphere.BO.spherewarpingorigin_BO import BO as BO_origin

from HyperSphere.BO.axv.spherewarpingboth_BO import BO as BO_both

if __name__ == '__main__':
	optimizer_str = sys.argv[1]
	optimizer_name_list = optimizer_str.split(',')
	optimizer_list = []
	for optimizer_name in optimizer_name_list:
		if optimizer_name == 'spherewarpingnone':
			optimizer = BO_none
		elif optimizer_name == 'spherewarpingorigin':
			optimizer = BO_origin
		elif optimizer_name == 'spherewarpingboundary':
			optimizer = BO_boundary
		elif optimizer_name == 'spherewarpingboth':
			optimizer = BO_both
		optimizer_list.append(optimizer)
	n_dim = int(sys.argv[2])
	n_eval = int(sys.argv[3])
	func_str = sys.argv[4]
	func_name_list = sys.argv[4].split(',')
	n_optimizer = len(optimizer_list)
	n_func = len(func_name_list)
	n_cores = multiprocessing.cpu_count()
	assert n_cores > n_func * n_optimizer * 5
	func_list = []
	for i in range(n_func):
		exec('func_list.append(' + func_name_list[i] + ')')

	pool_dict = {}
	for optimizer in optimizer_list:
		for func in func_list:
			pool_dict[(optimizer, func)] = multiprocessing.Pool(5)
			exp_result_dir = pool_dict[(optimizer, func)].map(partial(optimizer, kwargs={'func': func, 'dim': n_dim}), [n_eval] * 5)
			print(exp_result_dir)



