import multiprocessing
import sys

from HyperSphere.BO.spherewarpingboundary_BO import BO as BO_boundary
from HyperSphere.BO.spherewarpingnone_BO import BO as BO_none
from HyperSphere.BO.spherewarpingorigin_BO import BO as BO_origin

from HyperSphere.BO.spherewarpingboth_BO import BO as BO_both

if __name__ == '__main__':
	optimizer_str = sys.argv[1]
	if optimizer_str == 'spherewarpingnone':
		optimizer = BO_none
	elif optimizer_str == 'spherewarpingorigin':
		optimizer = BO_origin
	elif optimizer_str == 'spherewarpingboundary':
		optimizer = BO_boundary
	elif optimizer_str == 'spherewarpingboth':
		optimizer = BO_both
	n_dim = int(sys.argv[2])
	n_eval = int(sys.argv[3])
	n_func = len(sys.argv) - 4
	func_name_list = sys.argv[4:]
	n_cores = multiprocessing.cpu_count()
	assert n_cores > n_func * 5
	func_list = []
	for i in range(n_func):
		exec('func_list.append(' + func_name_list[i] + ')')

	n_runs = int(n_cores / 5) * 5
	pool_list = []
	for i in range(n_func):
		pool_list.append(multiprocessing.Pool(5))

	for pool, func in zip(pool_list, func_list):
		optimizer_args = {'n_eval': n_eval, '**kwargs': {'func': func, 'dim': n_dim}}
		exp_result_dir = pool.map(optimizer, [optimizer_args] * 5)
		print(exp_result_dir)


