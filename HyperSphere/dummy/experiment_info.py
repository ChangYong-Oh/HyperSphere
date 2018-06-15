import os


def how_many_evaluations():
	exp_dir = os.path.join('/'.join(os.path.realpath(__file__).split('/')[:-3]), 'Experiments', 'Hypersphere')
	exp_list = os.listdir(exp_dir)
	for exp in sorted(exp_list):
		eval_logs = os.listdir(os.path.join(exp_dir, exp, 'log'))
		if len(eval_logs) > 0:
			print(exp, int(max(eval_logs).split('.')[0]))
		else:
			print(exp, 0)


if __name__ == '__main__':
	how_many_evaluations()