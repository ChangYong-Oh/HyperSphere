import os
import os.path
import sys
import time
import shutil
from datetime import datetime
import argparse
import multiprocessing
import subprocess32
import tempfile
import socket
import smtplib

from utils.datafile_utils import EXPERIMENT_DIR

valid_config_str_list = ['cube', 'cubeard', 'cubeboundary', 'cubeardboundary',
                         'spherenone', 'sphereorigin', 'sphereboundary', 'sphereboth',
                         'spherewarpingnone', 'spherewarpingorigin', 'spherewarpingboundary', 'spherewarpingboth']
valid_func_name_list = ['levy', 'styblinskitang', 'rosenbrock', 'schwefel', 'perm', 'michalewicz']


def argument_generate(config_str):
	assert config_str in valid_config_str_list

	if config_str == 'cube':
		return ' -g cube '
	elif config_str == 'cubeard':
		return ' -g cube --ard '
	elif config_str == 'cubeboundary':
		return ' -g cube --boundary '
	elif config_str == 'cubeardboundary':
		return ' -g cube --ard --boundary '
	elif config_str == 'spherenone':
		return ' -g sphere '
	elif config_str == 'sphereorigin':
		return ' -g sphere --origin '
	elif config_str == 'sphereboundary':
		return ' -g sphere --boundary'
	elif config_str == 'sphereboth':
		return ' -g sphere --origin --boundary '
	elif config_str == 'spherewarpingnone':
		return ' -g sphere --warping '
	elif config_str == 'spherewarpingorigin':
		return ' -g sphere --warping --origin '
	elif config_str == 'spherewarpingboundary':
		return ' -g sphere --warping --boundary '
	elif config_str == 'spherewarpingboth':
		return ' -g sphere --warping --origin --boundary '


def beginning_command_str_generate(current_file, optimizer_config_list, func_name_list, ndim, n_eval):
	cmd_list = []
	cmd = 'python ' + os.path.join(os.path.split(current_file)[0], 'run_BO.py') + ' '
	for optim_config in optimizer_config_list:
		assert optim_config in valid_config_str_list
		for func_name in func_name_list:
			assert func_name in valid_func_name_list
			cmd_list.append(cmd + '-f ' + func_name + ' -d' + str(ndim) + argument_generate(optim_config) + ' -e ' + str(n_eval))
	return cmd_list * 5


def continuing_command_str_generate(current_file, path_list, n_eval):
	cmd_list = []
	cmd = 'python ' + os.path.join(os.path.split(current_file)[0], 'run_BO.py') + ' '
	for path in path_list:
		cmd_list.append(cmd + '-p ' + path + ' -e ' + str(n_eval))
	return cmd_list

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Bayesian Optimization multicore runner')
	parser.add_argument('-e', '--n_eval', dest='n_eval', type=int, default=1)
	parser.add_argument('-d', '--dim', dest='ndim', type=int)
	parser.add_argument('-f', '--func', dest='func_name_list')
	parser.add_argument('-o', '--optimizer', dest='optimizer_config_list')
	parser.add_argument('-p', '--path', dest='path')
	parser.add_argument('--continue', dest='continuing', action='store_true', default=False)

	current_file = sys.argv[0]
	try:
		args = parser.parse_args()
		assert (args.path is None) != ((args.ndim is None) and (args.func_name_list is None) and (args.optimizer_config_list is None))
		if args.path is None:
			assert args.n_eval == 1
			optimizer_config_list = args.optimizer_config_list.split(',')
			func_name_list = args.func_name_list.split(',')
			n_runs = len(optimizer_config_list) * len(func_name_list) * 5
			if args.continuing:
				path_list = []
				for optimizer_config in optimizer_config_list:
					for func_name in func_name_list:
						path_list += [os.path.join(EXPERIMENT_DIR, elm) for elm in os.listdir(EXPERIMENT_DIR) if func_name + '_D' + str(args.ndim) + '_' + optimizer_config in elm]
				for elm in path_list:
					print(elm)
				cmd_str_list = continuing_command_str_generate(current_file, path_list, args.n_eval)
			else:
				cmd_str_list = beginning_command_str_generate(current_file, optimizer_config_list, func_name_list, args.ndim, args.n_eval)
		else:
			path_list = args.path.split(',')
			n_runs = len(path_list) * 5
			cmd_str_list = continuing_command_str_generate(current_file, path_list, args.n_eval)
		if n_runs > int(multiprocessing.cpu_count() / 5) * 5:
			print('Too many runs, reduce the number of optimizer or the number of functions')
			exit()
	except:
		print('Multiple arguments can be given with comma seperation(no other seperator is allowed.)')
		if args.path is None:
			print('Valid optimizer arguments')
			for elm in valid_config_str_list:
				print('    -' + elm)
			print('Valid function arguments')
			for elm in valid_func_name_list:
				print('    -' + elm)
		else:
			print('Invalid path')
		exit()

	process_list = []
	start_time = datetime.now().strftime('%Y%m%d-%H:%M:%S:%f')
	log_file_list = [tempfile.NamedTemporaryFile('w', delete=False) for _ in range(n_runs)]
	for cmd_str, log_file in zip(cmd_str_list, log_file_list):
		process_list.append(subprocess32.Popen(cmd_str.split(), stdout=log_file, stderr=log_file))
	dir_created = [os.path.isdir(elm) for elm in process_list[0].args].count(True) == 1
	exp_dir_list = None
	if dir_created:
		exp_dir_list = [[elm for elm in process_list[i].args if os.path.isdir(elm)][0] for i in range(n_runs)]

	process_status_list = [elm.poll() for elm in process_list]
	previous_process_status_list = process_status_list[:]
	cnt_normal_terimnation = 0
	cnt_abnormal_termination = 0
	sys.stdout.write('Process status check... %s -- Start\n' % datetime.now().strftime('%Y%m%d-%H:%M:%S'))
	while None in process_status_list:
		time.sleep(60)
		sys.stdout.write('\rProcess status check... ' + datetime.now().strftime('%Y%m%d-%H:%M:%S'))
		sys.stdout.flush()
		process_status_list = [elm.poll() for elm in process_list]
		for i, prev_p_status, p_status in zip(range(n_runs), previous_process_status_list, process_status_list):
			if p_status is not None and prev_p_status is None:
				if p_status == 0:
					cnt_normal_terimnation += 1
				else:
					cnt_abnormal_termination += 1
				log_filedir, log_filename = os.path.split(log_file_list[i].name)
				if dir_created:
					moved_filename = os.path.join(exp_dir_list[i], 'log', 'L' + start_time + '_' + datetime.now().strftime('%Y%m%d-%H:%M:%S:%f') + '_'+ log_filename + '.log')
					shutil.move(log_file_list[i].name, moved_filename)
				else:
					moved_filename = log_file_list[i].name
				if p_status == 0:
					sys.stdout.write('\n          Experiment in %s has finished with exit code %d' % (exp_dir_list[i], process_list[i].returncode))
				else:
					sys.stdout.write('\n    !!!!! Experiment in %s has finished with exit code %d !!!!!' % (exp_dir_list[i], process_list[i].returncode))
					sender = 'coh@' + socket.gethostbyaddr(socket.gethostname())[0]
					receiver = 'changyong.oh0224@gmail.com'
					message = "Subject: %2d(+%2d:-%2d)/%2d Terminated(%d) in %s(%s)\n\ncheck file %s" \
					          % (cnt_normal_terimnation + cnt_abnormal_termination, cnt_normal_terimnation, cnt_abnormal_termination, n_runs,
					             process_list[i].returncode, sender.split('@')[1], start_time, moved_filename)
					smtplib.SMTP('localhost').sendmail(sender, receiver, message)
				sys.stdout.write('\n')
		previous_process_status_list = process_status_list[:]
