import os
import os.path
import sys
import time
from datetime import datetime
import argparse
import multiprocessing
import subprocess
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
		process_list.append(subprocess.Popen(cmd_str.split(), stdout=log_file, stderr=log_file))
	log_filename_list = [os.path.split(elm.name) for elm in log_file_list]
	log_filename_list = [os.path.join(elm[0], start_time + str(i).zfill(2) + '-' + elm[1] + '.log') for (i, elm) in enumerate(log_filename_list)]

	sender = 'coh@' + socket.gethostbyaddr(socket.gethostname())[0]
	receiver = 'changyong.oh0224@gmail.com'
	smtpObj = smtplib.SMTP('localhost')
	process_status_list = [elm.poll() for elm in process_list]
	previous_process_status_list = process_status_list[:]
	cnt = 0
	while None in process_status_list:
		time.sleep(60)
		print('process status checking...' + datetime.now().strftime('%Y%m%d-%H:%M:%S'))
		sys.stdout.flush()
		process_status_list = [elm.poll() for elm in process_list]
		for i, prev_p_status, p_status in zip(range(n_runs), previous_process_status_list, process_status_list):
			if p_status is not None and prev_p_status is None:
				cnt += 1
				message = "Subject: %2d/%2d Terminated exit code(%d) in %s\n\ncheck file %s" % (cnt, n_runs, process_list[i].returncode, sender.split('@')[1], log_filename_list[i])
				try:
					smtpObj.sendmail(sender, receiver, message)
				except:
					print(message)
					sys.stdout.flush()
		previous_process_status_list = process_status_list[:]
