import os
import sys
import pickle
import time
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

SUGGESTION = -0.39

def preditive_distribution_3d_plot(x, train_x, train_y, pred_mean, pred_std, which_x):
	z_coef = 1.96
	max_y = 2.6
	min_y = -1.55
	print(min_y, max_y)
	y = np.linspace(min_y - 0.05 * (max_y - min_y), max_y + 0.05 * (max_y - min_y), x.size)
	plot_data_z = np.empty((y.size, 0))
	mean_data_z = np.empty((1, 0))
	lower_data_z = np.empty((1, 0))
	upper_data_z = np.empty((1, 0))
	for i in range(x.size):
		plot_data_z = np.hstack((plot_data_z, stats.norm.pdf(y, loc=pred_mean[i], scale=pred_std[i]).reshape((y.size, 1))))
		mean_data_z = np.hstack((mean_data_z, stats.norm.pdf(pred_mean[i], loc=pred_mean[i], scale=pred_std[i]).reshape((1, 1))))
		lower_data_z = np.hstack((lower_data_z, stats.norm.pdf(pred_mean[i] - z_coef * pred_std[i], loc=pred_mean[i], scale=pred_std[i]).reshape((1, 1))))
		upper_data_z = np.hstack((upper_data_z, stats.norm.pdf(pred_mean[i] + z_coef * pred_std[i], loc=pred_mean[i], scale=pred_std[i]).reshape((1, 1))))
	plot_data_x, plot_data_y = np.meshgrid(x, y)

	fig = plt.figure()
	ax = fig.add_subplot(211)
	ax.set_xlim([np.min(x), np.max(x)])
	ax.set_ylim([min_y, max_y])
	ax.set_xlabel('x', fontsize=12, color='b')
	ax.xaxis.set_label_coords(1.01, 0.03)
	ax.set_ylabel('f(x)', fontsize=12, color='b', rotation=0)
	ax.yaxis.set_label_coords(-0.025, 0.9)
	plt.tight_layout()
	plt.show()

	fig = plt.figure()
	ax = fig.add_subplot(211)
	ax.set_xlim([np.min(x), np.max(x)])
	ax.set_ylim([min_y, max_y])
	for i in range(train_y.size-1):
		ax.plot(train_x[i], train_y[i], 'k+', ms=12, mew=4)
	ax.plot(train_x[train_y.size-1], train_y[train_y.size-1], 'k+', ms=12, mew=4, label='Traning Data')
	ax.set_xlabel('x', fontsize=12, color='b')
	ax.xaxis.set_label_coords(1.01, 0.03)
	ax.set_ylabel('f(x)', fontsize=12, color='b', rotation=0)
	ax.yaxis.set_label_coords(-0.025, 0.9)
	plt.tight_layout()
	ax.legend(loc=2)
	plt.show()

	fig = plt.figure()
	ax = fig.add_subplot(211)
	ax.set_xlim([np.min(x), np.max(x)])
	ax.set_ylim([min_y, max_y])
	ax.plot(x, pred_mean, color='b', label='predictive mean')
	ax.fill_between(x, pred_mean - z_coef * pred_std, pred_mean + z_coef * pred_std, color='g', alpha=0.5, label='predictive std')
	for i in range(train_y.size-1):
		ax.plot(train_x[i], train_y[i], 'k+', ms=12, mew=4)
	ax.plot(train_x[train_y.size-1], train_y[train_y.size-1], 'k+', ms=12, mew=4, label='Traning Data')
	ax.legend(loc=2)
	ax.set_xlabel('x', fontsize=12, color='b')
	ax.xaxis.set_label_coords(1.01, 0.03)
	ax.set_ylabel('f(x)', fontsize=12, color='b', rotation=0)
	ax.yaxis.set_label_coords(-0.025, 0.9)
	plt.tight_layout()
	plt.show()

	fig = plt.figure()
	ax = fig.add_subplot(211)
	ax.set_xlim([np.min(x), np.max(x)])
	ax.set_ylim([min_y, max_y])
	ax.plot(x, pred_mean, color='b', label='predictive mean')
	ax.fill_between(x, pred_mean - z_coef * pred_std, pred_mean + z_coef * pred_std, color='g', alpha=0.5, label='predictive std')
	for i in range(train_y.size-1):
		ax.plot(train_x[i], train_y[i], 'k+', ms=12, mew=4)
	ax.plot(train_x[train_y.size-1], train_y[train_y.size-1], 'k+', ms=12, mew=4, label='Traning Data')
	ax.set_xlabel('x', fontsize=12, color='b')
	ax.xaxis.set_label_coords(1.01, 0.03)
	ax.set_ylabel('f(x)', fontsize=12, color='b', rotation=0)
	ax.yaxis.set_label_coords(-0.025, 0.9)
	ax.legend(loc=2)
	acquisition = expected_improvement(pred_mean, pred_std, np.min(train_y))
	print('Acquisition function Maximum : %.2f' % x[np.argmax(acquisition)])
	ax_sub = fig.add_subplot(212)
	ax_sub.set_xlim([np.min(x), np.max(x)])
	ax_sub.plot(x, acquisition, color='c')
	ax_sub.set_xlabel('x', fontsize=12, color='b')
	ax_sub.xaxis.set_label_coords(1.01, 0.03)
	ax_sub.set_ylabel('a(x)', fontsize=12, color='b', rotation=0)
	ax_sub.yaxis.set_label_coords(-0.03, 0.91)
	ax_sub.set_yticks([0, 0.1, 0.2])
	plt.tight_layout()
	plt.show()

	fig = plt.figure()
	ax = fig.add_subplot(211)
	ax.set_xlim([np.min(x), np.max(x)])
	ax.set_ylim([min_y, max_y])
	ax.plot(x, pred_mean, color='b', label='predictive mean')
	ax.fill_between(x, pred_mean - z_coef * pred_std, pred_mean + z_coef * pred_std, color='g', alpha=0.5, label='predictive std')
	for i in range(train_y.size-1):
		ax.plot(train_x[i], train_y[i], 'k+', ms=12, mew=4)
	ax.plot(train_x[train_y.size-1], train_y[train_y.size-1], 'k+', ms=12, mew=4, label='Traning Data')
	ax.set_xlabel('x', fontsize=12, color='b')
	ax.xaxis.set_label_coords(1.01, 0.03)
	ax.set_ylabel('f(x)', fontsize=12, color='b', rotation=0)
	ax.yaxis.set_label_coords(-0.025, 0.9)
	ax.legend(loc=2)
	acquisition = expected_improvement(pred_mean, pred_std, np.min(train_y))
	ax_sub = fig.add_subplot(212)
	ax_sub.set_xlim([np.min(x), np.max(x)])
	ax_sub.plot(x, acquisition, color='c')
	ax_sub.set_xlabel('x', fontsize=12, color='b')
	ax_sub.xaxis.set_label_coords(1.01, 0.03)
	ax_sub.set_ylabel('a(x)', fontsize=12, color='b', rotation=0)
	ax_sub.yaxis.set_label_coords(-0.03, 0.91)
	ax_sub.set_yticks([0, 0.1, 0.2])
	ax_sub.axvline(x[np.argmax(acquisition)], ls='--', color='k')
	plt.tight_layout()
	plt.show()

	fig = plt.figure()
	ax = fig.add_subplot(211)
	ax.set_xlim([np.min(x), np.max(x)])
	ax.set_ylim([min_y, max_y])
	ax.plot(x, pred_mean, color='b', label='predictive mean')
	ax.fill_between(x, pred_mean - z_coef * pred_std, pred_mean + z_coef * pred_std, color='g', alpha=0.5, label='predictive std')
	for i in range(train_y.size - 1):
		ax.plot(train_x[i], train_y[i], 'k+', ms=12, mew=4)
	ax.plot(train_x[train_y.size - 1], train_y[train_y.size - 1], 'k+', ms=12, mew=4, label='Traning Data')
	ax.plot(SUGGESTION, min_y, 'x', ms=12, mew=4, color='m', label='Next evaluation point')
	ax.set_xlabel('x', fontsize=12, color='b')
	ax.xaxis.set_label_coords(1.01, 0.03)
	ax.set_ylabel('f(x)', fontsize=12, color='b', rotation=0)
	ax.yaxis.set_label_coords(-0.025, 0.9)
	ax.legend(loc=2)
	plt.tight_layout()
	plt.show()

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.plot(x, pred_mean, mean_data_z.flatten(), color='b')
	ax.plot(x, pred_mean - z_coef * pred_std, lower_data_z.flatten(), color='g', alpha=0.5)
	ax.plot(x, pred_mean + z_coef * pred_std, upper_data_z.flatten(), color='g', alpha=0.5)
	ax.plot_wireframe(plot_data_x, plot_data_y, plot_data_z, rstride=2, cstride=2, color='k', alpha=0.25)
	ax.set_xlabel('x', fontsize=12, color='b')
	ax.set_ylabel('f(x)', fontsize=12, color='b')
	ax.set_zlabel('Probability Density p(f|x)', fontsize=12, color='b')
	plt.tight_layout()
	plt.show()

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.plot(x, pred_mean, mean_data_z.flatten(), color='b')
	ax.plot(x, pred_mean - z_coef * pred_std, lower_data_z.flatten(), color='g', alpha=0.5)
	ax.plot(x, pred_mean + z_coef * pred_std, upper_data_z.flatten(), color='g', alpha=0.5)
	ax.plot_wireframe(plot_data_x, plot_data_y, plot_data_z, rstride=2, cstride=2, color='k', alpha=0.25)
	ax.add_collection3d(Poly3DCollection([zip(list(y * 0 + x[which_x]), list(y), list(plot_data_z[:, which_x]))], color='cyan'))
	ax.set_xlabel('x', fontsize=12, color='b')
	ax.set_ylabel('f(x)', fontsize=12, color='b')
	ax.set_zlabel('Probability Density p(f|x)', fontsize=12, color='b')
	plt.tight_layout()
	plt.show()

	min_train_y = np.min(train_y)

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(y, plot_data_z[:, which_x], color='cyan')
	ax.axvline(pred_mean[which_x], label=u"\u03BC" + '(x=' + ('%.2f' % x[which_x]) + '|D)', color='b')
	ax.axhline(0.2, xmin=0.29, xmax=0.545, label=u"\u03C3\u00B2" + '(x=' + ('%.2f' % x[which_x]) + '|D)', color='g')
	ax.axvline(min_train_y, label='f_min', color='k', ls='--')
	ax.fill_between(y[y <= min_train_y], y[y <= min_train_y] * 0, plot_data_z[y <= min_train_y, which_x], color='m')
	ax.set_xlabel('f(x=' + ('%.2f' % x[which_x]) + ')', fontsize=12)
	ax.set_ylabel('Density', fontsize=12)
	plt.tight_layout()
	plt.legend()
	plt.show()

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(y, plot_data_z[:, which_x], color='b', label='Original pdf')
	pdf_lower_mean = stats.norm.pdf(y, loc=pred_mean[which_x] - 0.5, scale=pred_std[which_x])
	pdf_higher_std = stats.norm.pdf(y, loc=pred_mean[which_x], scale=pred_std[which_x] * 2.0)
	ax.plot(y, pdf_lower_mean, label='Lower mean pdf', color='g')
	ax.plot(y, pdf_higher_std, label='Higher std pdf', color='y')
	ax.axvline(min_train_y, label='f_min', color='k', ls='--')
	ax.fill_between(y[y <= min_train_y], y[y <= min_train_y] * 0, plot_data_z[y <= min_train_y, which_x], color='m')
	ax.fill_between(y[y <= min_train_y], y[y <= min_train_y] * 0, pdf_lower_mean[y <= min_train_y], color='g', alpha=0.5)
	ax.fill_between(y[y <= min_train_y], y[y <= min_train_y] * 0, pdf_higher_std[y <= min_train_y], color='y', alpha=0.5)
	ax.set_xlabel('f(x=' + ('%.2f' % x[which_x]) + ')', fontsize=12)
	ax.set_ylabel('Density', fontsize=12)
	plt.tight_layout()
	plt.legend()
	plt.show()


def expected_improvement(mean, std, reference):
	standardized = (-mean + reference) / std
	acquisition = std * stats.norm.pdf(standardized) + (-mean + reference) * stats.norm.cdf(standardized)
	return acquisition


def predictive_distribution(x, train_x, train_y):
	obs_var = 0.1
	amp = 1.0
	ls = 0.5
	gram_mat_inv = np.linalg.inv(amp * np.exp(-(train_x.reshape(train_x.size, 1) - train_x.reshape(1, train_x.size)) ** 2 / ls ** 2) + obs_var * np.eye(train_x.size))
	k_star_D = amp * np.exp(-(np.repeat(x.reshape(x.size, 1), train_x.size, axis=1) - np.repeat(train_x.reshape(1, train_x.size), x.size, axis=0)) ** 2 / ls ** 2)
	k_x_x = amp
	pred_mean = np.dot(k_star_D, gram_mat_inv).dot(train_y.reshape(train_y.size, 1) - np.mean(train_y)) + np.mean(train_y)
	pred_var = k_x_x - np.sum(np.dot(k_star_D, gram_mat_inv) * k_star_D, 1, keepdims=True)
	return pred_mean.flatten(), pred_var.flatten()


def optimizee(train_x):
	return np.exp(train_x) * np.sin(train_x * np.pi)


if __name__ == '__main__':
	x = np.linspace(-1, 1, 100)
	which_x = 50
	# train_x = np.random.uniform(-1, 1, 3)
	train_x = np.array([-0.7, -0.1, 0.4])
	# train_x = np.array([-0.7, -0.1, 0.4, SUGGESTION])
	train_y = optimizee(train_x)
	pred_mean, pred_var = predictive_distribution(x, train_x, train_y)
	pred_std = pred_var ** 0.5

	preditive_distribution_3d_plot(x, train_x, train_y, pred_mean, pred_std, which_x)
	#expected_improvement_plot(x, pred_mean, pred_std, which_x, np.min(train_y))

