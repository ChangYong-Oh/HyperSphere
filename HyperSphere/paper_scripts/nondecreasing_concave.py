import os
import sys
import pickle
import time
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt


def warping_sample():
	x = np.linspace(0, 1, 100)
	y = np.log(50 * x + 1)
	expand_begin = 3
	expand_end = 13
	shrink_begin = 75
	shrink_end = 90
	x_max = np.max(x)
	y_max = np.max(y)
	y = y / y_max
	plt.xlim(0, 1)
	plt.ylim(0, 1)
	plt.plot(x, y)
	plt.axhline(y[expand_begin], xmin=0, xmax=x[expand_begin], color='g', ls='--')
	plt.axhline(y[expand_end], xmin=0, xmax=x[expand_end], color='g', ls='--')
	plt.axvline(x[expand_begin], ymin=0, ymax=y[expand_begin], color='g', ls='--')
	plt.axvline(x[expand_end], ymin=0, ymax=y[expand_end], color='g', ls='--')

	plt.axhline(y[shrink_begin], xmin=0, xmax=x[shrink_begin], color='y', ls='--')
	plt.axhline(y[shrink_end], xmin=0, xmax=x[shrink_end], color='y', ls='--')
	plt.axvline(x[shrink_begin], ymin=0, ymax=y[shrink_begin], color='y', ls='--')
	plt.axvline(x[shrink_end], ymin=0, ymax=y[shrink_end], color='y', ls='--')
	plt.tight_layout()
	plt.show()


def kumaraswamy_cdf():
	x = np.linspace(0, 1, 100)
	a_list = [1.0/4, 1.0/2, 1.0]
	b_list = [1, 2, 4]
	for a in a_list:
		for b in b_list:
			plt.plot(x, (1 - (1 - x ** a) ** b), label=('a=%.2f' % a) + ',b=' + str(b))
	plt.legend()
	plt.tight_layout()
	plt.show()


if __name__ == '__main__':
	kumaraswamy_cdf()