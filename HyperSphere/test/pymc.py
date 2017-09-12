import torch
import numpy as np
import pymc3 as pm
from HyperSphere.GP.models.gp_regression import GPRegression
from HyperSphere.GP.kernels.modules.squared_exponential import SquaredExponentialKernel


if __name__ == '__main__':
	gp = GPRegression(SquaredExponentialKernel(ndim=5))
	with pm.Model() as model:
		for key, value in gp.parameter_prior():
			vars()[key] = value

	a = 1 + 2
