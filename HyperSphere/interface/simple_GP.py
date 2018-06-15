#!/usr/bin/env python
# Created by Tijmen Blankevoort 2018 | tijmen@qti.qualcomm.com

#  ============================================================================
#
#  @@-COPYRIGHT-START-@@
#
#  Copyright 2018 Qualcomm Technologies, Inc. All rights reserved.
#  Confidential & Proprietary - Qualcomm Technologies, Inc. ("QTI")
#
#  The party receiving this software directly from QTI (the "Recipient")
#  may use this software as reasonably necessary solely for the purposes
#  set forth in the agreement between the Recipient and QTI (the
#  "Agreement"). The software may be used in source code form solely by
#  the Recipient's employees (if any) authorized by the Agreement. Unless
#  expressly authorized in the Agreement, the Recipient may not sublicense,
#  assign, transfer or otherwise provide the source code to any third
#  party. Qualcomm Technologies, Inc. retains all ownership rights in and
#  to the software
#
#  This notice supersedes any other QTI notices contained within the software
#  except copyright notices indicating different years of publication for
#  different portions of the software. This notice does not supersede the
#  application of any third party copyright notice to that third party's
#  code.
#
#  @@-COPYRIGHT-END-@@
#
#  ============================================================================

import numpy as np

from HyperSphere.interface.hyperparameter_search_method import HyperParameterSearchMethod
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.optimize import minimize
from scipy.stats import norm


class SimpleGP(HyperParameterSearchMethod):
    def __init__(self, *args):
        super(SimpleGP, self).__init__(*args)
        self.gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            n_restarts_optimizer=25,
            alpha=1e-4,
        )

    def make_X_Y(self):
        X = np.vstack([result[0] for result in self.result_list])
        Y = np.vstack([result[1] for result in self.result_list])
        return X, Y

    def expected_improvement(self, x, gaussian_process, evaluated_loss, n_params=1):
        """ expected_improvement
        Expected improvement acquisition function.
        Arguments:
        ----------
            x: array-like, shape = [n_samples, n_hyperparams]
                The point for which the expected improvement needs to be computed.
            gaussian_process: GaussianProcessRegressor object.
                Gaussian process trained on previously evaluated hyperparameters.
            evaluated_loss: Numpy array.
                Numpy array that contains the values off the loss function for the previously
                evaluated hyperparameters.
            greater_is_better: Boolean.
                Boolean flag that indicates whether the loss function is to be maximised or minimised.
            n_params: int.
                Dimension of the hyperparameter space.
        """

        x_to_predict = x.reshape(-1, n_params)

        mu, sigma = gaussian_process.predict(x_to_predict, return_std=True)

        loss_optimum = np.min(evaluated_loss)

        scaling_factor = -1

        # In case sigma equals zero
        with np.errstate(divide='ignore'):
            Z = scaling_factor * (mu - loss_optimum) / sigma
            expected_improvement = scaling_factor * (mu - loss_optimum) * norm.cdf(Z) + sigma * norm.pdf(Z)
            expected_improvement[sigma == 0.0] == 0.0

        return -1 * expected_improvement


    def sample_next_hyperparameter(self,
                                   evaluated_loss,
                                   bounds=(0, 10), n_restarts=25):
        """ sample_next_hyperparameter
        Proposes the next hyperparameter to sample the loss function for.
        Arguments:
        ----------
            acquisition_func: function.
                Acquisition function to optimise.
            gaussian_process: GaussianProcessRegressor object.
                Gaussian process trained on previously evaluated hyperparameters.
            evaluated_loss: array-like, shape = [n_obs,]
                Numpy array that contains the values off the loss function for the previously
                evaluated hyperparameters.
            greater_is_better: Boolean.
                Boolean flag that indicates whether the loss function is to be maximised or minimised.
            bounds: Tuple.
                Bounds for the L-BFGS optimiser.
            n_restarts: integer.
                Number of times to run the minimiser with different starting points.
        """
        best_x = None
        best_acquisition_value = 1
        n_params = bounds.shape[0]

        for starting_point in np.random.uniform(bounds[:, 0], bounds[:, 1],
                                                size=(n_restarts, n_params)):

            res = minimize(fun=self.expected_improvement,
                           x0=starting_point.reshape(1, -1),
                           bounds=bounds,
                           method='L-BFGS-B',
                           args=(self.gp, evaluated_loss,n_params))

            if res.fun < best_acquisition_value:
                best_acquisition_value = res.fun
                best_x = res.x

        return best_x

    def get_new_setting(self):
        X, Y = self.make_X_Y()
        self.gp.fit(X, Y)

        next_sample = self.sample_next_hyperparameter(Y, bounds=np.array(self.ranges), n_restarts=100)

        return next_sample