
import torch
import torch.multiprocessing as multiprocessing


from HyperSphere.GP.models.gp_regression import GPRegression
from HyperSphere.GP.kernels.modules.matern52 import Matern52
from HyperSphere.GP.kernels.modules.radialization import RadializationKernel
from HyperSphere.GP.inference.inference import Inference
from HyperSphere.BO.shadow_inference.inference_sphere_origin import ShadowInference as origin_ShadowInference
from HyperSphere.BO.acquisition.acquisition_functions import expected_improvement
from HyperSphere.BO.acquisition.acquisition_maximization import suggest, optimization_candidates, optimization_init_points, deepcopy_inference
from HyperSphere.feature_map.modules.kumaraswamy import Kumaraswamy
from HyperSphere.feature_map.functionals import sphere_bound


class BlackBoxFunctionOptimization():
    def __init__(self, *args):
        pass

    def next_suggestion(self, input_configuration, output_evaluation):
        raise NotImplementedError('This is a base method, not implemented')


class BayesianOptimization(BlackBoxFunctionOptimization):
    def __init__(self, *args):
        super(BayesianOptimization, self).__init__(*args)
        self.surrogate_model = None
        self.surrogate_inference = None
        self.acquisition_function = None

    def restore_surrogate_model(self, model_path):
        raise NotImplementedError('This is not implemented in General BayesianOptimization Class')


class SpearmintTypeBayesianOptimization(BayesianOptimization):
    def __init__(self, original2cube=None, cube2original=None):
        """
        Sometimes original search space is not hypercube [-1, 1]^D
        You can specify this and its inverse transformation
        :param original2cube: transformation from original search space to hypercube [-1, 1]^D
        :param cube2original: transformation from hypercube [-1, 1]^D to original search space
        """
        super(SpearmintTypeBayesianOptimization, self).__init__()
        self.original2cube = lambda x: x if original2cube is None else original2cube
        self.cube2original = lambda x: x if cube2original is None else cube2original
        n_acq_optim_cand = 20
        self.pool = multiprocessing.Pool(n_acq_optim_cand)

    def save_model(self, model_path):
        """
        In Spearmint, surrogate model is trained using MCMC sampling, for stability it keeps track of samples
        this is the information that is requires to be saved.
        :param model_path:
        """
        torch.save(self.surrogate_model, model_path)

    def restore_model(self, model_path):
        """
        In Spearmint, surrogate model is trained using MCMC sampling, for stability it keeps track of samples
        this is the information that is requires to be retrieved from the file
        :param model_path:
        """
        self.surrogate_model = torch.load(model_path)

    def next_suggestion(self, input_configuration, output_evaluation):
        """
        According to the information you need, you can change return value
        :param input_configuration: n by p tensor
        :param output_evaluation: n by 1 tensor
        :return:
        """
        inference = self.surrogate_inference((self.original2cube(input_configuration), output_evaluation))
        reference, ref_ind = torch.min(output_evaluation, 0)
        reference = reference.data.squeeze()[0]
        gp_hyper_params = inference.sampling(n_sample=10, n_burnin=0, n_thin=1)
        inferences = deepcopy_inference(inference, gp_hyper_params)
        x0_cand = optimization_candidates(input_configuration, output_evaluation, -1, 1)
        x0, sample_info = optimization_init_points(x0_cand, reference=reference, inferences=inferences)
        next_x_point, pred_mean, pred_std, pred_var, pred_stdmax, pred_varmax = suggest(x0=x0, reference=reference,
                                                                                        inferences=inferences,
                                                                                        acquisition_function=self.acquisition_function,
                                                                                        bounds=sphere_bound(self.radius), pool=self.pool)
        return self.cube2original(next_x_point)


class NonArdSpearmintBayesianOptimization(BayesianOptimization):
    def __init__(self, ndim, original2cube=None, cube2original=None):
        super(NonArdSpearmintBayesianOptimization, self).__init__(original2cube, cube2original)
        self.surrogate_model = GPRegression(kernel=Matern52(ndim=ndim, ard=False))
        self.surrogate_inference = Inference
        self.acquisition_function = expected_improvement


class ArdSpearmintBayesianOptimization(BayesianOptimization):
    def __init__(self, ndim, original2cube=None, cube2original=None):
        super(NonArdSpearmintBayesianOptimization, self).__init__(original2cube, cube2original)
        self.surrogate_model = GPRegression(kernel=Matern52(ndim=ndim, ard=True))
        self.surrogate_inference = Inference
        self.acquisition_function = expected_improvement


class CylindricalKernelBayesianOptimization(BayesianOptimization):
    def __init__(self, ndim, original2cube=None, cube2original=None):
        super(CylindricalKernelBayesianOptimization, self).__init__(original2cube, cube2original)
        self.radius = ndim ** 0.5
        radius_input_map = Kumaraswamy(ndim=1, max_input=self.radius)
        self.surrogate_model = GPRegression(kernel=RadializationKernel(max_power=3, search_radius=self.radius, radius_input_map=radius_input_map))
        self.surrogate_inference = origin_ShadowInference
        self.acquisition_function = expected_improvement
