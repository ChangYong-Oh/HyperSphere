# HyperSphere

This is the implementation of the paper **BOCK: Bayesian Optimization with Cylindrical Kernels**(https://arxiv.org/pdf/1806.01619.pdf).
Goal of this paper is to propose a new Bayesian Optimization algorithm for high dimensional problems. Usually, in Bayesian Optimization, we have just a handful of data compared to these days' big data. We infuse our strong prior knowledge that we want to find a solution around the center of the search space. Moreover, due to statistical efficiency, we try to keep the number of learnable optimal in a sense that it is samll enough not to degrade statistical efficiency and large enough to cover variety of functions we bump into in black-box function optimization problems.

## 1. Setup

To set up this repo, go:

** Virtual Environment _Without conda_ **

```
git clone https://github.com/ChangYong-Oh/HyperSphere.git
cd HyperSphere
source setup_pip.sh
```

** Virtual Environment _With conda_ **

```
conda create -n HyperSphere python=2.7.14 anaconda --yes
cd "`which python | xargs dirname | xargs dirname`/envs/HyperSphere"
git clone https://github.com/ChangYong-Oh/HyperSphere.git
source HyperSphere/setup_conda.sh
```

Default python should be the anaconda python.

Different python version is possible. For avaialbe version search
```
conda search "^python$"
```

** Import in existing Python environment **


Or to be able to import this code in an existing Python environment, go:

```
pip install -e git+https://github.com/ChangYong-Oh/HyperSphere.git#egg=HyperSphere
```

## 2. How to run
* Set **EXPERIMENT_DIR** in **HyperSphere/BO/run_BO.py**
* (Default values are given) Set other configurations on acqusition function maxmization **N_SPREAD**, **N_SPRAY**, **N_INIT** in **HyperSphere/BO/acquisition/acquisition_maximization.py**


### 2-1. Algorithm options
* -g [sphere/cube] : 'sphere' corresponds to cylindrical transformation / 'cube' corresponds to ordinary cube search space
* --origin [True] : This is valid only when __-g sphere__ is given, this adjusts the tranformation of the center point according to the location of the prediction point.
* --warping [True] : This is valid only when __-g sphere__ is given, this applies learnable warping to radius component in cylindrical kernels.
* --boundary [True] : This is valid only when __-g sphere__ is given, this adds fake point on the boundary sphere of the search space like a stationary satellite to reduce the predictive variance near the boundary.
* --ard [True] : This is valid only when __-g cube__ is given, this makes BO use ARD kernel without cylindrical transformation.
* --parallel [True] : Acquisition function is maximized with multiple initial points (number of initial points can be set in HyperSphere/BO/acquisition/acquisition_maximization.py). This options is for efficiency.

### 2-2. Options for a function to be optimized
* -f : The name of the python function used for evaluation
* -e : The number of additional evaluations
* -d : For some test functions, dimension should be specificed, otherwise, this is ignored.
* -p : When you want to continue an experiment, you can specify the directory, this overwrites all options with the optiones specified in given directory.

### 2-3. Examples
BOCK for 95 dimensional branin with 47 evaluations (exclusive of first 2 evaluations)
```
../HyperSphere $ python HyperSphere/BO/run_BO.py -g sphere --origin --warping --parallel -func branin -d 95 -e 47
```
BOCK-W (without warping) for 32 dimensional bird with 54 evaluations (exclusive of first 2 evaluations)
```
../HyperSphere $ python HyperSphere/BO/run_BO.py -g sphere --origin --parallel -func bird -d 32 -e 54
```
BOCK+B (with boundary) for 32 dimensional bird with 54 evaluations (exclusive of first 2 evaluations)
```
../HyperSphere $ python HyperSphere/BO/run_BO.py -g sphere --origin --warping --boundary --parallel -func bird -d 32 -e 54
```
Matern for 32 dimensional bird with 54 evaluations (exclusive of first 2 evaluations)
```
../HyperSphere $ python HyperSphere/BO/run_BO.py -g cube --parallel -func bird -d 32 -e 54
```
Matern-ARD for 32 dimensional bird with 54 evaluations (exclusive of first 2 evaluations)
```
../HyperSphere $ python HyperSphere/BO/run_BO.py -g cube --ard --parallel -func bird -d 32 -e 54
```
Continuing an existing experiment with 10 more evaluations
```
../HyperSphere $ python HyperSphere/BO/run_BO.py -p [EXPERIMENT_DIR]/branin_D20_spherewarpingorigin_20180728-12:13:32:828257 -e 10
```

### 2-5. Optimizing new functions
Only thing you need to do is to provide python function for evaluation.
When you have a file target.py with below function.
```
def factory_cooling_bill(control_factors)
	...
    return cost
factory_cooling_bill.dim = [number of elements in control_factors] 
#if you don't set dimension here, then dimension should be given as an option (e.g -d 57).
```
Then you can import this function in **HyperSphere/BO/runb_BO.py** as
```
from ../../../target import factory_cooling_bill
```
then you run BO with the option **-f factory_cooling_bill**.

Input for evaluation (e.g control_factors) is assumed to lie on a ball with radius D^0.5, where D is input dimension.
Preprocessing on input can be included in the python function for evaluation.
For example, if you optimize layerwise learning rates for 10 layer neural network, then as a preprocessing, suggested point [x_1, ..., x_D] can be transformed into [exp(0.01 + x_1), ..., exp(0.01 + x_D)] (in this example, you are assuming that 0.01 is a good learning rate for each layer or you want to find an optima near the learning rate of 0.01).

You can refer to example files in **HyperSphere/test_functions**.

### 2-6. Some implementation details
* When selecting good initial points for acquisition function maxmization, up to 1100 dimensional problem, sobol sequence is used and for higher dimensions, uniform random is used. This may degrade the performance on the problem with dimensions > 1100.


## 3. Common Errors from sampyl
You may encounter some errors from sampyl package.
This may be due to an error in the code or bad prior specification (prior for GP hyperparameters)
* Code errors : This occurs with old version of sampyl, sampyl appears to be updated constantly so you can check https://github.com/mcleonard/sampyl and download newest version. Or you can just replace the part of the code as below.
* Bad prior specification : You need to set your prior so that numerical instability is prevented.

Belows are the common errors I ran into.

### 3-1. (code errors) ValueError: setting an array element with a sequence.
```
File "../../site-packages/sampyl/samplers/base.py", line xxx, in sample
    samples[i] = next(self.sampler).tovector
ValueError: setting an array element with a sequence.
```
You can change the code to
```
    samples[i] = tuple(next(self.sampler).values())
```

### 3-2. (bad prior) Exception : Slice sampler shrank to zero!
This exception happens mostly when your priors for GP hyperparameters is set vulnerable to numerical instability.