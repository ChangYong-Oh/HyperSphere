#!/bin/bash

python HyperSphere/BO/sphere_bayesian_optimization.py rosenbrock 5 100
python HyperSphere/BO/cube_bayesian_optimization.py rosenbrock 5 100
python HyperSphere/BO/sphere_bayesian_optimization.py rosenbrock 10 200
python HyperSphere/BO/cube_bayesian_optimization.py rosenbrock 10 200

python HyperSphere/BO/sphere_bayesian_optimization.py levy 5 100
python HyperSphere/BO/cube_bayesian_optimization.py levy 5 100
python HyperSphere/BO/sphere_bayesian_optimization.py levy 10 200
python HyperSphere/BO/cube_bayesian_optimization.py levy 10 200
