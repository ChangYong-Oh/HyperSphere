#!/bin/bash

VENV_ROOT_DIR="`which python | xargs dirname | xargs dirname`/envs/HyperSphere"
if [ -d "$VENV_ROOT_DIR" ]; then
	cd "$VENV_ROOT_DIR"
	if [ ! -d "$VENV_ROOT_DIR/.git" ]; then
		echo "Data in HyperSphere is moved to virtual environment root directory."
  		mv HyperSphere HyperSphere_TBR
  		cp -a HyperSphere_TBR/. ./
  		rm -rf HyperSphere_TBR
	else
		echo "Data has been moved"
	fi
	source activate HyperSphere 
	conda install --yes pytorch torchvision -c soumith -n HyperSphere
	pip install -r requirements.txt
else
	echo "Already in virtual environment"
	conda install --yes pytorch torchvision -c soumith -n HyperSphere
	pip install -r requirements.tx
fi
