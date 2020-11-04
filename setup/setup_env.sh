#!/bin/bash

ENV_NAME='amr'

conda create -n ${ENV_NAME} python=3.7
conda install pandas -n ${ENV_NAME} 
conda install scikit-learn -n ${ENV_NAME} 
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch -n ${ENV_NAME} 

# These are for notebooks and plotting
conda install jupyter -n ${ENV_NAME} 
conda install matplotlib -n ${ENV_NAME} 
conda install seaborn -n ${ENV_NAME} 
conda install tqdm -n ${ENV_NAME} 

conda activate ${ENV_NAME} 
