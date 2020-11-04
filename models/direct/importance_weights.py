# Code modified from https://github.com/vodp/py-kmm

import pandas as pd
import numpy as np
import math

from tqdm import tqdm
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import rbf_kernel
from cvxopt import matrix, spmatrix, solvers
import argparse

from datetime import datetime
import logging

def kernel_mean_matching(X, Z, kern='lin', B=1.0, eps=None):
    nx = X.shape[0]
    nz = Z.shape[0]
    if eps == None:
        eps = B/math.sqrt(nz)
    if kern == 'lin':
        K = np.dot(Z, Z.T)
        kappa = np.sum(np.dot(Z, X.T)*float(nz)/float(nx),axis=1)
    elif kern == 'rbf':
        K = compute_rbf_simple(Z,Z)
        kappa = np.sum(compute_rbf_simple(Z,X),axis=1)*float(nz)/float(nx)
    else:
        raise ValueError('unknown kernel')
    
    print(type(K))
    print(type(kappa))

    # K = scipy_sparse_to_spmatrix(csr_matrix(K)) 
    K =  matrix(K)
    kappa = matrix(kappa) 

    print(K.size) 
    print(kappa.size) 
    
    G = matrix(np.r_[np.ones((1,nz)), -np.ones((1,nz)), np.eye(nz), -np.eye(nz)])
    h = matrix(np.r_[nz*(1+eps), nz*(eps-1), B*np.ones((nz,)), np.zeros((nz,))])
    
    sol = solvers.qp(K, -kappa, G, h)
    coef = np.array(sol['x'])
    return coef

def compute_rbf_simple(X, Z, gamma=0.5):
    return rbf_kernel(X, Z, gamma=gamma) 

def compute_rbf(X, Z, sigma=1.0):
    K = np.zeros((X.shape[0], Z.shape[0]), dtype=float)
    
    for i, vx in tqdm(enumerate(X)):
        K[i,:] = np.exp(-np.sum((vx-Z)**2, axis=1)/(2.0*sigma))
    return K

def scipy_sparse_to_spmatrix(A):
    coo = A.tocoo()
    SP = spmatrix(coo.data.tolist(), coo.row.tolist(), coo.col.tolist(), size=A.shape)
    return SP

parser = argparse.ArgumentParser(description='process parameters for experiment')

parser.add_argument('--features_data_path',
                    type=str, required=True,
                    help='Filepath for cohort features')

parser.add_argument('--subcohort_info_path',
                    type=str,
                    help='Path to subcohort metadata CSV')

parser.add_argument('--max_weight',
                    type=float,
                    help='upper bound on importance weight')

parser.add_argument('--kernel',
                    type=str,
                    help='Choice of kernel (RBF vs. linear)')


if __name__ == '__main__':
    args = parser.parse_args()

    # Setting up directories for storing logs and trained models
    log_time = datetime.now().strftime("%d-%m-%Y_%H%M%S")

    logging.basicConfig(filename=f"importance_weight_logs/construct_importance_weights_{log_time}.log",
                        format='%(asctime)s - %(message)s',
                        level=logging.INFO)
    logging.info(args)

    logging.info("Reading in data...")

    train_cohort = pd.read_csv(args.features_data_path)
    train_uncomp_info_df = pd.read_csv(args.subcohort_info_path)

    train_uncomp_cohort_df = train_cohort[
        train_cohort['example_id'].isin(train_uncomp_info_df['example_id'].values)
    ]
    
    train_comp_cohort_df = train_cohort[
        ~train_cohort['example_id'].isin(train_uncomp_info_df['example_id'].values)
    ]
   
    importance_weights = kernel_mean_matching(
            (train_uncomp_cohort_df.drop(columns=['example_id']).values),
            (train_comp_cohort_df.drop(columns=['example_id']).values)[40000:60000],
            kern=args.kernel,
            B=args.max_weight
        )
    
    importance_weights_df = train_comp_cohort_df[['example_id']].copy().iloc[40000:60000]
    importance_weights_df['weight'] = importance_weights
    importance_weights_df.to_csv(f'importance_weight_results/importance_weights_{args.kernel}_{log_time}.csv', index=None) 

