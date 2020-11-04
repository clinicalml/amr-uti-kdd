import os
import sys
from tqdm import tqdm 

import numpy as np
import pandas as pd

sys.path.append('../')
from models.synthetic.data_generation import *
from models.synthetic.indirect_learning import *
from models.synthetic.direct_learning import *

from matplotlib import pyplot as plt
import seaborn as sns

sns.set_style('white')
sns.set_palette('deep')

plt.rcParams.update({'font.size': 20})

if __name__ == '__main__':
    mean_outcomes_indirect_final, mean_outcomes_direct_final = [], []
    stdev_outcomes_indirect_final, stdev_outcomes_direct_final = [], []

    num_trials = 25

    # Generate ground truth coefficients
    coeffs = generate_model_coeffs()
    
    # Generate held out test data + filter to specimens with non-uniform outcomes
    X_test_raw, X_test, y_test = generate_data(1000000, coeffs, seed=49)
    max_treatable = y_test.max(axis=1).mean()

    y_test_sums = y_test.sum(axis=1)
    X_test_raw_filtered = X_test_raw[(y_test_sums > 0) & (y_test_sums < 3)]
    y_test_filtered = y_test[(y_test_sums > 0) & (y_test_sums < 3)]

    optimal = get_bayes_outcome(X_test_raw_filtered, y_test_filtered)
    print(f'Bayes optimal treatment rate in (filtered) test set is: {optimal}')

    n_samples_list = [250, 500, 1000, 2000, 5000, 10000]

    for n_samples in n_samples_list:

        print(f'Processing results for {n_samples} training samples')
        
        indirect_outcomes_for_trial, direct_outcomes_for_trial = [], []
        
        for i in tqdm(range(num_trials)):
            X_train_raw, X_train, y_train = generate_data(n_samples, coeffs, seed=49+i)

            # Indirect learning
            mean_outcome_indirect = run_indirect_policy_learning(X_train_raw, X_test_raw_filtered, 
                                                                 y_train, y_test_filtered)
            indirect_outcomes_for_trial.append(1 - mean_outcome_indirect)
            
            # Direct learning 
            model = train_direct_policy(X_train_raw, y_train,
                                 lr=0.1, num_epochs=50,
                                 lambda_reg=0, split_val=False)
            mean_outcome_direct_trial = evaluate_direct_model(model, X_test_raw_filtered,  y_test_filtered)
            direct_outcomes_for_trial.append(1 - mean_outcome_direct_trial)
            
            
        mean_outcomes_indirect_final.append(np.mean(indirect_outcomes_for_trial))
        mean_outcomes_direct_final.append(np.mean(direct_outcomes_for_trial))
        
        stdev_outcomes_indirect_final.append(np.std(indirect_outcomes_for_trial))
        stdev_outcomes_direct_final.append(np.std(direct_outcomes_for_trial))

        print(mean_outcomes_indirect_final)
        print(mean_outcomes_direct_final)

    
    plt.figure(figsize=(10,8))

    lower_bound_direct = [
        (mean - stdev/np.sqrt(num_trials))
        for (mean, stdev) in zip(mean_outcomes_direct_final, stdev_outcomes_direct_final)
    ]

    upper_bound_direct = [
        (mean + stdev/np.sqrt(num_trials))
        for (mean, stdev) in zip(mean_outcomes_direct_final, stdev_outcomes_direct_final)
    ]

    lower_bound_indirect = [
        (mean - stdev/np.sqrt(num_trials))
        for (mean, stdev) in zip(mean_outcomes_indirect_final, stdev_outcomes_indirect_final)
    ]

    upper_bound_indirect = [
        (mean + stdev/np.sqrt(num_trials))
        for (mean, stdev) in zip(mean_outcomes_indirect_final, stdev_outcomes_indirect_final)
    ]

    plt.plot(n_samples_list,
             (np.array(mean_outcomes_direct_final)),
             label='Direct')
    plt.fill_between(n_samples_list, 
                     lower_bound_direct, upper_bound_direct, 
                     alpha=.3)

    plt.plot(n_samples_list,
            (np.array(mean_outcomes_indirect_final)),
            label='Indirect')
    plt.fill_between(n_samples_list, 
                     lower_bound_indirect, upper_bound_indirect, 
                     alpha=.3)

    plt.axhline(y=optimal, linestyle='--',
                label='Optimal',  color='g')

    plt.xlabel("Number of samples")
    plt.ylabel("Mean outcome")
    plt.legend();

    plt.savefig("figures/synthetic-exp-results.pdf", bbox_inches='tight')



    