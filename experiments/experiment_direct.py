import argparse
import logging
from datetime import datetime

import os
import sys
sys.path.append('../')

import numpy as np
import pandas as pd

from run_experiment_direct import run_experiment
from run_experiment_direct import process_experiment_results, save_frontier_to_csv
from run_experiment_direct import process_model_performance_at_epochs

parser = argparse.ArgumentParser(description='process parameters for experiment')

# General experiment setup parameters

parser.add_argument('--exp_name',
                     type=str, required=True,
                     help='Name of experiment')

parser.add_argument('--mode',
                    type=str, choices=['no_defer', 'defer', 'defer_two_stage', 'joint_rejector'],
                    default='no_defer', help='Model training mode')

parser.add_argument('--num_trials',
                     type=int, default=1,
                     help='Number of trials to run experiment')



# Parameters for model training and optimization

parser.add_argument('--num_epochs',
                    type=int, default=10,
                    help='Number of epochs for each trial')

parser.add_argument('--lr',
                    type=float, default=1e-3,
                    help='Learning rate for use in optimizer')

parser.add_argument('--optimizer',
                    type=str, default='adam',
                    help='Optimizer to be used')

parser.add_argument('--reg_type',
                    type=str, default=None,
                    help='Type of regularization (L1 or L2)')

parser.add_argument('--lambda_reg',
                    type=float, default=0,
                    help='Regularization strength')


# Parameters for reward function construction

parser.add_argument('--omegas',
                     type=float, nargs="+",
                     help='Range of omega values to be tested')

parser.add_argument('--omega',
                     type=float,
                     help='Single omega value to be tested')

parser.add_argument('--r_defer_vals',
                     type=float, nargs="+",
                     help='Range of deferral rewards to be used')

parser.add_argument('--use_any_resist',
                    action='store_true',
                    help='Incentivize deferral on examples with resistance to at least one agent')

parser.add_argument('--inc_alg_error_cost',
                    action='store_true',
                    help='Incentivize deferral with asymmetric cost on clinician errors')

parser.add_argument('--importance_weights_path',
                    type=str, required=False,
                    help='Filepath for importance weights to be used during training')


# Data for training / validation

parser.add_argument('--features_data_path',
                    type=str, required=True,
                    help='Filepath for cohort features')

parser.add_argument('--cohort_info_path',
                     type=str,
                     help='Filepath for cohort metadata')

parser.add_argument('--subcohort_info_path',
                    type=str,
                    help='Path to subcohort metadata CSV')

parser.add_argument('--resist_path',
                    type=str, required=True,
                    help='Filepath for resistance data')

parser.add_argument('--validate',
                    action='store_true',
                    help='Whether to perform train / val splits')


# Data for evaluation on test set


parser.add_argument('--eval_test',
                    action='store_true',
                    help='Whether to evaluate on actual test set')

parser.add_argument('--test_features_data_path',
                    type=str,
                    help='Filepath for test cohort features')

parser.add_argument('--test_resist_path',
                    type=str,
                    help='Filepath for test resistance data')


# Parameters for train / validation splitting

parser.add_argument('--random_state',
                     type=int, default=24,
                     help='Random seed')

parser.add_argument('--split_by_hosp',
                    action='store_true',
                    help='Perform train / validation splitting by hospital (MGH/BWH)')

parser.add_argument('--train_both_hosp',
                    action='store_true',
                     help='Used if splitting train/val by hospital. If true, use data from both hospitals in training')

# Miscellaneous parameters


parser.add_argument('--num_samples',
                     type=int, default=0,
                     help='Number of samples to be used in training')



if __name__ == '__main__':
    args = parser.parse_args()

    # Setting up directories for storing logs and trained models
    log_time = datetime.now().strftime("%d-%m-%Y_%H%M%S")
    log_folder_path = f"experiment_results/direct/{args.exp_name}/experiment_logs"
    model_folder_path = f"experiment_results/direct/{args.exp_name}/models"

    if not os.path.exists(log_folder_path):
        os.makedirs(log_folder_path)

    if not os.path.exists(model_folder_path):
        os.makedirs(model_folder_path)

    logging.basicConfig(filename=f"experiment_results/direct/{args.exp_name}/experiment_logs/experiment_{log_time}.log",
                        format='%(asctime)s - %(message)s',
                        level=logging.INFO)
    logging.info(args)

    # Reading in training / validation data
    logging.info("Reading in data...")
    train_cohort = pd.read_csv(args.features_data_path)
    train_resist_df = pd.read_csv(args.resist_path)

    train_cohort_info = None
    if not args.eval_test:
        train_cohort_info = pd.read_csv(args.cohort_info_path)

    logging.info(f"Train cohort shape: {train_cohort.shape}")
    assert list(train_cohort['example_id'].values) == list(train_resist_df['example_id'].values)

    # Create dictionary of parameters for constructing reward function based on training mode
    if args.mode == 'no_defer':
        start, stop, count = args.omegas
        omegas = np.linspace(start, stop, int(count))
        params = {
            'omegas': [round(omega, 4) for omega in omegas]
        }

    else:
        start, stop, count = args.r_defer_vals
        r_defer_vals = np.linspace(start, stop, int(count))

        params = {
            'r_defer_vals': [round(r_defer, 4) for r_defer in r_defer_vals],
            'omega': args.omega,
            'use_any_resist': args.use_any_resist,
            'inc_alg_error_cost': args.inc_alg_error_cost
        }

    # Load in subcohort EIDs - used for appropriate splitting of train / validation set when training on all UTIs
    subcohort_eids = None
    if args.subcohort_info_path is not None:
        subcohort_eids=pd.read_csv(args.subcohort_info_path)['example_id'].values
        logging.info(f"{len(subcohort_eids)} examples in specified subcohort.")

    # Load in importance weights - used for weighting complicated UTIs
    importance_weights_df=pd.read_csv(args.importance_weights_path) if args.importance_weights_path else None
    if importance_weights_df is not None:
        logging.info(f"{len(importance_weights_df)} importance weights loaded.")


    model_dicts_list, train_val_splits = run_experiment(args.exp_name, args.num_trials, args.mode, params,
                                                         train_cohort, train_cohort_info, train_resist_df,
                                                         args.num_epochs, args.optimizer, args.lr,
                                                         args.reg_type, args.lambda_reg,
                                                         args.validate,
                                                         importance_weights=importance_weights_df,
                                                         random_state=args.random_state,
                                                         num_samples=args.num_samples,
                                                         subcohort_eids=subcohort_eids,
                                                         split_by_hosp=args.split_by_hosp,
                                                         train_both_hosp=args.train_both_hosp)


    # If training on all UTIs, we only evaluate on the validation set - evaluating metrics on the training set is not useful
    if subcohort_eids is not None:
        val_splits_only = [{'val': split['val']} for split in train_val_splits]
        avg_frontier_dict = process_experiment_results(model_dicts_list, val_splits_only, train_resist_df)

    # If training on only uncomplicated UTIs, evaluate metrics on both the train/validation splits
    else:
        avg_frontier_dict = process_experiment_results(model_dicts_list, train_val_splits, train_resist_df)

    # Evaluation of trained models on specified test cohort
    if args.eval_test:
        assert (len(args.test_features_data_path) > 0) and (len(args.test_resist_path) > 0)

        test_cohort = pd.read_csv(args.test_features_data_path)
        test_resist_df = pd.read_csv(args.test_resist_path)

        test_splits = [{'test': test_cohort} for _ in range(len(model_dicts_list))]

        avg_frontier_dict_test = process_experiment_results(model_dicts_list, test_splits, test_resist_df)
        avg_frontier_dict.update(avg_frontier_dict_test)


    logging.info("Storing computed policy frontiers...")
    for cohort, frontier in avg_frontier_dict.items():
        save_frontier_to_csv(frontier, args.exp_name, log_time, cohort)

    logging.info("Experiment completed")




