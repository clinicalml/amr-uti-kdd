import logging
import sys
sys.path.append('../')

from datetime import datetime
from collections import defaultdict

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from models.create_rewards_linear import get_reward_mapping
from models.direct.policy_model_pytorch import ABXPolicyModel
from utils.create_cohorts import split_cohort, split_cohort_new, split_cohort_with_subcohort


def run_experiment(exp_name, num_trials, mode, params,
                   cohort_df, cohort_info_df, cohort_resist_df,
                   num_epochs, optimizer, lr, reg_type, lambda_reg,
                   validate=True,
                   importance_weights=None,
                   random_state=10,
                   num_samples=None,
                   subcohort_eids=None,
                   split_by_hosp=False,
                   train_both_hosp=True):

    model_dicts_list, train_val_splits = [], []

    assert mode in ['no_defer', 'defer']

    for trial in range(num_trials):
        logging.info(f"Starting trial {trial}")

        if validate:
            train_cohort_df, train_resist_df, val_cohort_df, val_resist_df = split_cohort_with_subcohort(cohort_df, cohort_resist_df,
                                                                                                        cohort_info_df,
                                                                                                        seed=24+trial, train_prop=.7,
                                                                                                        subcohort_eids=subcohort_eids,
                                                                                                        split_by_hospital=split_by_hosp,
                                                                                                        train_both_hospitals=train_both_hosp)
        else:
            train_cohort_df, val_cohort_df = cohort_df, cohort_df.copy()
            train_resist_df, val_resist_df = cohort_resist_df, cohort_resist_df.copy()

        train_val_splits.append({'train': train_cohort_df, 'val': val_cohort_df})


        if mode == 'no_defer':
            models_dict_for_trial = train_model_no_defer(exp_name, trial,
                                                        train_cohort_df, val_cohort_df,
                                                        train_resist_df, val_resist_df,
                                                        params, num_epochs, optimizer, lr, reg_type, lambda_reg,
                                                        importance_weights=importance_weights,
                                                        random_state=random_state,
                                                        num_samples=num_samples)

        elif mode == 'defer':
            models_dict_for_trial = train_model_defer(exp_name, trial,
                                                        train_cohort_df, val_cohort_df,
                                                        train_resist_df, val_resist_df,
                                                        params, num_epochs, optimizer, lr, reg_type, lambda_reg,
                                                        random_state=random_state,
                                                        num_samples=num_samples)


        model_dicts_list.append(models_dict_for_trial)

    return model_dicts_list, train_val_splits



def train_model_no_defer(exp_name, trial,
                          train_cohort_df, val_cohort_df,
                          train_resist_df, val_resist_df,
                          params, num_epochs, optimizer, lr, reg_type, lambda_reg,
                          importance_weights=None,
                          random_state=10, num_samples=0):
    models_dict = {}

    assert 'omegas' in params
    omegas = params['omegas']

    for omega in omegas:

        logging.info(f"Running experiment for omega={omega}")

        train_reward_df = get_reward_mapping(train_resist_df, omega,
                                             importance_weights_df=importance_weights)
        val_reward_df = get_reward_mapping(val_resist_df, omega)

        model_desc = f'trial_{trial}_omega_{omega}'
        model = ABXPolicyModel(num_inputs=train_cohort_df.shape[1]-1,
                               num_outputs=4, desc=model_desc, exp_name=exp_name)

        if num_samples > 0:
            train_cohort_shuffled = train_cohort_df.sample(frac=1, random_state=random_state)
            train_cohort_df = train_cohort_shuffled.iloc[:n_samples]
            train_reward_df = train_reward_df.iloc[train_cohort_df.index]

        model.train_abx_policy(train_cohort_df, val_cohort_df,
                               train_reward_df, val_reward_df,
                               train_resist_df, val_resist_df,
                               num_epochs=num_epochs,
                               optimizer=optimizer, lr=lr,
                               reg_type=reg_type, lambda_reg=lambda_reg)

        models_dict[omega] = model

    return models_dict


def train_model_defer(exp_name, trial,
                      train_cohort_df, val_cohort_df,
                      train_resist_df, val_resist_df,
                      params, num_epochs, optimizer, lr, reg_type, lambda_reg,
                      random_state=10, num_samples=0):
    models_dict = {}

    assert ('omega' in params) and ('r_defer_vals' in params) and ('use_any_resist' in params)

    omega, r_defer_vals = params['omega'], params['r_defer_vals']
    use_any_resist = params['use_any_resist']
    inc_alg_error_cost = params['inc_alg_error_cost']

    for r_defer in r_defer_vals:
        logging.info(f"Running experiment for r_defer={r_defer}")

        train_reward_df = get_reward_mapping(train_resist_df,
                                             omega, r_defer=0.1,
                                             use_any_resist=use_any_resist,
                                             increase_alg_error_cost=inc_alg_error_cost,
                                             include_defer=True)

        val_reward_df = get_reward_mapping(val_resist_df,
                                           omega, r_defer=0.1,
                                           use_any_resist=use_any_resist,
                                           increase_alg_error_cost=inc_alg_error_cost,
                                           include_defer=True)

        model_desc = f'trial_{trial}_rdefer_{r_defer}'
        model = ABXPolicyModel(num_inputs=train_cohort_df.shape[1]-1,
                              num_outputs=5,
                              desc=model_desc,
                              exp_name=exp_name)

        if num_samples > 0:
            train_cohort_shuffled = train_cohort_df.sample(frac=1, random_state=random_state)
            train_cohort_df = train_cohort_shuffled.iloc[:n_samples]
            train_reward_df = train_reward_df.iloc[train_cohort_df.index]

        model.train_abx_policy(train_cohort_df, val_cohort_df,
                               train_reward_df, val_reward_df,
                               train_resist_df, val_resist_df,
                               num_epochs=25,
                               optimizer=optimizer, lr=lr, reg_type=reg_type,
                               lambda_reg=lambda_reg, avoid_last=True)

        train_reward_df = get_reward_mapping(train_resist_df,
                                                  omega, r_defer=r_defer,
                                                  use_any_resist=use_any_resist,
                                                  increase_alg_error_cost=inc_alg_error_cost,
                                                  include_defer=True)
        val_reward_df = get_reward_mapping(val_resist_df,
                                           omega, r_defer=r_defer,
                                           use_any_resist=use_any_resist,
                                           increase_alg_error_cost=inc_alg_error_cost,
                                           include_defer=True)

        model.train_abx_policy(train_cohort_df,
                             val_cohort_df,
                             train_reward_df,
                             val_reward_df,
                             train_resist_df,
                             val_resist_df,
                             num_epochs=num_epochs,
                             optimizer=optimizer,
                             lr=lr,
                             reg_type=reg_type,
                             lambda_reg=lambda_reg)

        models_dict[r_defer] = model

    return models_dict


def process_experiment_results(model_dicts_list,
                               train_val_splits,
                               resist_df):
    all_stats = defaultdict(list)

    for train_val_split, model_dict in zip(train_val_splits, model_dicts_list):
        param_vals = sorted(list(model_dict.keys()))

        for cohort_name, cohort_df in train_val_split.items():
            stats_for_param = []
            for param in param_vals:
                logging.info("Storing primary outcomes...")

                cohort_resist_df = cohort_df[['example_id']].merge(resist_df, on='example_id', how='inner')
                iat_stats, broad_stats = model_dict[param].get_iat_broad_stats(cohort_df,
                                                                               cohort_resist_df)
                logging.info("Storing deferral outcomes...")
                iat_defer_stats, broad_defer_stats = model_dict[param].get_decision_cohort_stats(cohort_df,
                                                                                            cohort_resist_df)

                defer_rate = model_dict[param].get_action_distribution(cohort_df).get('defer', 0)/len(cohort_df)

                stats = [iat_stats[0], broad_stats[0], defer_rate] + iat_defer_stats + broad_defer_stats
                stats_for_param.append(np.array(stats))

            all_stats[cohort_name].append(np.array(stats_for_param))

    columns = [
        'param', 'iat', 'broad', 'defer_rate',
        'iat_decision', 'iat_doc', 'iat_defer',
        'broad_decision', 'broad_doc', 'broad_defer'
    ]

    stats_dict_final = {}
    for cohort_name, stats_for_cohort in all_stats.items():
        stats_means = np.array(stats_for_cohort).mean(axis=0)
        logging.info(f"Shape of means: {stats_means.shape}")

        stats_final = np.hstack([np.array([param_vals]).T,
                                stats_means])

        logging.info("Completed calculating means")
        stats_dict_final[cohort_name] =  pd.DataFrame(stats_final, columns=columns)

    return stats_dict_final



def process_model_performance_at_epochs(models_folder, params,
                                         num_trials, num_epochs,
                                         cohort_df, resist_df,
                                         epoch_interval=10):

    frontiers_by_epoch = {}

    # Calculate frontier for each set of epochs
    for num_epochs in range(epoch_interval, num_epochs + epoch_interval, epoch_interval):

        model_dicts_list = []

        # Create a list of model dicts for each number of epochs
        for trial in range(num_trials):
            models_dict = {}

            for param in params:
                model = ABXPolicyModel(num_inputs=cohort_df.shape[1]-1,
                                       num_outputs=4)
                model.load_weights(f"{models_folder}/trial_{trial}_omega_{param}_{num_epochs}_epochs.pth")
                models_dict[param] = model

            model_dicts_list.append(models_dict)

        stats_for_epoch = process_experiment_results(params,
                                                     model_dicts_list,
                                                     cohort_df,
                                                     resist_df)
        frontiers_by_epoch[num_epochs] = stats_for_epoch


    return frontiers_by_epoch


def save_frontier_to_csv(frontier, exp_name, log_time,
                         cohort, num_epochs=None):

    path_name = f"experiment_results/direct/{exp_name}/frontier_{cohort}"
    if num_epochs:
        path_name = f"{path_name}_{num_epochs}_epochs"

    frontier.to_csv(f"{path_name}.csv", index=None)



def get_model_eval_at_checkpoints(model_dicts_list, omegas):
    iats_at_checkpoints_all = defaultdict(list)
    broads_at_checkpoints_all = defaultdict(list)

    for model_dict in model_dicts_list:
        iats_at_checkpoints = defaultdict(list)
        broads_at_checkpoints = defaultdict(list)

        for omega in sorted(list(model_dict.keys())):
            model = model_dict[omega]
            for checkpoint, stats in model.evaluate_checkpoints.items():
                iat, broad = stats
                iats_at_checkpoints[checkpoint].append(iat)
                broads_at_checkpoints[checkpoint].append(broad)

        for checkpoint, frontier in iats_at_checkpoints.items():
            iats_at_checkpoints_all[checkpoint].append(frontier)

        for checkpoint, frontier in broads_at_checkpoints.items():
            broads_at_checkpoints_all[checkpoint].append(frontier)

    for checkpoint, frontiers in iats_at_checkpoints_all.items():
        iats_at_checkpoints_all[checkpoint] = np.array(frontiers).mean(axis=0)

    for checkpoint, frontiers in broads_at_checkpoints_all.items():
        broads_at_checkpoints_all[checkpoint] = np.array(frontiers).mean(axis=0)


    return {
        checkpoint: pd.DataFrame([omegas,
                    iats_at_checkpoints_all[checkpoint],
                    broads_at_checkpoints_all[checkpoint]],
                    index=['omega','iat','broad']).transpose()
        for checkpoint in iats_at_checkpoints_all.keys()
    }



