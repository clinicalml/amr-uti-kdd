import os
import sys

import pandas as pd
import numpy as np

sys.path.append("../")

from utils.create_cohorts import split_cohort_new
from models.direct.policy_model_pytorch import ABXPolicyModel

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
# from xgboost import XGBClassifier

from sklearn.feature_selection import VarianceThreshold

def apply_variance_threshold(X, selector=None):
    if selector is None:
        selector = VarianceThreshold()
        selector.fit(X)

    X = selector.transform(X)
    return X, selector


def get_base_model(model_class):
    if model_class =='lr':
        clf = LogisticRegression()
    elif model_class == 'dt':
        clf = DecisionTreeClassifier()
    elif model_class == 'rf':
        clf = RandomForestClassifier()
    # elif model_class == 'xgb':
    #     clf = XGBClassifier()
    else:
        raise ValueError("Model class not supported.")

    return clf


def filter_cohort_for_label(cohort_df, resist_df, drug_code):
    eids_with_label = resist_df[~resist_df[drug_code].isna()]['example_id'].values
    return cohort_df[cohort_df['example_id'].isin(eids_with_label)], resist_df[resist_df['example_id'].isin(eids_with_label)]


def load_direct_policy_actions(exp_path,
	                        features_df, resist_df,
	                        cohort_info_df, omega,
	                        num_trials=20):

    all_action_dfs = []
    for trial in range(num_trials):

        train_cohort, _, val_cohort, _ = split_cohort_new(features_df, resist_df, cohort_info_df,
                                                          seed=24+trial,
                                                          train_prop=.7)


        direct_model = ABXPolicyModel(num_inputs=features_df.shape[1]-1,
                                      num_outputs=4)

        model_name = f'models/trial_{trial}_omega_{omega}_final.pth'
        direct_model.load_weights(os.path.join(exp_path, model_name))

        train_actions_df = direct_model.get_actions(train_cohort)
        train_actions_df['is_train'] = 1

        val_actions_df = direct_model.get_actions(val_cohort)
        val_actions_df['is_train'] = 0

        actions_for_split_df = pd.concat([train_actions_df, val_actions_df], axis=0)
        actions_for_split_df['split_ct'] = trial
        all_action_dfs.append(actions_for_split_df)

    return pd.concat(all_action_dfs, axis=0)
