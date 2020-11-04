import pandas as pd
import numpy as np

import os
import sys

sys.path.append("../../")
from utils.evaluation_utils import get_iat_broad, get_iat_broad_bootstrapped

def get_policy_omega(row, omega):
    rewards = {}
    
    for abx in ['NIT', 'SXT']:
        rewards[abx] = (1-row[f'predicted_prob_{abx}'])* (1) + (row[f'predicted_prob_{abx}'])*(1-omega)
  
    for abx in ["CIP", "LVX"]:
        rewards[abx] = (1-row[f'predicted_prob_{abx}']) * (omega)

    return pd.Series(rewards).idxmax() 
    

def construct_frontier_df(preds_df, resist_df, omegas,
                          num_trials=20):

    stats_all_splits = []
    
    for split in range(num_trials):

        if num_trials == 1:
            preds_for_split_df = preds_df[(preds_df['is_train'] == 0)].reset_index(drop=True)
        else:
            preds_for_split_df = preds_df[(preds_df['split_ct'] == split) & (preds_df['is_train'] == 0)].reset_index(drop=True)

        preds_resist_df = preds_for_split_df.merge(resist_df, on='example_id', how='inner')

        for omega in omegas:
            preds_resist_df['policy'] = preds_resist_df.apply(lambda x: get_policy_omega(x, omega), axis=1)    
            iat, broad = get_iat_broad_bootstrapped(preds_resist_df, col_name='policy') if num_trials == 1 else get_iat_broad(preds_resist_df, col_name='policy')

            stats_all_splits.append([omega, iat, broad])

    stats_df = pd.DataFrame(stats_all_splits, columns=['omega', 'iat', 'broad'])
    return stats_df.groupby('omega').mean().reset_index()



