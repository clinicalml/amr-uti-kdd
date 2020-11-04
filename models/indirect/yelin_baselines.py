import argparse
import pandas as pd
import numpy as np

import os
import sys

sys.path.append('../')
from utils.evaluation_utils import get_iat_broad, get_iat_broad_bootstrapped


def get_policy_constrained(preds_df, resist_df, params=None):

    if params is None:
        params = {
            'init_step_size': 0.1, 
            'tol': 0.9,
            'relax': 0.95,
            'max_iter': 10000
        }
    
    n, d = len(preds_df), 4
    abx_list = ['NIT', 'SXT', 'CIP', 'LVX']
    
    tol = params['tol'] / n
    relax = params['relax']
    max_iter = params['max_iter']

    dcost = params['init_step_size'] * np.ones(d)
    cost = np.zeros(d)
    
    target_freq = resist_df['prescription'].value_counts(normalize=True)
    target_freq = np.array([target_freq.get(abx) for abx in abx_list])
    current_freq = target_freq.copy()
    
    preds = preds_df[[f'predicted_prob_{abx}' for abx in abx_list]].values
    
    for i in range(max_iter):
        cost += dcost * np.sign(current_freq - target_freq)
        cost -= np.mean(cost)
        
        dcost *= relax
        
        adjusted_preds = preds + cost
        curr_policy = np.argmin(adjusted_preds, axis=1)
        min_preds = np.min(adjusted_preds, axis=1)
        current_freq = np.bincount(curr_policy, minlength=4) / n
        
        error = np.max(np.abs(current_freq - target_freq)) * n
        
        if i % 1000 == 0:
            print(current_freq)
            print(f"Error at iteration {i}: {error}")
        
        if error < tol:
            break
        
    final_policy_df = preds_df[['example_id']].copy()
    final_policy_idx = np.argmin(preds + cost, axis=1)
    
    final_policy_df['policy'] = [abx_list[action_idx] for action_idx in final_policy_idx]
    
    policy_resist_df = final_policy_df.merge(resist_df, on='example_id')
    stats = get_iat_broad(policy_resist_df, col_name='policy')
    
    return final_policy_df, cost, stats
    

def get_policy_unconstrained(preds_df, resist_df):

    outcomes = ['NIT', 'SXT', 'CIP', 'LVX']

    def get_policy_for_row(row):
        row_preds = row[[f'predicted_prob_{outcome}' for outcome in outcomes]]
        return row_preds.idxmin()[-3:]

    preds_df['policy'] = preds_df.apply(get_policy_for_row, axis=1)
    policy_df = preds_df[['example_id', 'policy']].copy()
    policy_outcomes_df  = policy_df.merge(resist_df, on='example_id')

    stats = get_iat_broad(policy_outcomes_df, col_name='policy')

    return policy_df, stats



parser = argparse.ArgumentParser(description='process parameters for experiment')

parser.add_argument('--resist_path', 
                    type=str, required=True,
                    help='Path to resistance data')

parser.add_argument('--preds_path',
                    type=str, required=True,
                    help='Path to predictions from indirect outcome models')

parser.add_argument('--mode', 
                    type=str, choices=['constrained', 'unconstrained'],
                    default='constrained', help='Baseline option')


if __name__ == '__main__':
    args = parser.parse_args()

    preds_df = pd.read_csv(args.preds_path)
    preds_df = preds_df[preds_df['is_train'] == 0] 

    resist_df = pd.read_csv(args.resist_path)

    if args.mode == 'constrained':
        _, _, stats = get_policy_constrained(preds_df, resist_df)

    elif args.mode == 'unconstrained':
        _, stats = get_policy_unconstrained(preds_df, resist_df)
        
    print(f'IAT: {stats[0]}, 2nd line usage: {stats[1]}')

