import argparse
import logging
from datetime import datetime 

import numpy as np
import pandas as pd

import os
import sys
sys.path.append('../')

from models.indirect.expected_reward_maximization import *


parser = argparse.ArgumentParser(description='process parameters for experiment')

parser.add_argument('--exp_name',
                     type=str, required=True,
                     help='Name of experiment')

parser.add_argument('--preds_path',
                    type=str,
                    help='Filepath for predicted resistance probabilities from trained outcome models')

parser.add_argument('--resist_path',
                    type=str, required=True,
                    help='Filepath for resistance data')

parser.add_argument('--omegas',
                     type=float, nargs="+",
                     help='Range of omega values to be tested')

parser.add_argument('--eval_cohort', 
                    type=str, required=True,
                    help='Cohort to be evaluated')

parser.add_argument('--num_trials', 
                    type=int, default=20,
                    help='Number of trials in provided predictions file')



if __name__ == '__main__':
    args = parser.parse_args()       

    log_time = datetime.now().strftime("%d-%m-%Y_%H%M%S")

    log_folder_path = f"experiment_results/expected_reward/{args.exp_name}/logs" 
    results_path = f"experiment_results/expected_reward/{args.exp_name}/results" 

    if not os.path.exists(log_folder_path): 
        os.makedirs(log_folder_path)

    if not os.path.exists(results_path): 
        os.makedirs(results_path)


    logging.basicConfig(filename=f"experiment_results/expected_reward/{args.exp_name}/logs/experiment_{log_time}.log",
                        format='%(asctime)s - %(message)s',
                        level=logging.INFO)
    logging.info(args)
    logging.info("Reading in data...")

    preds_df = pd.read_csv(args.preds_path)
    resist_df = pd.read_csv(args.resist_path)

    start, stop, count = args.omegas
    omegas = [round(omega, 4) for omega in np.linspace(start, stop, int(count))]
    
    frontier_eval_df = construct_frontier_df(preds_df, resist_df,
                                             omegas, num_trials=args.num_trials)
    frontier_eval_df.to_csv(os.path.join(results_path, f"frontier_{args.eval_cohort}.csv"), index=None)

    logging.info("Finished with model training.")


