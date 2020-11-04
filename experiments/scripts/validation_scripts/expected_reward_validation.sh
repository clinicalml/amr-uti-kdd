#!/bin/bash

python experiment_expected_reward.py --exp_name 'expected_reward_validation' \
                                     --preds_path "${VAL_OUTCOME_MODEL_PATH}/val_predictions.csv" \
				     --resist_path "${DATA_PATH}/train_uncomp_resist_data.csv" 
