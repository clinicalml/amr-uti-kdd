#!/bin/bash

python experiment_expected_reward.py --exp_name 'expected_reward_eval_test' \
                                      --preds_path "${TEST_OUTCOME_MODEL_PATH}/test_predictions.csv" \
                                      --resist_path "${DATA_PATH}/test_uncomp_resist_data.csv" \
                                      --eval_cohort 'test' \
                                      --omega 0.8 1 81 \
                                      --num_trials 1
