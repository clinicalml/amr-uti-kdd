#!/bin/bash

source setup/paths.sh
conda env list

cd experiments

echo "Training / evaluating outcome models with saved hyperparameters"
bash scripts/eval_test_scripts/train_outcome_models_eval_test_rep.sh

echo "Thresholding on test with saved validation results"
bash scripts/eval_test_scripts/thresholding_test_rep.sh

echo "Expected Reward Maximization on test"
bash scripts/eval_test_scripts/expected_reward_test_rep.sh
