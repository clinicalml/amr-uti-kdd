#!/bin/bash

source setup/paths.sh
conda env list

cd experiments

echo "Building outcome models - Validation"
bash scripts/validation_scripts/train_outcome_models_validation.sh

echo "Thresholding experiment - Validation"
bash scripts/validation_scripts/thresholding_validation.sh

echo "Training models with chosen hyperparmeters"
bash scripts/eval_test_scripts/train_outcome_models_eval_test.sh

echo "Thresholding / Exp Reward Max on test"
bash scripts/eval_test_scripts/thresholding_test.sh
bash scripts/eval_test_scripts/expected_reward_test.sh

echo "Direct Learning - Test"
bash scripts/eval_test_scripts/direct_learning_test.sh

echo "Ran all experiments except for Figure 2"
echo "which can be run via the command"
echo "bash scripts/eval_test_scripts/direct_learning_defer_interventions_test.sh"
