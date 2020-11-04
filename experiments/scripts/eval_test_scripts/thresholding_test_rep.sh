#!/bin/bash

python experiment_thresholding.py --exp_name 'thresholding_eval_test_rep' \
			      	   --preds_path "${TEST_OUTCOME_MODEL_PATH}/test_predictions.csv" \
                                   --resist_data_path "${DATA_PATH}/train_uncomp_resist_data.csv" \
			      	   --test_resist_data_path "${DATA_PATH}/test_uncomp_resist_data.csv" \
                                   --eval_test \
                                   --val_combo_results_path "${REPO_PATH}/models/replication_hyperparameters/fnr_settings_rep.csv"
