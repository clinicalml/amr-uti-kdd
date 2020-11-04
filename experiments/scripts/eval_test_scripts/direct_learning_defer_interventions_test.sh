#!/bin/bash

python experiment_direct.py --exp_name "direct_learning_defer_interventions_test" \
                          --features_data_path "${DATA_PATH}/train_uncomp_uti_features.csv" \
                          --resist_path "${DATA_PATH}/train_uncomp_resist_data.csv" \
                          --eval_test \
                          --test_features_data_path "${DATA_PATH}/test_uncomp_uti_features.csv" \
                          --test_resist_path "${DATA_PATH}/test_uncomp_resist_data.csv" \
                          --mode 'defer' \
                          --omega 0.92 \
                          --r_defer_vals 0.0 0.1 21 \
                          --num_trials 100 \
                          --lr 0.0001 \
                          --num_epochs 50 \
                          --reg_type 'l2' \
                          --lambda_reg 0.003 

