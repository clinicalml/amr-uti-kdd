#!/bin/bash

python experiment_direct.py --exp_name "direct_learning_test" \
                          --features_data_path "${DATA_PATH}/train_uncomp_uti_features.csv" \
                          --resist_path "${DATA_PATH}/train_uncomp_resist_data.csv" \
                          --eval_test \
                          --test_features_data_path "${DATA_PATH}/test_uncomp_uti_features.csv" \
                          --test_resist_path "${DATA_PATH}/test_uncomp_resist_data.csv" \
                          --omegas 0.80 1.0 41 \
                          --num_trials 20 \
                          --lr 0.0001 \
                          --num_epochs 50 \
                          --reg_type 'l2' \
                          --lambda_reg 0.003 
