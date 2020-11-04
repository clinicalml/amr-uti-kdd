#!/bin/bash

python experiment_direct.py --exp_name "direct_learning_validation" \
                          --features_data_path "${DATA_PATH}/train_uncomp_uti_features.csv" \
                          --resist_path "${DATA_PATH}/train_uncomp_resist_data.csv" \
                          --validate \
                          --omegas 0.85 1.0 31 \
                          --num_trials 20 \
                          --lr 0.0001 \
                          --num_epochs 50 \
                          --reg_type 'l2' \
                          --lambda_reg 0.003 
