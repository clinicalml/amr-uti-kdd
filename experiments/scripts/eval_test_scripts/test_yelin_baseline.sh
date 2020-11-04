#!/bin/bash

python ../models/indirect/yelin_baselines.py --mode 'constrained' \
                                      --preds_path "${TEST_OUTCOME_MODEL_PATH}/test_predictions.csv" \
                                      --resist_path "${DATA_PATH}/test_uncomp_resist_data.csv" 

python ../models/indirect/yelin_baselines.py --mode 'unconstrained' \
                                      --preds_path "${TEST_OUTCOME_MODEL_PATH}/test_predictions.csv" \
                                      --resist_path "${DATA_PATH}/test_uncomp_resist_data.csv" 

