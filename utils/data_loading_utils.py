import os
import sys

sys.path.append("../")

from models.direct.policy_model_pytorch import ABXPolicyModel

def load_models(exp_results_folder,
                test_folder, 
                num_trials, 
                num_inputs,
                omegas):

    '''
        Loads models from the specified experiment folder, returns list of model dictionaries.
            - Each element in list contains a dictionary of models for a particular trial in the experiment.
            - Each dictionary is a map from omega value to the corresponding trained model.
    '''

    models_root_path = f"{exp_results_folder}/{test_folder}/models"
    omegas_cleaned = [round(omega, 3) for omega in omegas]
    
    models_dict_list = []
    
    for trial_num in range(num_trials):
        models_dict_for_trial = {}
        
        for omega in omegas_cleaned:
            model_weights_path = f"{models_root_path}/trial_{trial_num}_omega_{omega}_final.pth"
            
            model = ABXPolicyModel(num_inputs=num_inputs,
                                   num_outputs=4)
            model.load_weights(model_weights_path)
            models_dict_for_trial[omega] = model

        models_dict_list.append(models_dict_for_trial)
    
    return models_dict_list


def load_frontier(exp_results_folder, 
                  test_folder,
                  cohort_name):

    '''
        Loads the computed frontier for a specific cohort (train / val / test)
        in the given experiment.
    '''
    assert cohort in ['train', 'val', 'test']
    return pd.read_csv(f"{exp_results_folder}/{test_folder}/frontier_{cohort_name}.csv")
