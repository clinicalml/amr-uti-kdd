import pandas as pd
import numpy as np

def get_reward_mapping(cohort_resist_data, 
                       omega, importance_weights_df=None,
                       r_defer=0, include_defer=False,
                       use_any_resist=False,
                       increase_alg_error_cost=False):
    '''
        Returns DataFrame of rewards across all specimens and actions in 
        specified cohort
    '''
    if importance_weights_df is not None:
        cohort_resist_data = cohort_resist_data.merge(importance_weights_df, on='example_id', how='inner')

    reward_df = cohort_resist_data.apply(lambda x: get_reward_for_patient(x, omega,
                                                                          r_defer, include_defer,
                                                                          use_any_resist=use_any_resist,
                                                                          increase_alg_error_cost=increase_alg_error_cost), axis=1)
    reward_df['example_id'] = reward_df['example_id'].astype('int32')
    return reward_df


def get_reward_for_patient(resist_data, omega,
                           r_defer=0, include_defer=False,
                           use_any_resist=False,
                           increase_alg_error_cost=False):
    '''
        Given resistance profile for a single specimen, returns rewards 
        associated with each action for this patient
    '''
    reward = {}
    abx_list = ['NIT', 'SXT', 'CIP', 'LVX']

    def get_reward_for_abx(abx, omega):
        return omega * (1 - resist_data[abx]) + (1 - omega) * int(abx in ['NIT', 'SXT'])

    # Rewards for actions other than deferral
    for abx in abx_list:
        reward[abx] = get_reward_for_abx(abx, omega=omega)
  
    # Reward construction for deferral 
    if include_defer:  
        abx_list.append('defer')

        # Deferring to minimize algorithm errors in decision cohort
        if increase_alg_error_cost:
            reward['defer'] = reward[resist_data['prescription']] - r_defer 
            
            if resist_data[resist_data['prescription']] == 1:
                reward['defer'] += omega 

            if resist_data['prescription'] in ['CIP', 'LVX']:
                reward['defer'] += (1-omega)*0.5 

        # Deferring to minimize algorithm errors in decision cohort using any resistance as proxy information
        elif use_any_resist:
            any_resist = any([resist_data[abx] == 1 for abx in ['NIT', 'SXT', 'CIP', 'LVX']])
            beta = 0.3

            if any_resist:
                reward['defer'] = np.max([reward[abx] for abx in abx_list if abx != 'defer']) + r_defer 
            else:
                reward['defer'] = np.min([reward[abx] for abx in abx_list if abx != 'defer']) - beta*r_defer
    

        # Deferring to minimize algorithm interventions in decision cohort
        else:  
            reward['defer'] = get_reward_for_abx(abx=resist_data['prescription'], omega=0.80) + r_defer

    multiplier = resist_data['weight'] if 'weight' in list(resist_data.index) else 1 

    return pd.Series([resist_data['example_id']] + [multiplier*reward[abx] for abx in abx_list],
                     index=['example_id'] + abx_list).astype('float32')
    

