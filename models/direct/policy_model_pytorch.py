import sys
sys.path.append('../')

import math
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
import torch.optim as optim

import numpy as np
import pandas as pd

from models.direct.policy_net import PolicyNetLinear, PolicyNetSingleLayer, policy_loss

class ABXPolicyModel:
    def __init__(self,
                 num_inputs,
                 num_outputs,
                 exp_name=None, desc=None):
        self.action_map = {
            0: 'NIT',
            1: 'SXT',
            2: 'CIP',
            3: 'LVX',
            4: 'defer'
        }
        self.model = PolicyNetLinear(num_inputs, num_outputs)

        # Name of experiment in which model was created
        self.exp_name = exp_name

        # Description string for this model
        self.desc = desc

        # Stores evaluation of primary outcomes at periodic intervals
        self.evaluate_checkpoints = {}


    def load_weights(self, model_path):
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        logging.info("Model loaded for evaluation.")


    def get_weights(self):
        return list(model.model.parameters())[0].detach().numpy().T

    def get_actions(self, cohort, avoid_last=False):
        cohort_tensor = torch.from_numpy(cohort.drop(columns=['example_id']).values)
        action_probs = self.model(cohort_tensor.float(), avoid_last=avoid_last)

        chosen_actions = np.argmax(action_probs.detach().numpy(), axis=1)
        actions_df = pd.DataFrame({'example_id': [eid for eid in cohort['example_id'].values],
                                   'rec': [self.action_map[action] for action in chosen_actions]})

        return actions_df


    def get_action_distribution(self, cohort):
        actions_df = self.get_actions(cohort)
        return actions_df["rec"].value_counts()


    def get_iat_alg_doc_df(self,
                           cohort,
                           resist_df,
                           avoid_last=False):

        actions_df = self.get_actions(cohort, avoid_last=avoid_last)
        resist_actions_df = resist_df.merge(actions_df, on='example_id', how='inner')

        def get_defer_prescrip(row):
            return row['prescription'] if row['rec'] == 'defer' else row['rec']

        resist_actions_df['rec_final'] = resist_actions_df.apply(get_defer_prescrip, axis=1)

        resist_actions_df['iat_alg'] = resist_actions_df.apply(lambda x: x[x.rec_final] == 1.0, axis=1)
        resist_actions_df['iat_doc'] = resist_actions_df.apply(lambda x: x[x.prescription] == 1.0, axis=1)

        resist_actions_df['broad_alg'] = resist_actions_df['rec_final'].isin(['CIP', 'LVX'])
        resist_actions_df['broad_doc'] = resist_actions_df['prescription'].isin(['CIP', 'LVX'])

        return resist_actions_df


    def get_iat_broad_stats(self, cohort, resist_df, avoid_last=False):

        resist_actions_df = self.get_iat_alg_doc_df(cohort, resist_df, avoid_last=avoid_last)

        iat_alg_all = resist_actions_df['iat_alg'].mean()
        iat_doc_all = resist_actions_df['iat_doc'].mean()

        broad_alg_all = resist_actions_df['broad_alg'].mean()
        broad_doc_all = resist_actions_df['broad_doc'].mean()

        iat_for_action_dict = {}
        for action in self.action_map.keys():
            cohort_for_action = resist_actions_df[resist_actions_df.rec == action]
            if len(cohort_for_action) > 0:
                iat_for_action_dict[action] = cohort_for_action['iat_alg'].mean()
            else:
                iat_for_action_dict[action] = 0

        return [iat_alg_all, iat_doc_all, iat_for_action_dict], [broad_alg_all, broad_doc_all]


    def get_decision_cohort_stats(self, cohort, resist_df):

        resist_actions_df = self.get_iat_alg_doc_df(cohort, resist_df)

        decision_cohort = resist_actions_df[resist_actions_df.rec != 'defer']
        defer_cohort = resist_actions_df[resist_actions_df.rec == 'defer']

        alg_decision_iat = decision_cohort['iat_alg'].mean()
        doc_decision_iat = decision_cohort['iat_doc'].mean()

        alg_decision_broad = decision_cohort['broad_alg'].mean()
        doc_decision_broad = decision_cohort['broad_doc'].mean()

        doc_defer_iat = defer_cohort['iat_doc'].mean()
        doc_defer_broad = defer_cohort['broad_doc'].mean()

        return [alg_decision_iat, doc_decision_iat, doc_defer_iat], [alg_decision_broad, doc_decision_broad, doc_defer_broad]


    def get_iat_eids(self, cohort, resist_df, avoid_last=False):
        resist_actions_df = self.get_iat_alg_doc_df(cohort, resist_df, avoid_last=avoid_last)
        return resist_actions_df[resist_actions_df['iat_alg'] == 1].example_id.values


    def train_abx_policy(self,
                         train_cohort, val_cohort,
                         train_rewards_df, val_rewards_df,
                         train_resist_prescrip_df, val_resist_prescrip_df,
                         num_epochs=20,
                         optimizer='adam',
                         lr=1e-3,
                         reg_type=None,
                         lambda_reg=0,
                         avoid_last=False):

        train_features_tensor = torch.from_numpy(train_cohort.drop(columns=['example_id']).values)
        train_rewards_tensor = torch.from_numpy(train_rewards_df.drop(columns=['example_id']).values)

        train_dataset = data_utils.TensorDataset(train_features_tensor.float(),
                                                 train_rewards_tensor.float())

        train_loader = data_utils.DataLoader(train_dataset, batch_size=64, shuffle=True)

        val_features_tensor = torch.from_numpy(val_cohort.drop(columns=['example_id']).values)
        val_rewards_tensor = torch.from_numpy(val_rewards_df.drop(columns=['example_id']).values)

        if optimizer == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=lr)

        elif optimizer == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=lr,
                                  nesterov=True, momentum=.9, weight_decay=.005)

        for epoch in range(num_epochs):
            for feats, rewards in train_loader:
                optimizer.zero_grad()
                output = self.model(feats, avoid_last=avoid_last)

                # Loss from policy distribution
                loss = policy_loss(output, rewards, avoid_last=avoid_last)
                regularization_loss = 0

                # L1 regularization
                if reg_type == 'l1':
                    for name, param in self.model.named_parameters():
                        if 'bias' not in name:
                            regularization_loss += torch.sum(torch.abs(param))

                 # L2 regularization
                if reg_type == 'l2':
                    for name, param in self.model.named_parameters():
                        if 'bias' not in name:
                            if avoid_last: param = param[:,:-1]
                            regularization_loss += torch.sum(torch.pow(param, 2))

                # L2 regularization of difference between weights
                if reg_type == 'l2_diff':
                    param_list = [param for name, param in self.model.named_parameters() if 'bias' not in name]
                    assert len(param_list) == 1

                    num_cols = param_list[0].shape[0]
                    assert num_cols == 5
                    param = param_list[0]

                    for i in range(num_cols):
                        for j in range(i+1, num_cols):
                            regularization_loss += torch.sum(torch.pow(param[i,:] - param[j:], 2))

                    # for i, param1 in enumerate(param_list):
                    #     for param2 in param_list[i+1:]:
                    #         regularization_loss += torch.sum(torch.pow(param1 - param2, 2))

                loss += lambda_reg * regularization_loss
                loss.backward()
                optimizer.step()

            if (epoch+1) % 5  == 0:
                train_iat, train_broad = self.get_iat_broad_stats(train_cohort,
                                                                  train_resist_prescrip_df,
                                                                  avoid_last=avoid_last)
                val_iat, val_broad = self.get_iat_broad_stats(val_cohort,
                                                              val_resist_prescrip_df,
                                                              avoid_last=avoid_last)

                self.evaluate_checkpoints[epoch + 1] = (val_iat, val_broad)

                logging.info(f'Finished with epoch {epoch + 1}')

                logging.info(f'Train IAT: {train_iat[0]}, Train 2nd line: {train_broad[0]}')
                logging.info(f'Val IAT: {val_iat[0]}, Val 2nd line: {val_broad[0]}')

                train_cohort_actions = self.get_actions(train_cohort, avoid_last=avoid_last)
                val_cohort_actions = self.get_actions(val_cohort, avoid_last=avoid_last)

                train_rewards_merged = train_rewards_df.merge(train_cohort_actions,
                                   on='example_id', how='inner')

                val_rewards_merged = val_rewards_df.merge(val_cohort_actions,
                                    on='example_id', how='inner')


                train_action_probs = self.model(train_features_tensor.float())
                val_action_probs = self.model(val_features_tensor.float())

                # mean_train_reward = train_rewards_merged.apply(lambda x: x[x['rec']], axis=1).mean()
                mean_train_reward = torch.mean(torch.sum(train_action_probs * train_rewards_tensor, axis=1)).item()
                logging.info(f'Mean train reward: {mean_train_reward}')

                # mean_val_reward = val_rewards_merged.apply(lambda x: x[x['rec']], axis=1).mean()
                mean_val_reward = torch.mean(torch.sum(val_action_probs * val_rewards_tensor, axis=1)).item()
                logging.info(f'Mean val reward: {mean_val_reward}')

            # if (epoch + 1) % 25 == 0:
            #     torch.save(self.model.state_dict(),
            #                f"experiment_results/{self.exp_name}/models/{self.desc}_{epoch+1}_epochs.pth")


        torch.save(self.model.state_dict(),
                   f"experiment_results/direct/{self.exp_name}/models/{self.desc}_final.pth")

