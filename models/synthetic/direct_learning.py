import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
import torch.optim as optim

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

sys.path.append('../../')
from models.direct.policy_net import PolicyNetLinear
from models.direct.policy_net import policy_loss


def train_direct_policy(X_train_all, y_train_all,
                        num_epochs=50, lr=0.1,
                        lambda_reg=0, split_val=True):
        
        model = PolicyNetLinear(num_inputs=10, num_outputs=3)
        
        if split_val:
            X_train, X_val, y_train, y_val = train_test_split(X_train_all, y_train_all, 
                                                            test_size=0.2, random_state=10)
        
            X_train_tensor = torch.from_numpy(X_train)
            y_train_tensor = torch.from_numpy(y_train)

            X_val_tensor = torch.from_numpy(X_val)
            y_val_tensor = torch.from_numpy(y_val)
         
        else:
            X_train, y_train = X_train_all, y_train_all
            X_train_tensor = torch.from_numpy(X_train)
            y_train_tensor = torch.from_numpy(y_train)
            
        train_dataset = data_utils.TensorDataset(X_train_tensor.float(), y_train_tensor.float())
        train_loader = data_utils.DataLoader(train_dataset, batch_size=64, shuffle=True)
        optimizer = optim.SGD(model.parameters(), lr=lr)
       
        for epoch in range(num_epochs):
            for feats, rewards in train_loader:
                optimizer.zero_grad()
                output = model(feats)
                
                # Loss from policy distribution
                loss = policy_loss(output, rewards) 
                
                # L2 regularization 
                regularization_loss = 0
                for name, param in model.named_parameters():
                    if 'bias' not in name:
                        regularization_loss += torch.sum(torch.pow(param,2))
                
                loss += lambda_reg * regularization_loss
                loss.backward()
                optimizer.step()
            
            train_preds = model(X_train_tensor.float())
            train_actions = np.argmax(train_preds.detach().numpy(), axis=1)
            train_outcomes = y_train[np.arange(len(y_train)), train_actions]
            # print(f'Mean train outcome: {np.mean(train_outcomes)}')
            
            if split_val:
                val_preds = model(X_val_tensor.float())
                val_actions = np.argmax(val_preds.detach().numpy(), axis=1)
                val_outcomes = y_val[np.arange(len(y_val)),
                                     val_actions]
                print(f'Mean val outcome: {np.mean(val_outcomes)}')
        
        return model
    

def evaluate_direct_model(model, X_test, y_test):
    X_test_tensor = torch.from_numpy(X_test).float()
    test_preds = model(X_test_tensor).detach().numpy()
    
    test_actions = np.argmax(test_preds, axis=1)
    test_outcomes = y_test[np.arange(len(y_test)), test_actions]
    
    return np.mean(test_outcomes)


def run_direct_policy_learning(X_train, X_test, y_train, y_test,
                               lr=0.1, num_epochs=50,
                               lambda_reg=0, split_val=False):
    model = train_direct_policy(X_train, y_train,
                               lr=lr, num_epochs=num_epochs,
                               lambda_reg=lambda_reg, split_val=split_val)
    return evaluate_direct_model(model, X_test, y_test)

