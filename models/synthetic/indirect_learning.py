from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import numpy as np

def train_outcome_model_for_action(X, y):

    clf_penalty_dict = {}
    for penalty in ['l1', 'l2']:
        clf_best_for_penalty = LogisticRegressionCV(cv=10, 
                                       random_state=15,
                                       penalty=penalty,
                                       solver='saga',
                                       scoring='roc_auc',
                                       Cs=[0.001, 0.005, 0.01, 0.05, 
                                           0.1, 0.5, 1, 10]).fit(X, y)

        clf_penalty_dict[penalty] = clf_best_for_penalty
        
    l1_model = clf_penalty_dict['l1']
    l2_model = clf_penalty_dict['l2']

    l1_max_score = l1_model.scores_[1].mean(axis=0).max()
    l2_max_score = l2_model.scores_[1].mean(axis=0).max()
    
    return l1_model if l1_max_score >= l2_max_score else l2_model
        

def train_all_outcome_models(X, y_all):
    action_model_dict = {}
    num_actions = y_all.shape[1]
    
    for i in range(num_actions):
        clf_best = train_outcome_model_for_action(X, y_all[:,i])
        penalty, C = clf_best.penalty, clf_best.C_[0]
            
        clf = LogisticRegression(penalty=penalty, C=C, solver='saga')
        clf.fit(X, y_all[:,i])
        action_model_dict[i] = clf
    
    return action_model_dict


def get_test_preds(X_test, model_dict):
    preds_all = []
    for action, model in model_dict.items():
        preds_all.append(model.predict_proba(X_test)[:,1])
    
    return np.array(preds_all).T


def run_indirect_policy_learning(X_train_raw, X_test_raw, y_train, y_test):
    action_model_dict = train_all_outcome_models(X_train_raw, y_train)
    test_preds = get_test_preds(X_test_raw, action_model_dict)
    chosen_actions = np.argmax(test_preds, axis=1)

    return np.mean(y_test[np.arange(len(y_test)), chosen_actions])
