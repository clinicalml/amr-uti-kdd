import numpy as np
import pandas as pd

def generate_synthetic_features(n_samples, n_dims=10):
    '''
        Generates specified number of samples of synthetic data,
        where each data point has specificied no. of dimensions, each
        drawn i.i.d from a standard Gaussian.

        Returns the true underlying data values, along with the nonlinear transformations
        of the data used to compute the outcome model.
    '''
    raw_feats = np.random.normal(size=(n_samples, n_dims), scale=1)
    feats_processed = np.hstack([
        raw_feats[:,:-2]**2,
        raw_feats[:,0:1] * raw_feats[:,1:2],
        raw_feats[:,3:4] * raw_feats[:,5:6],
        raw_feats[:,4:5] * raw_feats[:,7:8],
        raw_feats[:,-2:]
    ])
    
    return raw_feats, feats_processed


def generate_model_coeffs():    
    # Coefficients for features not involved in decision. We choose these coeffs 
    # in a way that ensures the average model output is somewhere around 0.5
    random_coeffs = np.array([[8, -8, -5, 5, 10, -10, 8, -8, 7, -7, 9.5]])
    
    # Constrain these coefficients to be same across outcome models for all actions
    random_coeffs = np.repeat(random_coeffs, 3, axis=0)
    
    # Coefficients for features used in decision making
    fixed_coeffs = [[1,0], [0,1], [0,0]]

    return np.hstack([random_coeffs, fixed_coeffs])


def generate_labels(features, coeffs):
    raw_outcomes = 1/(1 + np.exp(-1*np.dot(features, coeffs.T)))
    labels = np.random.binomial(1, p=raw_outcomes)
    
    return labels


def generate_data(n_samples, coeffs, seed=49):
    np.random.seed(seed)

    X_raw, X = generate_synthetic_features(n_samples)
    labels = generate_labels(X, coeffs)
    
    return X_raw, X, labels


def get_best_action(row):
    row_len = len(row)
    if row[row_len-1] < 0 and row[row_len-2] < 0:
        return 2
    elif row[row_len-2] > row[row_len-1]:
        return 0
    elif row[row_len-1] > row[row_len-2]:
        return 1


def get_bayes_outcome(X_raw, y):
    X_df = pd.DataFrame(X_raw)
    optimal_actions = X_df.apply(get_best_action, axis=1)
    return y[np.arange(len(y)), optimal_actions.values].mean()
