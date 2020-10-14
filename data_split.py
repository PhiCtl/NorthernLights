# -*- coding: utf-8 -*-
import numpy as np

"Generates train and test data from initial dataset"

def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8 
    you will have 80% of your data set dedicated to training 
    and the rest dedicated to testing
    """
    #setting seed ensures reproducibility
    np.random.seed(seed)
    
    #shuffle vectors x and y
    shuffle_indices = np.random.permutation(len(y))
    shuffled_y = y[shuffle_indices]
    shuffled_x = x[shuffle_indices]
    
    #splitting data
    split_i = int(len(y) *ratio)
    
    y_training = shuffled_y[:split_i]
    y_test = shuffled_y[split_i:]
    x_training = shuffled_x[:split_i]
    x_test = shuffled_x[split_i:]
   
    return y_training, x_training, y_test, x_test

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)
