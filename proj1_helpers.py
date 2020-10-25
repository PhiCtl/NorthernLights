# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np
from cross_validation import select_best_degree, select_best_lambda, choose_your_methods
from utils import accuracy, build_poly, predict_labels

#-----------------------------LOAD DATA---------------------------------------------------------------------------#
def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y=='b')] = -1
    
    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids



#-----------------------------GET JETS FOR FINAL TEST SET------------------------------------------------------------#

def extract_jet(x, categ_ind, jet_num, indices):
    """Extract data for which PRI_jet_num = jet_num and remove undefined features"""
    
    #data points for which jet = jet_num
    ind = np.nonzero(x[:,categ_ind] == jet_num)
    jet = x[ np.nonzero(x[:,categ_ind] == jet_num)] 
    #remove undefined features (indices) for those datapoints
    if(indices):
        jet = np.delete(jet, indices, 1)
    return jet, ind


def get_jets_final(x, cat_ind, undefined_features, list_ = True):
    
    """Get jets for final test set"""
    jet0, ind_0 = extract_jet(x, cat_ind, 0, undefined_features[0])
    jet1, ind_1 = extract_jet(x, cat_ind, 1, undefined_features[1])
    jet2, ind_2 = extract_jet(x, cat_ind, 2, undefined_features[2])
    jet3, ind_3 = extract_jet(x, cat_ind, 3, undefined_features[3])
    
    ind_list = [ind_0, ind_1, ind_2, ind_3]
    
    if list_:
        return [jet0, jet1, jet2, jet3], ind_list
    else:
        return jet0, jet1, jet2, jet3, ind_list

def combine_jets(y_jets, indices):
    
    #build y 
    n = 0
    for jet in y_jets:
        n += len(jet)
    y = np.zeros((n,)) #don't write (n,1) it doesn't like it... 
    
    #fill in y with corresponding y_jets in order
    for i, ind in enumerate(indices):
        y[ind] = y_jets[i]
    
    return y

#----------------------------------------RUN FUNCTIONS----------------------------------------------------------#
def best_w(y,x,method,best_lambda,best_deg, gamma = 0.00001):
    """Returns the optimal weights for choosen method, with optimal lambda, degree and gamma
    Parameters:
    - prediction y training
    - method
    - training data matrix x
    - best_lambda from cross validation
    - best_deg: best degree from cross validation
    - gamma: default value from experimentation
    
    
    Method flags:
    1 : Least squares
    2 : Least squares gradient descent (least squares GD)
    3 : Least squares stochastic gradient descent (least squares SGD)
    4 : Ridge regression
    5 : Logistic regression
    6 : Regularized logistic regression
    """
    
    tx_tr_opt = build_poly(x,best_deg)
    
    # Compute optimal weight 
    initial_w = np.zeros((tx_tr_opt.shape[1],))
    w_opt,_,_=choose_your_methods(method, y, tx_tr_opt, best_lambda, gamma)
    
    return w_opt


def select_best_parameter(y, x, method, param_type,  by_accuracy = True, seed = 1 , k_fold = 5, degrees = np.arange(1,10,1), lambdas = np.logspace(-20,-10,3), gamma = 0.0000001 ):
    """Returns the best parameter (either 'lambda' or 'degree') for a given method (see methode coding in readme)
    
    Input parameters:
    
    y               : train set prediction vector
    x               : train set data
    method          : 1, 2, 3, 4, 5, 6 (see method coding above)
    param_type      : degree or lambda
    by_accuracy     : if True,  returns parameters that yield the best accuracy for kfold cross validation
                      if False, returns parameters that minimize the loss for a given method
    seed            : set to 1
    k_fold          : 5 folds were chosen
    degrees         : degree range on which the methods are evaluated
    lambdas         : lambda range on which the methods are evaluated -> MUST BE SET TO NP.ARRAY([0]) FOR METHODS WITHOUT REGULARIZATION
    gamma           : for methods implying a gradient descent"""
    
    print("For method n°:{n}".format(n = method))
    
    if param_type == 'degree':
        return select_best_degree(y, x, method, by_accuracy, seed, k_fold, degrees, lambdas, gamma)
    
    if param_type == 'lambda':
        return select_best_lambda(y, x, method, by_accuracy, seed, k_fold, degrees, lambdas, gamma)
    print('Please select a parameter')

#---------------------------------------CREATE SUBMISSION FILE---------------------------------------------------#

def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})
