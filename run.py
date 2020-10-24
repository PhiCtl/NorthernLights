# -*- coding: utf-8 -*-

import numpy as np
from cross_validation import select_best_degree, select_best_lambda, choose_your_methods
from utils import accuracy_2, build_poly
from proj1_helpers import *
from data_preprocessing import *

#------------------------RUN FUNCTION---------------------------------------------------#


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

def compute_accuracy(y_test,jet_list,index_te,w_opt_list):
    
    #compute y_pred for each jet 
    for jet in jet_list :
        for w in w_opt_list:
            y_pred = jet.dot(w)
            y_pred_list.append(y_pred)
    
    y_predict = combine_jets(y_pred_list, index_te)

    return accuracy_2(y_test, y_predict)

def select_best_parameter(y, x, method, param_type, seed = 1 , k_fold = 5, degrees = np.arange(1,10,1), lambdas = np.logspace(-20,-10,3), gamma = 0.0000001 ):
    
    print("For method n°:{n}".format(n = method))
    
    if param_type == 'degree':
        
        return select_best_degree(y, x, method, seed, k_fold, degrees, lambdas, gamma)
    
    if param_type == 'lambda':
        return select_best_lambda(y, x, method, seed, k_fold, degrees, lambdas, gamma)
    
def run():
   
    #load data
    y, tX, ids = load_csv_data('data/train.csv')
    #train set and test set
    y_tr, x_tr, y_te, x_te = split_data(tX, y, 0.8, seed=20)
    
    #categorical extraction
    undefined_features = [[4, 5, 6, 12, 22, 23, 24, 25, 26,
                       27, 28, 29], [4, 5, 6, 12, 22, 26, 27, 28], [22], [22]]
    PRI_jet_num = 22
    jet_train, y_jet_train, index = get_jets(x_tr, y_tr, PRI_jet_num, undefined_features, list_ = True)
    jet_test, y_jet_test, index_te = get_jets(x_te, y_te, PRI_jet_num, undefined_features, list_ = True)
    
    #data preprocessing
    jet_process_tr = []
    jet_process_te = []
    for jet in jet_train:
        jet_process_tr.append(preprocessing_data(jet, False, False, False))
    for jet in jet_test:
        jet_process_te.append(preprocessing_data(jet, False, False, False))
    
    best_degs = []
    best_lbds = []
    w_list = []
    

    #hyperparameter screening for best method (4: Ridge)
    #we selected a special range for degrees and lambdas we came up with after several trials
    for (jet, y_jet, index) in zip(jet_process_tr, y_jet_train, np.arange(4)):
        print("Jet n°:", index)
        best_degree, rmse_te, best_lambda, _ = select_best_parameter(y_jet, jet, 4 , 'degree', seed = 200, k_fold = 5, degrees = np.arange(5,7,1), lambdas = [0.00999, 0.00001, 0.01])
        best_degs.append(best_degree)
        best_lbds.append(best_lambda)
        
        x_augm = build_poly(jet, best_degree)
        w = best_w(y_jet , jet , 4, best_lambda , best_degree)
        w_list.append(w)
    
    #compute accuracy on test set
    y_pred_list = []
    for(jet, w, deg) in zip(jet_process_te, w_list, best_degs):
        x_augm_t = build_poly(jet, deg)
        y_pred_ = x_augm_t.dot(w)
        y_pred_list.append(y_pred_)
    
    y_predict = combine_jets(y_pred_list, index_te)
    print("Ridge prediction: ",accuracy_2(y_te, y_predict))
    
    #generate submission file
    
    #load data
    _, tX_test, ids_test = load_csv_data('data/test.csv')
    jet_final, ind_list = get_jets_final(tX_test, 22, undefined_features)
    for jet in jet_final:
        jet = (preprocessing_data(jet, False, False, False))
        
    final_jet = []
    for (jet, w, deg) in zip(jet_final, w_list, best_degs):
        x_augm = build_poly(jet, deg )
        y_pred = x_augm.dot(w)
        final_jet.append(y_pred)
    
    y_final = combine_jets(final_jet, ind_list)
    Y = predict_labels_2(y_final)
    
    create_csv_submission(ids_test, Y, 'data/best_prediction_ridge.csv')
    
    