# -*- coding: utf-8 -*-

import numpy as np
from cross_validation_phi import select_best_degree, select_best_lambda
from utils import accuracy_2, build_poly
from proj1_helpers import combine_jets

#------------------------RUN FUNCTION---------------------------------------------------#

def best_w(y,x,method,best_lambda,best_deg, gamma = 0.00001):
    
    tx_tr_opt = build_poly(x,best_deg)
    
    # Compute optimal weight
    initial_w = np.zeros((tx_tr_opt.shape[1],))
    w_opt,_=choose_your_methods(method, y, tx_tr_opt, best_lambda, gamma)
    
    return w_opt

def compute_accuracy(y_test,jet_list,index_te,w_opt_list):
    
    #compute y_pred for each jet 
    for jet in jet_list :
        for w in w_opt_list:
            y_pred = jet.dot(w)
            y_pred_list.append(y_pred)
    
    y_predict = combine_jets(y_pred_list, index_te)

    return accuracy_2(y_test, y_predict)

def select_best_parameter(y, x, method, param_type, seed = 1 , k_fold = 10, degrees = np.arange(1,10,1), lambdas = np.logspace(-20,-10,3), gamma = 0.00001):
    
    print("For method n°:{n}".format(n = method))
    
    if param_type == 'degree':
        
        return select_best_degree(y, x, method, seed, k_fold, degrees, lambdas, gamma)
    
    if param_type == 'lambda':
        return select_best_lambda(y, x, method, seed, k_fold, degrees, lambdas, gamma)