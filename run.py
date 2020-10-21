# -*- coding: utf-8 -*-

import numpy as np
from cross_validation_phi import select_best_degree, select_best_lambda
from utils import combine jets, accuracy_2, build_poly

#------------------------RUN FUNCTION---------------------------------------------------#

def best_w(y,x,method,best_lambda,best_deg):
    
    tx_tr_opt = build_poly(x,best_deg)
    
    # Compute optimal weight
    initial_w = np.zeros((tx_tr_opt.shape[1],))
    w_opt,_=choose_your_methods(method, y, tx_tr_opt, best_lambda, gamma, max_iters, batch_size)
    
    return w_opt

def compute_accuracy(y_test,jet_list,index_te,w_opt_list):
    
    #compute y_pred for each jet 
    for jet in jet_list :
        for w in w_opt_list:
            y_pred = jet.dot(w)
            y_pred_list.append(y_pred)
    
    y_predict = combine_jets(y_pred_list, index_te)

    return accuracy_2(y_test, y_predict)

def select_best_parameter(y, x, method, param_type, sd = 1 , k_fld = 10, deg = np.arange(1,10,1), lbdas = np.logspace(-10,0,5)):
    
    print("For method n°:{n}".format(n = method))
    
    if param_type == 'degree':
        
        return select_best_degree(y, x, method, seed = sd, k_fold = k_fld, degrees = deg, lambdas = lbdas)
    
    if param_type == 'lambda':
        return select_best_lambda(y, x, method, seed = sd, k_fold = k_fld, degrees = deg, lambdas = lbdas)