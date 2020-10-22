# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from utils import build_poly, compute_loss
from implementations import least_squares_GD, ridge_regression
from cross_validation import build_k_indices
from plots import *

#--------------------------CROSS VALIDATION LEAST SQUARES : FIND BEST DEGREE ----------------------------------------#

def cross_validation_LS_GD(y, x, k_indices, k, degree, gamma, max_iters = 800):
    """Returns RMSE of simple least_squares regression using gradient descent"""
    
    #to be tuned: gamma 
    
    #get the indices
    k_te = k_indices[k]
    k_tr = k_indices[k_indices != k_te]
    
    # get k'th subgroup in test, others in train: 
    y_test = y[k_te]
    x_test = x[k_te]
    y_tr   = y[k_tr]  
    x_tr   = x[k_tr]
    
    # form data with polynomial degree: 
    x_augm_tr = build_poly(x_tr, degree)
    x_augm_test = build_poly(x_test, degree)
    
    #parameters that are not to be tuned
    initial_w = np.zeros((x_augm_tr.shape[1],))
   
    # least squares gradient descent and calculate loss for train and test data (RMSE)
    w_opt, rmse_tr = least_squares_GD(y_tr, x_augm_tr, initial_w, max_iters, gamma) #loss is rmse here
    
    rmse_te = compute_loss(y_test, x_augm_test, w_opt, loss_type = 'RMSE')
 
    return rmse_tr, rmse_te

def select_best_degree_LSGD(y, x, seed = 1, k_fold = 4, degrees = np.arange(1,10,1), gammas = np.logspace(-7,-3,1), screening_plot = False, variance_plot = False, verbose = True):
    
    """Returns best degree based on comparison of RMSE across degrees, and associated RMSE"""
    #set gamma and k folds
    
    
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    
    # define lists to store the loss of training data and test data
    rmse_tr = []
    rmse_te = []
    rmse_te_plot=np.empty((len(degrees),len(gammas)))
    best_gammas =[]
    
    # k-fold cross validation: loop for each degree on each lambda on the k folds
    for d, deg in enumerate(degrees):
       #temporary lists for test and training losses for each lambda
        rmse_tr_l = []
        rmse_te_l = []
            
        for g, gamma in enumerate(gammas):
            #temporary lists for test and training losses for each k_fold
            rmse_tr_k = []
            rmse_te_k = []
            
            for k in range(k_fold):
                #get rmse losses for test and training data, 
                #for least squares GD with hyperparams (gamma, degree)
                loss_tr, loss_te = cross_validation_LS_GD(y, x, k_indices, k, deg, gamma)
                rmse_tr_k.append(loss_tr)
                rmse_te_k.append(loss_te)
                
            #mean of the loss on the k folds for each gamma    
            rmse_tr_l.append(np.mean(rmse_tr_k))
            rmse_te_l.append(np.mean(rmse_te_k))
            rmse_te_plot[d,g] = rmse_te_l[-1]
                           
        
        #select best lambda for each degree
        ind_best_g = np.argmin(rmse_te_l)
        best_gammas.append(gammas[ind_best_g])
        
        #remember best lambda loss
        rmse_tr.append(rmse_tr_l[ind_best_g])
        rmse_te.append(rmse_te_l[ind_best_g])
        
        if verbose:                  
            print("Current degree={degree}, loss={l}, gamma={g}".format(degree=deg, l=rmse_te[-1], g = gammas[-1] ))
    
    #if loss went too high (->inf or nan) because of divergence, ignore it
    rmse_tr = np.ma.masked_array(rmse_tr, np.isnan(rmse_tr)) 
    rmse_te = np.ma.masked_array(rmse_te, np.isnan(rmse_te))
    
    if screening_plot:
        #visualize rmse_te versus degrees
        cross_validation_visualization(degrees, rmse_tr, rmse_te, 'degree')
   
    if variance_plot:
        #plot RMSE variance for each degree
        plot_variance(rmse_te_plot, 'degree')
    
    #find best degree
    ind_min = np.argmin(rmse_te)
    best_degree = degrees[ind_min]
    print("Best degree={deg}, best_gamma={g}, smallest RMSE={rmse}".format(deg = best_degree, rmse = rmse_te[ind_min], g = best_gammas[ind_min]))
    
    return best_degree, rmse_te[ind_min], best_gammas[ind_min]

#--------------------------CROSS VALIDATION RIDGE-REGRESSION : FIND BEST DEGREE FOR BEST LAMBDA ----------------------------------------#

def cross_validation_ridge(y, x, k_indices, k, lambda_, degree):
    """Returns RMSE loss of simple ridge regression for training and test sets."""

    
    #get the right indices:
    k_te = k_indices[k]
    k_te = k_indices[k]
    k_tr = k_indices[k_indices != k_te]
    
    # get k'th subgroup in test, others in train: 
    y_test = y[k_te]
    x_test = x[k_te]
    y_tr   = y[k_tr]  
    x_tr   = x[k_tr]
    
    # form data with polynomial degree: 
    x_augm_tr = build_poly(x_tr, degree)
    x_augm_test = build_poly(x_test, degree)
   
    # ridge regression and calculate loss for train and test data (RMSE)
    w_opt, loss_tr = ridge_regression(y_tr, x_augm_tr, lambda_) #loss is rmse here
    loss_te =compute_loss(y_test, x_augm_test, w_opt, loss_type = 'RMSE', lbd = lambda_)
    
    return loss_tr, loss_te

def select_best_degree_ridge(y, x, seed = 1, k_fold = 4, degrees = np.arange(5,10,1), lambdas = np.logspace(-20,-10,1), screening_plot = True, variance_plot = False, verbose = False):
    
    """Returns best degree based on RMSE loss comparisons across lambdas (k-folds cross validation)"""
    """Returns also associated lambda and RMSE loss"""
    
    
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    
    # define lists to store the loss of training data and test data
    rmse_tr = []
    rmse_te = []
    best_lambdas = []
    
    #define array to store loss along degrees
    rmse_te_plot = np.empty((len(degrees),len(lambdas)))
    
    # k-fold cross validation: loop for each degree on each lambda on the k folds
    for d, deg in enumerate(degrees):
       #temporary lists for test and training losses for each lambda
        rmse_tr_l = []
        rmse_te_l = []
            
        for l, lambda_ in enumerate(lambdas):
            #temporary lists for test and training losses for each k_fold
            rmse_tr_k = []
            rmse_te_k = []
            
            for k in range(k_fold):
                #get rmse losses for test and training data, 
                #for ridge_regression with hyperparams (lambda_, degree)
                loss_tr, loss_te = cross_validation_ridge(y, x, k_indices, k, lambda_, deg)
                rmse_tr_k.append(loss_tr)
                rmse_te_k.append(loss_te)
                
            #mean of the loss on the k folds for each lambda    
            rmse_tr_l.append(np.mean(rmse_tr_k))
            rmse_te_l.append(np.mean(rmse_te_k))
            rmse_te_plot[d,l] = rmse_te_l[-1]
                           
        
        #select best lambda for each degree
        ind_best_lambda = np.argmin(rmse_te_l)
        best_lambdas.append(lambdas[ind_best_lambda])
        
        #remember best lambda loss
        rmse_tr.append(rmse_tr_l[ind_best_lambda])
        rmse_te.append(rmse_te_l[ind_best_lambda])
        
        if verbose:
            #print degree loss
            print("Current degree={degree}, loss={l}, best lambda={lbd}".format(degree=deg, l=rmse_te[-1], lbd = best_lambdas[-1]))
    
    if screening_plot:
        cross_validation_visualization(degrees, rmse_tr, rmse_te, 'degree')
        
    if variance_plot:
        #plot RMSE variance for each degree
        plot_variance(rmse_te_plot, 'degree')
                            
                            
    #find best degree
    ind_min = np.argmin(rmse_te)
    best_degree = degrees[ind_min]
    print("Best degree ={degree}, loss for k-folds cross validation={l}, best lambda={lbd}".format(degree=best_degree, l=rmse_te[ind_min], lbd = best_lambdas[ind_min]))
                            
    return best_degree, rmse_te[ind_min], best_lambdas[ind_min], rmse_te_plot