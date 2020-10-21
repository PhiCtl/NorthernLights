# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

from utils import compute_loss, build_poly
from implementations import *
from plots import *


#---------------------------------CROSS VALIDATION UTILS-----------------------------------------------------------#
def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)



#---------------- GENERAL CROSS VALIDATION------------------------------------#

def cross_validation(y, x, k_indices, k, lambda_, degree, method):
    """Returns loss of choosen method for training and test sets."""

    
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
   
    # apply method to get w_opt and compute loss for train and test data 
    w_opt, loss_tr, l_type = choose_your_methods(method, y_tr, x_augm_tr, lambda_) 
    #loss depends on choosen method
    
    #if we're dealing with least squares GD, SGD, or normal equations, 
    #we don't need to compute the L2- regularization
    if (method == 5) or (method == 6):
        loss_te =compute_loss(y_test, x_augm_test, w_opt, loss_type = l_type, lbd = lambda_)
                                          
     
    #compute the right loss
    loss_te =compute_loss(y_test, x_augm_test, w_opt, loss_type = l_type)
    
    return loss_tr, loss_te



def select_best_degree(y, x, method, seed = 1, k_fold = 4, degrees = np.arange(1,10,1), lambdas = np.logspace(-20,0,5), screening_plot = False, variance_plot = False, verbose = False):
    
    """Returns best degree based on loss comparisons across lambdas (k-folds cross validation)"""
    """Returns also associated lambda and RMSE loss"""
    
    
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    
    # define lists to store the loss of training data and test data
    rmse_tr = []
    rmse_te = []
    best_lambdas = []
    
    #define array to store loss along degrees (for plotting)
    rmse_te_plot = np.empty((len(degrees), len(lambdas)))
    
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
                loss_tr, loss_te = cross_validation(y, x, k_indices, k, lambda_, deg, method)
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
        


def select_best_lambda(y, x, method, seed = 1, k_fold = 10, degrees = np.arange(1,10,1), lambdas = np.logspace(-10,0,5), screening_plot = False, variance_plot = False, verbose = False):
    
    """Returns best lambda across a degree range (based on smallest loss, depending on choosen method) and associated  loss"""
    
     # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    
    # define lists to store the loss of training data and test data
    rmse_tr = []
    rmse_te = []
    best_degrees = []
    
    #define array to store loss along degrees (for plotting)
    rmse_te_plot = np.empty((len(lambdas),len(degrees)))
    
    # k-fold cross validation: loop for each degree on each lambda on the k folds
    for l, lambda_ in enumerate(lambdas):
       #temporary lists for test and training losses for each lambda
        rmse_tr_l = []
        rmse_te_l = []
            
        for d, deg in enumerate(degrees):
            #temporary lists for test and training losses for each k_fold
            rmse_tr_k = []
            rmse_te_k = []
            
            for k in range(k_fold):
                #get rmse losses for test and training data, 
                #for ridge_regression with hyperparams (lambda_, degree)
                loss_tr, loss_te = cross_validation(y, x, k_indices, k, lambda_, degree, method)
                rmse_tr_k.append(loss_tr)
                rmse_te_k.append(loss_te)
                
            #mean of the loss on the k folds for each lambda    
            rmse_tr_l.append(np.mean(rmse_tr_k))
            rmse_te_l.append(np.mean(rmse_te_k))
            rmse_te_plot[l,d] = rmse_te_l[-1]
                           
        
        #select best lambda for each degree
        ind_best_degree = np.argmin(rmse_te_l)
        best_degrees.append(degrees[ind_best_degree])
        
        #remember best lambda loss
        rmse_tr.append(rmse_tr_l[ind_best_degree])
        rmse_te.append(rmse_te_l[ind_best_degree])
        
        if verbose:
            #print degree loss
            print("Current lambda={lbd}, loss={l}, best degree={deg}".format(lbd =lambda_, l=rmse_te[-1], deg = best_degrees[-1]))
    
    if screening_plot:
        cross_validation_visualization(degrees, rmse_tr, rmse_te, 'degree')
        
    if variance_plot:
        #plot RMSE variance for each degree
        plot_variance(rmse_te_plot, 'degree')
                            
                            
    #find best lambda
    ind_min = np.argmin(rmse_te)
    best_lambda = lambda_[ind_min]
    print("Best lambda ={lbd}, loss for k-folds cross validation={l}, best degree={deg}".format(lbd =best_lambda, l=rmse_te[ind_min], deg = best_degrees[ind_min]))
                            
    return best_lambda, rmse_te[ind_min], best_degrees[ind_min], rmse_te_plot

#-----------------------------------------------------------------------------------------------------#

def choose_your_methods(method, y_tr, tx_tr, lambda_, gamma = 0.000001, max_iters = 200, batch_size = 1):
    
   # create initial w for methods using it
        initial_w = np.zeros(tx_tr.shape[1])

        if method == 1:
            # Use least squares method
            w, loss = least_squares(y_tr,tx_tr)
            return w, loss, 'RMSE'
            
        if method == 2:
            # Use least squares GD
            w, loss = least_squares_GD(y_tr, tx_tr, initial_w,max_iters,gamma)
            return w, loss, 'RMSE'
            
        if method == 3:
            # Use least squares SGD
            w, loss = least_squares_SGD(y_tr, tx_tr, initial_w, batch_size, max_iters, gamma)
            return w, loss, 'RMSE'
            
        if method == 4:
            # Use ridge regression
            w, loss =ridge_regression(y_tr, tx_tr, lambda_)
            return w, loss, 'RMSE'
            
        if method == 5:
            # Use logistic regression
            w, loss = logistic_regression(y_tr, tx_tr, initial_w, max_iters, gamma)
            return w, loss, 'logREG'
            
        if method == 6:
            # Use regularized logistic regression
            w, loss = reg_logistic_regression(y_tr, tx_tr,initial_w, gamma, max_iters, lambda_)
            return w, loss, 'logREG'

