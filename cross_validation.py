# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

from utils import compute_loss, build_poly, accuracy
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

def cross_validation(y, x, k_indices, k, lambda_, degree, gamma, method):
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
    w_opt, loss_tr, l_type = choose_your_methods(method, y_tr, x_augm_tr, lambda_, gamma)
    #loss depends on choosen method
    
    #compute accuracy
    acc = accuracy(y_test, x_augm_test.dot(w_opt))
    
    #if we're dealing with least squares, 
    #we don't need to compute the L2- regularization
    if (method == 1) or (method == 2) or(method == 3):
        loss_te =compute_loss(y_test, x_augm_test, w_opt, loss_type = l_type)
    
    loss_te =compute_loss(y_test, x_augm_test, w_opt, loss_type = l_type, lbd = lambda_)
    
    
    return loss_tr, loss_te, acc


def select_best_degree(y, x, method, by_accuracy, seed, k_fold, degrees, lambdas, gamma, screening_plot = False, verbose = False):
    
    """Returns best degree based on loss comparisons across lambdas (k-folds cross validation)"""
    """Returns also associated lambda and test loss"""
    
    
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    
    # define lists to store the loss of training data and test data, and accuracies
    loss_tr = []
    loss_te = []
    best_lambdas = []
    accuracy_plot = np.empty((len(degrees), len(lambdas)))
    accuracy_te = []
    
    # k-fold cross validation: loop for each degree on each lambda on the k folds
    for d, deg in enumerate(degrees):
       #temporary lists for test and training losses for each lambda
        loss_tr_l = []
        loss_te_l = []
            
        for l, lambda_ in enumerate(lambdas):
            #temporary lists for test and training losses for each k_fold
            loss_tr_k = []
            loss_te_k = []
            
            for k in range(k_fold):
                #get losses for test and training data of the k_fold, 
                acc_tot = 0
                l_tr, l_te, acc = cross_validation(y, x, k_indices, k, lambda_, deg, gamma, method)
                loss_tr_k.append(l_tr)
                loss_te_k.append(l_te)
                acc_tot += acc
                
            #mean of the loss on the k folds for each lambda    
            loss_tr_l.append(np.mean(loss_tr_k))
            loss_te_l.append(np.mean(loss_te_k))
            
            #compute mean accuracy
            accuracy_plot[d,l] = acc_tot/k_fold
                           
  
        if by_accuracy:
            #select best lambda for each degree -> the one with biggest accuracy
            ind_best_lambda = np.argmax(accuracy_plot[d,:])
        else:
            #select best lambda for each degree -> the one with smallest loss
            ind_best_lambda = np.argmin(loss_te_l)
            
        #remeber best lambda and accuracy
        best_lambdas.append(lambdas[ind_best_lambda])
        accuracy_te.append(accuracy_plot[d,ind_best_lambda])
            
        #remember best lambda loss
        loss_tr.append(loss_tr_l[ind_best_lambda])
        loss_te.append(loss_te_l[ind_best_lambda])
        
        
        if verbose:
            print("Current degree={degree}, loss={l}, best lambda={lbd}".format(degree=deg, l=loss_te[-1], lbd = best_lambdas[-1]))
    
    if screening_plot:
        cross_validation_visualization(degrees, loss_tr, loss_te, 'degree')
                            
    #find best degree                        
    if by_accuracy:
        ind = np.argmax(accuracy_te)
    else:
        ind = np.argmin(loss_te)
    best_degree = degrees[ind]
        
        
    print("Best degree ={degree}, loss for k-folds cross validation={l}, best lambda={lbd}, accuracy={a}".format(degree=best_degree, l=loss_te[ind], lbd = best_lambdas[ind], a = accuracy_te[ind]))
                            
    return best_degree, accuracy_te[ind], best_lambdas[ind], accuracy_plot



def select_best_lambda(y, x, method, seed, k_fold, degrees, lambdas, gamma, screening_plot = False, verbose = False):
    
    """Returns best lambda across a degree range (based on smallest loss, depending on choosen method) and associated loss"""
    
     # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    
    # define lists to store the loss of training data and test data
    loss_tr = []
    loss_te = []
    best_degrees = []
    
    # k-fold cross validation: loop for each lambda on each degree on the k folds
    for l, lambda_ in enumerate(lambdas):
       #temporary lists for test and training losses for each lambda
        loss_tr_l = []
        loss_te_l = []
            
        for d, deg in enumerate(degrees):
            #temporary lists for test and training losses for each k_fold
            loss_tr_k = []
            loss_te_k = []
            
            for k in range(k_fold):
                #get losses for test and training data, 
                
                l_tr, l_te = cross_validation(y, x, k_indices, k, lambda_, degree, gamma, method)
                loss_tr_k.append(l_tr)
                loss_te_k.append(l_te)
                
            #mean of the loss on the k folds for each lambda    
            loss_tr_l.append(np.mean(loss_tr_k))
            loss_te_l.append(np.mean(loss_te_k))
                           
        
        #select best lambda for each degree
        ind_best_degree = np.argmin(loss_te_l)
        best_degrees.append(degrees[ind_best_degree])
        
        #remember best lambda loss
        loss_tr.append(loss_tr_l[ind_best_degree])
        loss_te.append(loss_te_l[ind_best_degree])
        
        if verbose:
            #print degree loss
            print("Current lambda={lbd}, loss={l}, best degree={deg}".format(lbd =lambda_, l=loss_te[-1], deg = best_degrees[-1]))
    
    if screening_plot:
        cross_validation_visualization(degrees, loss_tr, loss_te, 'degree')
                            
    #find best lambda
    ind_min = np.argmin(loss_te)
    best_lambda = lambda_[ind_min]
    print("Best lambda ={lbd}, loss for k-folds cross validation={l}, best degree={deg}".format(lbd =best_lambda, l=loss_te[ind_min], deg = best_degrees[ind_min]))
                            
    return best_lambda, loss_te[ind_min], best_degrees[ind_min]

#-----------------------------------------------------------------------------------------------------#

def choose_your_methods(method, y_tr, tx_tr, lambda_, gamma, max_iters = 700, batch_size = 1):
    
        """
        Methods:
        1 : Least squares
        2 : Least squares gradient descent
        3 : least squares stochastic gradient descent
        4 : Ridge regression
        5 : Logstic regression
        6 : Regularized logistic regression
        """
    
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
            w, loss = least_squares_SGD(y_tr, tx_tr, batch_size, max_iters, gamma)
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

